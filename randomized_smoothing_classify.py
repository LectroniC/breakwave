## classify.py -- actually classify a sequence with DeepSpeech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
import argparse

import scipy.io.wavfile as wav

import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")
import DeepSpeech
import pandas as pd

try:
    import pydub
    import struct
except:
    print("pydub was not loaded, MP3 compression will not work")

from tf_logits import get_logits, get_logits_smooth
sys.path.append("DeepSpeech/util")
from util.preprocess import pmap
from util.text import wer, levenshtein
from attrdict import AttrDict

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

def process_decode_result(item):
    label, decoding, distance, loss = item
    sample_wer = wer(label, decoding)
    return AttrDict({
        'src': label,
        'res': decoding,
        'loss': loss,
        'distance': distance,
        'wer': sample_wer,
        'levenshtein': levenshtein(label.split(), decoding.split()),
        'label_length': float(len(label.split())),
    })

def calculate_report(labels, decodings, distances):
    r'''
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = pmap(process_decode_result, zip(labels, decodings, distances))

    total_levenshtein = sum(s.levenshtein for s in samples)
    total_label_length = sum(s.label_length for s in samples)

    # Getting the WER from the accumulated levenshteins and lengths
    samples_wer = total_levenshtein / total_label_length

    # Order the remaining items by their loss (lowest loss on top)
    #samples.sort(key=lambda s: s.loss)

    # Then order by WER (highest WER on top)
    samples.sort(key=lambda s: s.wer, reverse=True)

    return samples_wer, samples


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    parser.add_argument('--smooth_sigma', type=float,
                        required=True,
                        help="randomized smoothing sigma")
    parser.add_argument('--sample_num', type=int,
                        required=True,
                        help="sample number of randomized smoothing")
    parser.add_argument('--smooth_type', type=str,
                        required=True,
                        help="smoothing type: mean, median, majority")
    parser.add_argument('--labels_csv', type=str,
                        required=True,
                        help='location for labels.csv')
    parser.add_argument('--input_folder', type=str,
                        required=True,
                        help='location for the folder of the wav files')
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()


    decoded_list = []
    final_logits_list = []
    transcripts = []
    
    with open(args.labels_csv,'r') as file:
        lines = file.readlines()
        for line in lines:
            test_audio_path = line.split(',')[0].strip()
            test_audio_path = os.path.join(args.input_folder, "adv_"+test_audio_path)
            transcripts.append(line.split(',')[-2].strip())
            saver = tf.train.Saver()
            with tf.Session() as sess:
                # if args.input.split(".")[-1] == 'mp3':
                if test_audio_path.split(".")[-1] == 'mp3':
                    # raw = pydub.AudioSegment.from_mp3(args.input)
                    raw = pydub.AudioSegment.from_mp3(test_audio_path)
                    audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
                # elif args.input.split(".")[-1] == 'wav':
                elif test_audio_path.split(".")[-1] == 'wav':
                    _, audio = wav.read(test_audio_path)
                else:
                    raise Exception("Unknown file format")
                N = len(audio)
                print(audio.shape)
                np.set_printoptions(threshold=sys.maxsize)
                new_input = tf.placeholder(tf.float32, [1, N])
                lengths = tf.placeholder(tf.int32, [1])

                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    logits = get_logits(new_input, lengths)

                saver.restore(sess, args.restore_path)

                decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)

                print('logits shape', logits.shape)
                length = (len(audio)-1)//320
                l = len(audio)

                new_logits_ls = []
                for _ in range(args.sample_num):
                    noise = np.random.normal(0.0, args.smooth_sigma, size=audio.shape)
                    new_audio = audio + noise
                    output_logits, _ = sess.run((logits, decoded), {new_input: [new_audio], lengths: [length]})
                    new_logits_ls.append(output_logits)
                new_logits_ls_np = np.array(new_logits_ls)
                if args.smooth_type == 'mean':
                    logits_smooth = np.mean(new_logits_ls_np, axis=0)
                elif args.smooth_type == 'median':
                    logits_smooth = np.median(new_logits_ls_np, axis=0)
                else:
                    raise Exception("No implementation of this type of smoothing, please choose from mean, median, majority")
                
                final_logits_list.append(logits_smooth)

                # Run beam-search only
                final_logits_holder = tf.placeholder(tf.float32, logits.shape)
                final_decoded, _ = tf.nn.ctc_beam_search_decoder(final_logits_holder, lengths, merge_repeated=False, beam_width=500)
                # TODO: Fix here.
                r = sess.run((final_decoded), {final_logits_holder: logits_smooth, lengths: [length]})
                decoded_list.append(r)
                print(logits_smooth)
                print(r)

    predictions = []
    predictions.extend(["".join([toks[x] for x in d[0].values])] for d in decoded_list)

    ground_truths = []
    ground_truths.extend([l] for l in transcripts)

    distances = [levenshtein(a, b) for a, b in zip(ground_truths, predictions)]
    wer, samples = calculate_report(ground_truths, predictions, distances)
    mean_edit_distance = np.mean(distances)
    print('Test - WER: %f, CER: %f' %
          (wer, mean_edit_distance))

    # print("-"*80)
    # print("-"*80)

    # print("Classification:")
    # print("".join([toks[x] for x in r[0].values]))
    # print("-"*80)
    # print("-"*80)

main()
