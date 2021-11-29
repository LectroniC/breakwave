## randomized_smoothing_classify.py -- actually classify a sequence with DeepSpeech with randomized smoothing
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

try:
    import pydub
    import struct
except:
    print("pydub was not loaded, MP3 compression will not work")

from tf_logits import get_logits, get_logits_smooth


# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"



def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input",
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
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
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()
    with tf.Session() as sess:
        if args.input.split(".")[-1] == 'mp3':
            raw = pydub.AudioSegment.from_mp3(args.input)
            audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
        elif args.input.split(".")[-1] == 'wav':
            _, audio = wav.read(args.input)
        else:
            raise Exception("Unknown file format")
        N = len(audio)
        print(audio.shape)
        np.set_printoptions(threshold=sys.maxsize)
        print(audio)
        new_input = tf.placeholder(tf.float32, [1, N])
        lengths = tf.placeholder(tf.int32, [1])

        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            logits = get_logits_smooth(new_input, lengths, args.smooth_sigma, args.sample_num, args.smooth_type)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)

        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)

        print('logits shape', logits.shape)
        length = (len(audio)-1)//320
        l = len(audio)
        r = sess.run(decoded, {new_input: [audio],
                               lengths: [length]})

        print("-"*80)
        print("-"*80)

        print("Classification:")
        print("".join([toks[x] for x in r[0].values]))
        print("-"*80)
        print("-"*80)

main()
