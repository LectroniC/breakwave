## A attacker uses the loss proposed in Qin et al.

import numpy as np
import tensorflow as tf
import argparse
from shutil import copyfile
import generate_masking_threshold as generate_mask

import scipy.io.wavfile as wav

import struct
import time
import os
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")

import DeepSpeech

from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits
from Defender.spectral import waveform_to_mel_to_waveform_tf
import Defender.audioio as audioio

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

WINDOW_SIZE = 2048

class Transform(object):
    """
    Return: PSD
    """

    def __init__(self, window_size):
        self.scale = 8.0 / 3.0
        self.frame_length = int(window_size)
        self.frame_step = int(window_size // 4)
        self.window_size = window_size

    def __call__(self, x, psd_max_ori):
        win = tf.contrib.signal.stft(x, self.frame_length, self.frame_step)
        z = self.scale * tf.abs(win / self.window_size)
        psd = tf.square(z)
        PSD = tf.pow(10.0, 9.6) / tf.reshape(psd_max_ori, [-1, 1, 1]) * psd
        return PSD

class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate_stage1=10, learning_rate_stage2=10, 
                 num_iterations_stage1=5000, num_iterations_stage2=5000, 
                 batch_size=1,
                 fs=None,
                 mp3=False, l2penalty=float('inf'), restore_path=None):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """
        
        self.sess = sess
        self.learning_rate_stage1 = learning_rate_stage1
        self.learning_rate_stage2 = learning_rate_stage2
        self.num_iterations_stage1 = num_iterations_stage1
        self.num_iterations_stage2 = num_iterations_stage2
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        self.mp3 = mp3

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know which
        # ones are ours so when we restore the session we don't
        # clobber them.
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_phrase_lengths')

        self.th = tf.placeholder(tf.float32, shape=[batch_size, None, None], name="qq_th")
        self.psd_max_ori = tf.placeholder(tf.float32, shape=[batch_size], name="qq_psd")
        self.alpha = tf.Variable(np.ones((batch_size), dtype=np.float32) * 0.05, name="qq_alpha")
        

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -2000, 2000)*self.rescale

        # We set the new input to the model to be the abve delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = self.apply_delta*mask + original

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape,
                                 stddev=2)
        pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in, lengths)

        # Apply the waveguard transform and keep the same format
        NFFT = 1024
        NHOP = 256
        mel_bins = 80
        preprocess_passin = audioio.audio_preprocess_tf(pass_in)
        waveguarded_pass_in = waveform_to_mel_to_waveform_tf(preprocess_passin, fs, NFFT, NHOP, mel_num_bins=mel_bins)
        transformed_pass_in = audioio.audio_postprocess_tf(waveguarded_pass_in)

        with tf.variable_scope("", reuse=True):
            self.logits_transformed = logits_transformed = get_logits(transformed_pass_in, lengths, reuse=True)

        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)

        # Choose the loss function we want -- either CTC or CW
        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            raise NotImplemented("The current version of this project does not include the original CTC loss formation.")
        elif loss_fn == "CW":
            raise NotImplemented("The current version of this project does not include the improved CW loss function implementation.")
        elif loss_fn == "QWGCTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths)
            
            ctcloss_1 = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)
            ctcloss_2 = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits_transformed, sequence_length=lengths)
            

            # compute the loss for masking threshold
            self.loss_th_list = []
            self.transform = Transform(WINDOW_SIZE)
            for i in range(self.batch_size):
                logits_delta = self.transform(
                    (self.apply_delta[i, :]), (self.psd_max_ori)[i]
                )
                loss_th = tf.reduce_mean(tf.nn.relu(logits_delta - (self.th)[i]))
                loss_th = tf.expand_dims(loss_th, dim=0)
                self.loss_th_list.append(loss_th)
            self.loss_th = tf.concat(self.loss_th_list, axis=0)
            
            # Again here we donnot penalize distortion
            if not np.isinf(l2penalty):
                loss = tf.reduce_mean((self.new_input-self.original)**2,axis=1) + \
                    l2penalty*ctcloss_1 + l2penalty*ctcloss_2
            else:
                loss = ctcloss_1 + ctcloss_2
            self.expanded_loss = tf.constant(0)
        else:
            raise NotImplemented("Unsupported attack type")

        self.loss = loss
        self.ctcloss_1 = ctcloss_1
        self.ctcloss_2 = ctcloss_2
        
        # Set up the Adam optimizer to perform gradient descent for us
        start_vars = set(x.name for x in tf.global_variables())

        optimizer_stage1 = tf.train.AdamOptimizer(learning_rate_stage1)
        grad,var = optimizer_stage1.compute_gradients(self.loss, [delta])[0]
        self.train = optimizer_stage1.apply_gradients([(tf.sign(grad),var)])


        optimizer_stage2 = tf.train.AdamOptimizer(learning_rate_stage2)
        grad21,var21 = optimizer_stage2.compute_gradients(self.loss, [delta])[0]
        grad22,var22 = optimizer_stage2.compute_gradients(self.alpha * self.loss_th, [delta])[0]
        self.train21 = optimizer_stage2.apply_gradients([(grad21, var21)])
        self.train22 = optimizer_stage2.apply_gradients([(grad22, var22)])
        self.train2 = tf.group(self.train21, self.train22)

        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        sess.run(tf.variables_initializer(new_vars+[delta]))

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)
        self.decoded_transformed, _ = tf.nn.ctc_beam_search_decoder(logits_transformed, lengths, merge_repeated=False, beam_width=100)


    def attack_stage_1(self, audio, lengths, target, finetune=None):
        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths)-1)//320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths)-1)//320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None]*self.batch_size

        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune-audio))
        
        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iterations_stage1
        for i in range(MAX):
            iteration = i
            now = time.time()

            # Print out some debug information every 10 iterations.
            if i%10 == 0:
                new, delta, r_out, r_logits, r_out_transformed, r_logits_transformed = sess.run((self.new_input, self.delta, self.decoded, self.logits, self.decoded_transformed, self.logits_transformed))
                lst = [(r_out, r_logits, r_out_transformed, r_logits_transformed)]

                for out, logits, out_transformed, logits_transformed in lst:

                    print("undefended:\n")
                    chars = out[0].values
                    res = np.zeros(out[0].dense_shape)+len(toks)-1
                    for ii in range(len(out[0].values)):
                        x,y = out[0].indices[ii]
                        res[x,y] = out[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    print("\n".join(res))
                    
                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))

                    print("defended:\n")
                    chars = out_transformed[0].values
                    res = np.zeros(out_transformed[0].dense_shape)+len(toks)-1
                    for ii in range(len(out_transformed[0].values)):
                        x,y = out_transformed[0].indices[ii]
                        res[x,y] = out_transformed[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    print("\n".join(res))
                    
                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits_transformed,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))

            feed_dict = {}
                
            # Actually do the optimization ste
            d, el, cl_1, cl_2, l, logits, logits_transformed, new_input, _ = sess.run((self.delta, self.expanded_loss,
                                                           self.ctcloss_1, self.ctcloss_2, self.loss,
                                                           self.logits, self.logits_transformed, self.new_input,
                                                           self.train),
                                                          feed_dict)
                    
            # Report progress
            print("cl_1 %.3f"%np.mean(cl_1), "\t", "\t".join("%.3f"%x for x in cl_1))
            print("cl_2 %.3f"%np.mean(cl_2), "\t", "\t".join("%.3f"%x for x in cl_2))

            for ii in range(self.batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.

                if (self.loss_fn == "QWGCTC" and i%10 == 0 and res[ii] == "".join([toks[x] for x in target[ii]])) \
                   or (i == MAX-1 and final_deltas[ii] is None):
                    # Get the current constant
                    rescale = sess.run(self.rescale)
                    if rescale[ii]*2000 > np.max(np.abs(d)):
                        # If we're already below the threshold, then
                        # just reduce the threshold to the current
                        # point and save some time.
                        print("It's way over", np.max(np.abs(d[ii]))/2000.0)
                        rescale[ii] = np.max(np.abs(d[ii]))/2000.0

                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.
                    rescale[ii] *= .8

                    # Adjust the best solution found so far
                    final_deltas[ii] = new_input[ii]

                    print("Worked i=%d ctcloss_1=%f bound=%f"%(ii,cl_1[ii], 2000*rescale[ii][0]))
                    print("Worked i=%d ctcloss_2=%f bound=%f"%(ii,cl_2[ii], 2000*rescale[ii][0]))
                    #print('delta',np.max(np.abs(new_input[ii]-audio[ii])))
                    sess.run(self.rescale.assign(rescale))

                    # Just for debugging, save the adversarial example
                    # to /tmp so we can see it if we want
                    wav.write("/tmp/adv.wav", 16000,
                              np.array(np.clip(np.round(new_input[ii]),
                                               -2**15, 2**15-1),dtype=np.int16))

        return final_deltas, delta
    
    def attack_stage_2(self, audio, lengths, target, finetune=None, th_batch=None, psd_max_batch=None):

        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths)-1)//320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths)-1)//320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))
        sess.run(tf.assign(self.alpha, np.ones((self.batch_size), dtype=np.float32) * 0.05))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None]*self.batch_size
        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune-audio))
        
        # We'll make a bunch of iterations of gradient descent here
        now = time.time()
        MAX = self.num_iterations_stage2
        min_th = 0.0005
        for i in range(MAX):
            iteration = i
            now = time.time()

            # Print out some debug information every 10 iterations.
            if i%10 == 0:
                alpha = sess.run(self.alpha)
                new, delta, r_out, r_logits, r_out_transformed, r_logits_transformed = sess.run((self.new_input, self.delta, self.decoded, self.logits, self.decoded_transformed, self.logits_transformed))
                lst = [(r_out, r_logits, r_out_transformed, r_logits_transformed)]

                for out, logits, out_transformed, logits_transformed in lst:

                    print("undefended:\n")
                    chars = out[0].values
                    res = np.zeros(out[0].dense_shape)+len(toks)-1
                    for ii in range(len(out[0].values)):
                        x,y = out[0].indices[ii]
                        res[x,y] = out[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    print("\n".join(res))
                    
                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))

                    print("defended:\n")
                    chars = out_transformed[0].values
                    res = np.zeros(out_transformed[0].dense_shape)+len(toks)-1
                    for ii in range(len(out_transformed[0].values)):
                        x,y = out_transformed[0].indices[ii]
                        res[x,y] = out_transformed[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    print("\n".join(res))
                    
                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits_transformed,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))

            feed_dict = {
                self.th: th_batch,
                self.psd_max_ori: psd_max_batch
            }
                
            # Actually do the optimization ste
            d, el, cl_1, cl_2, l, logits, logits_transformed, new_input, _ = sess.run((self.delta, self.expanded_loss,
                                                           self.ctcloss_1, self.ctcloss_2, self.loss,
                                                           self.logits, self.logits_transformed, self.new_input,
                                                           self.train2),
                                                          feed_dict)
                    
            # Report progress
            print("stage 2 cl_1 %.3f"%np.mean(cl_1), "\t", "\t".join("%.3f"%x for x in cl_1))
            print("stage 2 cl_2 %.3f"%np.mean(cl_2), "\t", "\t".join("%.3f"%x for x in cl_2))

            for ii in range(self.batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.

                if (self.loss_fn == "QWGCTC" and i%10 == 0 and res[ii] == "".join([toks[x] for x in target[ii]])) \
                   or (i == MAX-1 and final_deltas[ii] is None):
                    # Get the current constant
                    rescale = sess.run(self.rescale)
                    if rescale[ii]*2000 > np.max(np.abs(d)):
                        # If we're already below the threshold, then
                        # just reduce the threshold to the current
                        # point and save some time.
                        print("It's way over", np.max(np.abs(d[ii]))/2000.0)
                        rescale[ii] = np.max(np.abs(d[ii]))/2000.0

                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.
                    rescale[ii] *= .8

                    # Adjust the best solution found so far
                    final_deltas[ii] = new_input[ii]

                    print("Worked i=%d ctcloss_1=%f bound=%f"%(ii,cl_1[ii], 2000*rescale[ii][0]))
                    print("Worked i=%d ctcloss_2=%f bound=%f"%(ii,cl_2[ii], 2000*rescale[ii][0]))
                    #print('delta',np.max(np.abs(new_input[ii]-audio[ii])))
                    sess.run(self.rescale.assign(rescale))

                    # Just for debugging, save the adversarial example
                    # to /tmp so we can see it if we want
                    wav.write("/tmp/adv.wav", 16000,
                              np.array(np.clip(np.round(new_input[ii]),
                                               -2**15, 2**15-1),dtype=np.int16))
                elif (i % 50 == 0):
                    alpha[ii] *= 0.8
                    alpha[ii] = max(alpha[ii], min_th)
                    sess.run(tf.assign(self.alpha, alpha))

        return final_deltas
    
def main():
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input", nargs='+',
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    parser.add_argument('--target', type=str,
                        required=True,
                        help="Target transcription")
    parser.add_argument('--out', type=str, nargs='+',
                        required=False,
                        help="Path for the adversarial example(s)")
    parser.add_argument('--outprefix', type=str,
                        required=False,
                        help="Prefix of path for adversarial examples")
    parser.add_argument('--finetune', type=str, nargs='+',
                        required=False,
                        help="Initial .wav file(s) to use as a starting point")
    parser.add_argument('--lr_stage1', type=int,
                        required=False, default=100,
                        help="Learning rate for stage 1 optimization")
    parser.add_argument('--lr_stage2', type=int,
                        required=False, default=1,
                        help="Learning rate for stage 2 optimization")
    parser.add_argument('--iterations_stage1', type=int,
                        required=False, default=1000,
                        help="Maximum number of iterations of stage 1")
    parser.add_argument('--iterations_stage2', type=int,
                        required=False, default=2000,
                        help="Maximum number of iterations of stage 2")
    parser.add_argument('--l2penalty', type=float,
                        required=False, default=float('inf'),
                        help="Weight for l2 penalty on loss function")
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()

    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        finetune = []
        audios = []
        lengths = []

        if args.out is None:
            assert args.outprefix is not None
        else:
            assert args.outprefix is None
            assert len(args.input) == len(args.out)
        if args.finetune is not None and len(args.finetune):
            assert len(args.input) == len(args.finetune)
        
        # Load the inputs that we're given
        for i in range(len(args.input)):
            fs, audio = wav.read(args.input[i])

            # Convert
            if max(audio) < 1:
                audio = audio * 32768
            else:
                audio = audio
            
            print("decoded audio dtype")
            print(audio.dtype)
            assert fs == 16000
            assert audio.dtype == np.int16
            print('source dB', 20*np.log10(np.max(np.abs(audio))))
            audios.append(list(audio))
            lengths.append(len(audio))

            if args.finetune is not None:
                finetune.append(list(wav.read(args.finetune[i])[1]))

        maxlen = max(map(len,audios))

        # Padding multiple audio files in a batch to the same length
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])

        th_batch = []
        psd_max_batch = []
        # Calculate global threshold:
        for i in range(len(audios)):
            # Hardcode sample rate and window_size 
            th, psd_max = generate_mask.generate_th(
                audios[i].astype(float), 16000, WINDOW_SIZE
            )
            th_batch.append(th)
            psd_max_batch.append(psd_max)
        th_batch = np.array(th_batch)
        psd_max_batch = np.array(psd_max_batch)

        phrase = args.target

        # Set up the attack class and run it
        attack = Attack(sess, 'QWGCTC', len(phrase), maxlen,
                        batch_size=len(audios),
                        learning_rate_stage1=args.lr_stage1,
                        learning_rate_stage2=args.lr_stage2,
                        num_iterations_stage1=args.iterations_stage1,
                        num_iterations_stage2=args.iterations_stage2,
                        l2penalty=args.l2penalty,
                        fs = fs,
                        restore_path=args.restore_path)
        deltas = attack.attack_stage_1(audios,
                            lengths,
                            [[toks.index(x) for x in phrase]]*len(audios),
                            finetune)
        
        deltas = attack.attack_stage_2(audios,
                            lengths,
                            [[toks.index(x) for x in phrase]]*len(audios),
                            finetune=deltas,
                            th_batch=th_batch,
                            psd_max_batch=psd_max_batch
                            )

        # And now save it to the desired output
        for i in range(len(args.input)):
            if args.out is not None:
                path = args.out[i]
            else:
                path = args.outprefix+str(i)+".wav"
            wav.write(path, 16000,
                        np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                        -2**15, 2**15-1),dtype=np.int16))
            print("Final distortion", np.max(np.abs(deltas[i][:lengths[i]]-audios[i][:lengths[i]])))

main()
