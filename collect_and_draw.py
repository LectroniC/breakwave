import numpy as np
import tensorflow as tf
import argparse

import scipy.io.wavfile as wav

import time
import os
import sys
from collections import namedtuple

# importing the required module
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input",
                        required=True,
                        help="Input audio .wav file(s), at 16KHz (separated by spaces)")
    parser.add_argument('--restore_path', type=str,
                        required=True,
                        help="Path to the DeepSpeech checkpoint (ending in model0.4.1)")

    
    plt.figure(figsize=(3.841, 7.195), dpi=1000)


    # Change resolution
    
    # x axis values
    x = [1,2,3]
    # corresponding y axis values
    y = [2,4,1]

    plt.xlim([25, 50])
    plt.ylim([25, 50])

    
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')

    plt.plot(x, y, '-', color="blue", label = "line 1")
    plt.plot(y, x, '--', color="blue", label = "line 2")
    
    # giving a title to my graph
    plt.title('My first graph!')
    
    # function to show the plot
    plt.savefig('my_fig.png', dpi=1000)
    

main()
