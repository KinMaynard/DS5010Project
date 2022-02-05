'''
Audio processing & visualization library

normalizes audio files
'''

import numpy as np


def normalize(array):
    '''
    Performs peak normalization on array of audio data

    array: array of audio data 64 bit floating point

    returns: normalized version of array
    '''
    # normalize data between -1 and 1
    array = 2 * ((array - np.min(array)) / (np.max(array) - np.min(array))) - 1
    normal = True
    return array, normal