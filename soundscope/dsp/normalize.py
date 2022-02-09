import numpy as np


def normalize(array):
    """
    Peak normalize audio data array.

    64 bit audio files range in amplitude from 0 to 1 and -1. Peak
    normalization of an audio array scales the amplitudes of all data
    in the array such that the maximum amplitude of the peak is 1.

    array: array of audio data 64 bit floating point

    returns: normalized version of array
    """

    # Normalize data between -1 and 1
    array = 2 * ((array-np.min(array)) / (np.max(array)-np.min(array))) - 1
    return array, True