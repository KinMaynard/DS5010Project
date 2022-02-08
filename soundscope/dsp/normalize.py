import numpy as np


def normalize(array):
    """
    Peak normalize audio data array.

    array: array of audio data 64 bit floating point

    returns: normalized version of array
    """

    # Normalize data between -1 and 1
    array = 2 * ((array-np.min(array)) / (np.max(array)-np.min(array))) - 1
    return array, True