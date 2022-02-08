import numpy as np


def split(array, channels):
    """
    Splits 2d array of audio data into 2 1d arrays.

    array: 2d numpy array of audio data
    channels: # of channels in signal, must be 2
    returns: Left and Right channels (or M/S)
    """

    # Divide array into stereo components
    array_list = np.hsplit(array, 2)
    left = array_list[0].flatten(order='F')
    right = array_list[1].flatten(order='F')
    return left, right