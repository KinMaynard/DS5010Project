"""
Audio processing & visualization library

Splits stereo audio arrays into 2 mono arrays
"""

import numpy as np


def split(array, channels):
    """
    splits 2d array of audio data into Left and Right 
        or Mid and Side channels
    array: 2d numpy array of audio data
    channels: # of channels in signal, must be 2
    returns: Left and Right channels (or M/S)
    """
    # divide array into stereo components
    array_list = np.hsplit(array, 2)
    left, right = array_list[0].flatten(order='F'), array_list[1].flatten(order='F')
    return left, right