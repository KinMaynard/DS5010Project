"""
Audio processing & visualization library

Inverts the polarity of an audio array (phase)
"""

import numpy as np


def invert(array):
    """
    Inverts the phase (polarity) of an array of audio data
    array: a numpy array of audio data (numbers), not empty
    returns: a version of array with the polarity inverted
    """
    # Inverts polarity of audio data
    return array * -1