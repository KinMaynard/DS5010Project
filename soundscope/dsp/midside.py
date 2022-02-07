"""
Audio processing & visualization library

Encodes a L/R audio array as Midside or decodes a midside array as L/R
"""

import numpy as np

from soundscope.util.split import split


def midside(array, channels, code=True):
    """
    Encode stereo array as midside or decode midside array to stereo.

    Sum and difference matrix:
    mid: (L+R)-3dB or 1/2(L+R)
    side: (L-R)-3dB or 1/2(L-R)
    L: (M+S)-3dB or L = M + S = 1/2(L+R) + 1/2(L-R) = 1/2 * L + 1/2 * L
    R: (M-S)-3dB or R = M - S = 1/2(L+R) - 1/2(L-R) = 1/2 * R + 1/2 * R
    -3db accounts for the +6dB change to the output of a double encoding

    array: 2d numpy array of audio data (L/R, or M/S)
    channels: # of channels in audio signal (must be 2)
    code: True when encoding Mid/Side, False when decoding Mid/Side
    (default True)
    returns: given L/R: a 2d array of audio data encoded as mid/side,
    given M/S: a 2d array of audio data encoded as L/R
    """
    # Check for stereo or mid/side array
    if channels == '1':
        # Treat mono array as stereo array of 2 mono components
        # (Will sum to only mid data no side)
        left, right = array, array

    else:
        # Divide array into stereo components
        left, right = split(array, channels)

    if code:
        # Mid/Side encoding
        mid = 0.5 * (left+right)
        side = 0.5 * (left-right)

        encoded = np.stack((mid, side), axis=-1)

        midside = True

        return encoded, midside

    else:
        # Mid/Side decoding
        # mid + side
        newleft = left + right
        # mid - side
        newright = left - right

        decoded = np.stack((newleft, newright), axis=-1)

        midside = False

        return decoded, midside