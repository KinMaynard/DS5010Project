import numpy as np

from soundscope.util.split import split


def midside(array, channels, code=True):
    """
    Encode stereo array as midside or decode midside array to stereo.

    In a stereo signal with left and right channels, there
    are components in both channels that are shared and those that are
    different between the sides. Midside encoding separates a stereo
    signal's sum and differences into separate channels. The mid channel
    contains what is common between the left and right channels and the
    side channel contains the differences.

    array: 2d numpy array of audio data (L/R, or M/S).
    channels: 1 for mono, 2 stereo signal.
    code: True encodes Mid/Side, False decodes Mid/Side (default True).
    returns: given L/R: a 2d array of audio data encoded as mid/side,
    given M/S: a 2d array of audio data encoded as L/R.
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