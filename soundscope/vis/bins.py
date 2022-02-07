import numpy as np


def bins(array, channels, sample_rate, bin_size=16):
    """
    Downsample array by averaging bins of bin_size into new array.

    array: numpy array of audio data
    channels: 1 mono, 2 stereo
    sample_rate: sampling rate of the audio file
    bin_size: [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    default 32
    returns: downsampled array
    """
    # Either of len bin_size if array divisible by bin_size or length
    # of the partial bin
    partial_bin = len(array) % bin_size
    to_fill = bin_size - partial_bin
    sample_rate = sample_rate / bin_size

    # Array didn't need padding
    if partial_bin == 0:
        # Separate array into bins of size bin_size, average the bins
        # into new array return array
        # Populate new array with averages of every (bin_size) samples
        return np.mean(array.reshape(-1, bin_size), axis=1), sample_rate

    # Array needed padding for last bin
    else:
        # Pad end of array with mean of last bin so array size divisible
        # by bin_size
        if channels == '1':
            width = ((0, to_fill),)
            padded = np.pad(array, pad_width=width, mode='mean', stat_length=(
                partial_bin,))
            downsampled = np.mean(padded.reshape(-1, bin_size), axis=1)

        else:
            width = ((0, to_fill), (0, 0))
            padded = np.pad(array, pad_width=width, mode='mean', stat_length=(
                partial_bin,))
            downsampled = padded.reshape(-1, bin_size, padded.shape[1]).mean(
                axis=1)
        return downsampled, sample_rate