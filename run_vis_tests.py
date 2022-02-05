"""
Visual tests requiring user.
"""

import unittest

import numpy as np
import matplotlib as mpl
import inquirer

from soundscope.io.import_array import import_array
from soundscope.vis.bins import bins
from soundscope.vis.waveform import waveform
from soundscope.vis.magnitude import magnitude
from soundscope.vis.spectrogram import spectrogram
from soundscope.vis.vectorscope import vectorscope
from soundscope.vis.visualizer import visualizer


if __name__ == '__main__':
    # Test selector
    questions = [inquirer.Checkbox('tests', message='Which tests to run?',
        choices=['Mono', 'Stereo', 'Downsample', 'Bins', 'Waveform',
                 'Magnitude', 'Spectrogram', 'Vectorscope', 'Visualizer'],),]

    answers = inquirer.prompt(questions)

    if 'Mono' in answers['tests']:
        # Waveform to perform tests on
        questions2 = [inquirer.List('waves', message=
            'Which mono wave to test?',
            choices=[
            ('Silence', '../binaries/silence_44100_-infdBFS_Mono.aiff'),
            ('White Noise', '../binaries/white_88k_-3dBFS.wav'),
            ('Linear Chirp', '../binaries/hdchirp_88k_-3dBFS_lin.wav'),
            ('Sin 100Hz', '../binaries/sin_44100_100Hz_-3dBFS_1s.wav'),
            ('Sweep', '../binaries/hdsweep_1Hz_44000Hz_-3dBFS_30s.wav'),
            ('Sin Out', '../binaries/sin100hz_180out.aiff')],
            default=('White Noise', '../binaries/white_88k_-3dBFS.wav')),]

        answers2 = inquirer.prompt(questions2)

        mono = answers2['waves']

        name, channels, data, subtype, sample_rate = import_array(mono)

        if 'Bins' in answers['tests']:
            # Downsampling test mono
            binned, bin_sample_rate = bins(data, channels, sample_rate)
            print(binned, bin_sample_rate)

        if 'Downsample' in answers['tests']:
            # Downsampling for visualization
            data, sample_rate = bins(data, channels, sample_rate)

        if 'Waveform' in answers['tests']:
            # Waveform plot test case mono file
            waveform(data, name, channels, sample_rate)

        if 'Magnitude' in answers['tests']:
            # Magnitude test mono file
            magnitude(data, name, channels, sample_rate)

        if 'Spectrogram' in answers['tests']:
            # Spectrogram test case mono file
            spectrogram(data, name, channels, sample_rate)

        if 'Vectorscope' in answers['tests']:
            # Vectorscope mono test
            vectorscope(data, name, channels, sample_rate)

        if 'Visualizer' in answers['tests']:
            # Visualizer mono plot
            visualizer(data, name, channels, sample_rate)

    if 'Stereo' in answers['tests']:
        # Waveform to perform tests on
        questions2 = [inquirer.List('waves', message=
                'Which stereo wave to test?',
            choices=[
            ('Silence', '../binaries/silence_44100_-infdBFS_Stereo.aiff'),
            ('White Noise', '../binaries/whitenoise_44100_0dBFS_Stereo.aiff'),
            ('Chirp Stereo', '../binaries/hdchirp_88k_-3dBFS_lin_Stereo.aiff'),
            ('Sin 440Hz', '../binaries/sin_44100_440Hz_-.8dBFS_Stereo.aiff'),
            ('Sin Out Phase', '../binaries/Sinoutphase.wav'),
            ('Lopez Song Stereo', '../binaries/Saija Original Mix.aiff')],
            default=('White Noise Stereo',
                     '../binaries/whitenoise_44100_0dBFS_Stereo.aiff')),]

        answers2 = inquirer.prompt(questions2)

        stereo = answers2['waves']

        name, channels, data, subtype, sample_rate = import_array(stereo)

        if 'Bins' in answers['tests']:
            # Downsampling test stereo
            binned, bin_sample_rate = bins(data, channels, sample_rate)
            print(binned, bin_sample_rate)

        if 'Downsample' in answers['tests']:
            # Downsampling for visualization
            data, sample_rate = bins(data, channels, sample_rate)

        if 'Waveform' in answers['tests']:
            # Waveform plot test case stereo file
            waveform(data, name, channels, sample_rate)

        if 'Magnitude' in answers['tests']:
            # Magnitude test stereo file
            magnitude(data, name, channels, sample_rate)

        if 'Spectrogram' in answers['tests']:
            # Spectrogram test case stereo file
            spectrogram(data, name, channels, sample_rate)

        if 'Vectorscope' in answers['tests']:
            # Vectorscope stereo test
            vectorscope(data, name, channels, sample_rate)

        if 'Visualizer' in answers['tests']:
            # Visualizer stereo plot
            visualizer(data, name, channels, sample_rate)