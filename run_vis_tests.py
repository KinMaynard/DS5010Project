'''
Visual tests requiring user.
'''

import unittest
import numpy as np
import inquirer
import matplotlib as mpl
from subpackages.io.import_array import import_array
from subpackages.vis.bins import bins
from subpackages.vis.waveform import waveform
from subpackages.vis.magnitude import magnitude
from subpackages.vis.spectrogram import spectrogram
from subpackages.vis.vectorscope import vectorscope
from subpackages.vis.visualizer import visualizer

# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# ignore divide by 0 error in log
np.seterr(divide = 'ignore')

if __name__ == '__main__':
	# Test selector
	questions = [inquirer.Checkbox('tests', message='Which tests to run?', 
		choices=['Mono', 'Stereo', 'Downsample', 'Bins', 'Waveform', 'Magnitude', 'Spectrogram', 'Vectorscope', 'Visualizer'],),]

	answers = inquirer.prompt(questions)

	if 'Mono' in answers['tests']:
		# Waveform to perform tests on
		questions2 = [inquirer.List('waves', message='Which mono wave to test?', 
			choices=[('Silence', '../binaries/silence_44100_-infdBFS_Mono.aiff'), ('White Noise', '../binaries/white_88k_-3dBFS.wav'), 
			('Linear Chirp', '../binaries/hdchirp_88k_-3dBFS_lin.wav'), ('Sin 100Hz', '../binaries/sin_44100_100Hz_-3dBFS_1s.wav'), 
			('Sweep 1-44kHz', '../binaries/hdsweep_1Hz_44000Hz_-3dBFS_30s.wav'), ('Sin Out', '../binaries/sin100hz_180out.aiff')],
			default=('White Noise', '../binaries/white_88k_-3dBFS.wav')),]

		answers2 = inquirer.prompt(questions2)

		mono = answers2['waves']

		name, channels, data, subtype, sample_rate = import_array(mono)

		if 'Bins' in answers['tests']:
			# downsampling test mono
			binned, bin_sample_rate = bins(data, channels, sample_rate)
			print(binned, bin_sample_rate)

		if 'Downsample' in answers['tests']:
			# downsampling for visualization
			data, sample_rate = bins(data, channels, sample_rate)

		if 'Waveform' in answers['tests']:
			# Waveform plot test case mono file
			waveform(data, name, channels, sample_rate)

		if 'Magnitude' in answers['tests']:
			# magnitude test mono file
			magnitude(data, name, channels, sample_rate)

		if 'Spectrogram' in answers['tests']:
			# spectrogram test case mono file
			spectrogram(data, name, channels, sample_rate)

		if 'Vectorscope' in answers['tests']:
			# vectorscope mono test
			vectorscope(data, name, channels, sample_rate)

		if 'Visualizer' in answers['tests']:
			# visualizer mono plot
			visualizer(data, name, channels, sample_rate)

	if 'Stereo' in answers['tests']:
		# Waveform to perform tests on
		questions2 = [inquirer.List('waves', message='Which stereo wave to test?', 
			choices=[('Silence Stereo', '../binaries/silence_44100_-infdBFS_Stereo.aiff'), ('White Noise Stereo', '../binaries/whitenoise_44100_0dBFS_Stereo.aiff'), 
			('Chirp Stereo', '../binaries/hdchirp_88k_-3dBFS_lin_Stereo.aiff'), ('Sin 440Hz Stereo', '../binaries/sin_44100_440Hz_-.8dBFS_Stereo.aiff'), 
			('Sin Out Phase', '../binaries/Sinoutphase.wav'), ('Lopez Song Stereo', '../binaries/Saija Original Mix.aiff')],
			default=('White Noise Stereo', '../binaries/whitenoise_44100_0dBFS_Stereo.aiff')),]

		answers2 = inquirer.prompt(questions2)

		stereo = answers2['waves']

		name, channels, data, subtype, sample_rate = import_array(stereo)

		if 'Bins' in answers['tests']:
			# downsampling test stereo
			binned, bin_sample_rate = bins(data, channels, sample_rate)
			print(binned, bin_sample_rate)

		if 'Downsample' in answers['tests']:
			# downsampling for visualization
			data, sample_rate = bins(data, channels, sample_rate)

		if 'Waveform' in answers['tests']:
			# Waveform plot test case stereo file
			waveform(data, name, channels, sample_rate)

		if 'Magnitude' in answers['tests']:
			# magnitude test stereo file
			magnitude(data, name, channels, sample_rate)

		if 'Spectrogram' in answers['tests']:
			# spectrogram test case stereo file
			spectrogram(data, name, channels, sample_rate)

		if 'Vectorscope' in answers['tests']:
			# vectorscope stereo test
			vectorscope(data, name, channels, sample_rate)

		if 'Visualizer' in answers['tests']:
			# visualizer stereo plot
			visualizer(data, name, channels, sample_rate)

