'''
Audio processing & visualization library

Unit Tests
'''

import numpy as np
import inquirer
import matplotlib as mpl
from import_array import import_array
from bins import bins
from trim import trim
from split import split
from normalize import normalize
from midside import midside
from invert import invert
from reverse import reverse
from export_array import export_array
from waveform import waveform
from magnitude import magnitude
from spectrogram import spectrogram
from vectorscope import vectorscope
from visualizer import visualizer

# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# ignore divide by 0 error in log
np.seterr(divide = 'ignore')

if __name__ == '__main__':
	# Test selector
	questions = [inquirer.Checkbox('tests', message='Which tests to run?', 
		choices=['Mono', 'Stereo', 'Downsample', 'Bins', 'Normalize', 'Midside', 'Invert', 'Reverse', 'Export', 'Waveform', 'Magnitude', 'Spectrogram', 
		'Vectorscope', 'Visualizer'],),]

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
		
		if 'Normalize' in answers['tests']:
			# before normalization
			print('Waveform before normalization.')
			waveform(data, name, channels, sample_rate)
			data, normal = normalize(data)
			# after normalization
			print('Waveform after normalization.')
			waveform(data, name, channels, sample_rate)

		if 'Midside' in answers['tests']:
			# midside encoding test mono
			encoded, ms = midside(data, channels, name)
			print(encoded, ms)

		if 'Invert' in answers['tests']:
			# Polarity inversion test
			print(data)
			print(invert(data))

		if 'Reverse' in answers['tests']:
			# Reverse whole array test
			print(data[:10])
			print(reverse(data[:10], channels))

			# Reverse both halves
			print(data[:10])
			print(reverse(data[:10], channels, 2))

			# Reverse last third
			print(data[:12])
			print(reverse(data[:12], channels, 3))

		if 'Export' in answers['tests']:
			# import array
			# check min max median mean
			stats = (np.min(data), np.median(data), np.max(data), np.mean(data))
			print(stats)
			# normalize
			data, normal = normalize(data)
			# export
			export_array('../binaries/test.aiff', data, sample_rate, subtype, normal)
			# import
			name, channels, data, subtype, sample_rate = import_array('../binaries/test.aiff')
			# check min max median mean
			n_stats = (np.min(data), np.median(data), np.max(data), np.mean(data))
			print(n_stats)
			# if measurements different then works

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

		if 'Normalize' in answers['tests']:
			# before normalization
			waveform(data, name, channels, sample_rate)
			dat, normal = normalize(data)
			# after normalization
			waveform(data, name, channels, sample_rate)

		if 'Midside' in answers['tests']:
			# midside encoding test stereo
			encoded, ms = midside(data, channels, name)
			print(encoded, ms)

			# midside decoding test stereo
			decoded, ms = midside(encoded, channels, name)
			print(decoded, ms)

		if 'Invert' in answers['tests']:
			# Polarity inversion test
			print(data)
			print(invert(data))

		if 'Reverse' in answers['tests']:
			# Reverse whole array test
			print(data[:10])
			print(reverse(data[:10], channels))

			# Reverse both halves
			print(data[:10])
			print(reverse(data[:10], channels, 2))

			# Reverse last third
			print(data[:12])
			print(reverse(data[:12], channels, 3))
		
		if 'Export' in answers['tests']:
			# import array
			# check min max median mean
			stats = (np.min(data), np.median(data), np.max(data), np.mean(data))
			print(stats)
			# normalize
			data, normal = normalize(data)
			# export
			export_array('../binaries/test.aiff', data, sample_rate, subtype, normal)
			# import
			name, channels, data, subtype, sample_rate = import_array('../binaries/test.aiff')
			# check min max median mean
			n_stats = (np.min(data), np.median(data), np.max(data), np.mean(data))
			print(n_stats)
			# if measurements different then works

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

