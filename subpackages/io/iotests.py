'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import numpy as np
import inquirer
from subpackages.io.import_array import import_array
from subpackages.io.export_array import export_array

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