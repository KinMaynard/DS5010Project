'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import numpy as np
import inquirer
from subpackages.io.import_array import import_array
from subpackages.dsp.normalize import normalize
from subpackages.dsp.midside import midside

# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# ignore divide by 0 error in log
np.seterr(divide = 'ignore')

if __name__ == '__main__':
	# Test selector
	questions = [inquirer.Checkbox('tests', message='Which tests to run?', 
		choices=['Mono', 'Stereo', 'Downsample', 'Normalize', 'Midside'],),]

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

		if 'Normalize' in answers['tests']:
			data, normal = normalize(data)
			# after normalization
			print('Waveform after normalization.')
			print(data)

		if 'Midside' in answers['tests']:
			# midside encoding test mono
			encoded, ms = midside(data, channels, name)
			print(encoded, ms)

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

		if 'Normalize' in answers['tests']:
			data, normal = normalize(data)
			# after normalization
			print(data)

		if 'Midside' in answers['tests']:
			# midside encoding test stereo
			encoded, ms = midside(data, channels, name)
			print(encoded, ms)

			# midside decoding test stereo
			decoded, ms = midside(encoded, channels, name)
			print(decoded, ms)