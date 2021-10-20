'''
Audio processing & visualization library
'''

import soundfile as sf
import numpy as np

def export_array(name, array, sample_rate, subtype, normal=False):
	'''
	Export numpy array as audio file
	
	name: file to write to (truncates & overwrites if file exists) (str, int or file like object)
	array: soundfile object, numpy array of audio data as 64 bit float
	sample_rate: sample rate of the audio data
	subtype: subtype of the audio data
	returns: none
	'''
	# if the data has been normalized then set the subtype to 64 bit float
	if normal == True:
		subtype = 'Double'

	sf.write(name, array, sample_rate, subtype)
	return None