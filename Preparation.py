'''
Preparation

This is a module for:
	Importing of audio files as numpy arrays
	Exporting of numpy arrays as audio files
	Trimming leading and trailing silence from audio files
	Peak normalization of audio files

Analysis:

This is a module for:
	Spectrogram (Mel Scale, DB Scale not amplitude)
	Tempo
		transient detection
	Key/Note
		fundemental frequency/pitch detection
		Detect all pitches in sample
'''

import soundfile as sf
import numpy as np
import sys

def import_array(file):
	'''
	Import audio file as 64 bit float array
	file: audio file
	returns: a filename, data (a 64 bit float numpy array of audio data), subtype: the files subtype (list of possible under sf.available_subtypes)
	'''
	# extracting the filename and subtype from soundfile's info object
	info = str(sf.info(file))
	name = info[:info.find('\n')]
	subtype = info[info.find('subtype:') + 9:]
	subtype = subtype[subtype.find('['):]
	# reading the audio file as a soundfile numpy array
	data, sample_rate = sf.read(file)
	return name, data, subtype, sample_rate

def mask(array):
	'''
	calculates a boolean mask of non zeros (values greater than positive epsilon smaller
	than negative epsilon)
	array: numpy array of audio data
	returns boolean mask of nonzeros (values greater than epsilon) in array
	'''
	epsilon = sys.float_info.epsilon
	mask = abs(array) > epsilon
	return mask

def first_nonzero(array, axis, invalid_val=-1, mask):
	'''
	Helper function for trim function that gets the index of the first non_zero element in an array
	array: 1d or 2d numpy array of audio data
	axis: generic axis specifier along which to access elements 
	invalid_value: marker for dimensions of only zeros
	mask: boolean array of non zeros (non epsilon) values in array
	returns: index of first non zero value in array
	
	argmax returns indicies of first matches (True values) in cases where max occurs multiple times, 
	using where() given any() as the condition we can return the index from argmax 
	for any true value in the mask and the invalid value marker otherwise.

	Column major order access.
	'''
	# boolean array of True where element of original array is nonzero false otherwise (if zero)
	return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(array, axis, invalid_val=-1, mask):
	'''
	Helper function for trim function that gets the index of the last non_zero element in an array
	array: 1d or 2d numpy array of audio data
	axis: generic axis specifier axis along which to access elements
	invalid_value: marker for dimensions of only zeros, default argument is -1
	mask: boolean array of non zeros (non epsilon) values in array
	returns: index of last non zero value in array

	Similar behavior to first_nonzero however we flip along the axis to access and use argmax again
	for the behavior of finding indicies of first occurence in tie cases
	we compensate for the flipping by ofsetting from the axis length.

	Accessing the array here in column major order.
	'''
	# boolean array of True where element of original array is nonzero false otherwise (if zero)
	dex_last_occur = array.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
	return np.where(mask.any(axis=axis), dex_last_occur, invalid_val)

def trim(array):
	'''
	truncates leading and trailing silence (0's) from audio array
	array: numpy array created from an audio file
	returns: array without leading and trailing silence

	want min index of any channel from first non zero and max index of any channel from last non zero to avoid 2 different sized channels

	future features: definable noise floor to choose what to truncate as silence
	'''
	# return a copy of array sliced from first nonzero element to last nonzero element
	# adds 1 to compensate for indexing from zero
	return array[np.amin(first_nonzero(array, 0)):np.amax(last_nonzero(array, 0)) + 1,:].copy()

def normalize(array):
	'''
	Performs peak normalization on array of audio data
	array: array of audio data 64 bit floating point
	returns: normalized version of array
	'''
	# factors an array of audio data by the difference between 0 dbfs and the peak signal
	return (1 / np.amax(np.abs(array))) * array

def export_array(name, array, sample_rate, subtype):
	'''
	Export numpy array as audio file
	array: soundfile object, numpy array of audio data as 64 bit float
	name: file to write to (truncates & overwrites if file exists) (str, int or file like object)
	sample_rate: sample rate of the audio data
	subtype: subtype of the audio data
	returns: none
	'''
	sf.write(name, array, sample_rate, subtype)
	return None

def spectrogram(array):
	'''
	Creates a spectrogram given an array of audio data
	returns a spectrogram   
	'''

if __name__ == '__main__':