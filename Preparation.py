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

Exceptions:
	Spectrogram
		divide by 0 error, happens when converting FFT to dBFS
			run trim on data before to fix this for trailing and leading 0's
			untested on zeros inside nonzero data

To Do:
export_array: 
	convert float64 array data to int data if subtype is int
	figure out which subtypes are int and which are float?
spectrogram:
	reconcile stereo plot top/bottom y axis
	smaller colorbar for mono plot
'''

import soundfile as sf
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

def import_array(file):
	'''
	Import audio file as 64 bit float array
	file: audio file
	returns: a filename, data (a 64 bit float numpy array of audio data), subtype: the files subtype (list of possible under sf.available_subtypes)
	'''
	# extracting the filename and subtype from soundfile's info object
	info = str(sf.info(file))

	# includes file path
	name = info[:info.find('\n')]
	channels = info[info.find('channels:') + 10:info.find('channels:') + 11]
	subtype = info[info.find('subtype:') + 9:]
	subtype = subtype[subtype.find('['):]

	# reading the audio file as a soundfile numpy array
	data, sample_rate = sf.read(file)
	return name, channels, data, subtype, sample_rate

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

def first_nonzero(array, axis, mask, invalid_val=-1):
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

def last_nonzero(array, axis, mask, invalid_val=-1):
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
	# mask of absolute value of values > epsilon
	mask1 = mask(array)
	# return a copy of array sliced from first nonzero element to last nonzero element
	# adds 1 to compensate for indexing from zero
	return array[np.amin(first_nonzero(array, 0, mask1)):np.amax(last_nonzero(array, 0, mask1)) + 1].copy()

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

def spectrogram(array, channels, sample_rate, name):
	'''
	Creates a spectrogram given an array of audio data
	array: 1 or 2d numpy array of audio data
	channels: 1 mono or 2 stereo, number of channels in audio array
	returns a spectrogram with y: frequency decibel scale logarithmic, x: time (seconds)
	'''
	# global fontsize change
	plt.rcParams.update({'font.size': 8})

	# Stereo subplots fasceted
	if channels == '2':
		# divide array into stereo components
		array_list = np.hsplit(array, 2)
		left, right = array_list[0].flatten(order='F'), array_list[1].flatten(order='F')
		
		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')
		fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
		
		# labeling axes & title
		ax2.set_xlabel('Time (s)')
		ax1.set_ylabel('Left Frequency (kHz)')
		ax2.set_ylabel('Right Frequency (kHz)')
		ax1.set_title('%s Spectrogram' % name)
		
		# x axis on top
		ax1.xaxis.tick_top()
		
		# plot spectrograms
		specl, fql, tl, iml = ax1.specgram(left, Fs=sample_rate, cmap='magma', scale='dB', vmin=-120, vmax=0)
		specr, fqr, tr, imr = ax2.specgram(right, Fs=sample_rate, cmap='magma', scale='dB', vmin=-120, vmax=0)
		
		# make space for colorbar & stack plots snug
		fig.subplots_adjust(right=0.84, hspace=0)
		
		# colorbar
		cbar_max = 0
		cbar_min = -120
		cbar_step = 5
		cbar_ax = fig.add_axes([0.845, 0.11, 0.007, 0.77])
		plt.colorbar(iml, ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step), cax=cbar_ax).set_label('Amplitude (dB)')
		
		# limit y axes to human hearing range
		ax1.set_ylim([0, 20000])
		ax2.set_ylim([0, 20000])

		# fq in kHz
		scale = 1e3
		ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
		ax1.yaxis.set_major_formatter(ticks)
		ax2.yaxis.set_major_formatter(ticks)
		return plt.show()

	# Mono case
	elif channels == '1':
		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')
		fig, ax = plt.subplots()

		# labeling axes & title
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('Frequency (kHz)')
		ax.set_title('%s Spectrogram' % name)
		
		# plot spectrogram
		spec, fq, t, im = plt.specgram(array, Fs= sample_rate, cmap='magma', scale='dB', vmin=-120, vmax=0)
		
		# colorbar
		# smaller colorbar for mono plot
		cbar_max = 0
		cbar_min = -120
		cbar_step = 5
		plt.colorbar(im, ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step)).set_label('Amplitude (dB)')
		
		# limit y axis to human hearing range
		plt.ylim([0, 20000])

		# fq in kHz
		scale = 1e3
		ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
		ax.yaxis.set_major_formatter(ticks)
		return plt.show()

	# wrong array case
	else:
		return ('invalid array')

if __name__ == '__main__':
	# spectrogram test case mono file
	name, channels, data, subtype, sample_rate = import_array('../binaries/Clap Innerworks 1.wav')
	data = trim(data)
	spectrogram(data, channels, sample_rate, name)

	# spectrogram test case stereo file
	name, channels, data, subtype, sample_rate = import_array('../binaries/Bottle.aiff')
	data = trim(data)
	spectrogram(data, channels, sample_rate, name)