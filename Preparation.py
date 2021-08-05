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
big plot with all other plots inside:
	sliders for zoom
	radio buttons for L R Both (log magnitude)
spectrogram:
	NFFT
	noverlap
	window
vectorscope:
	test cases
		panned hard L
			not showing anything in plot?
		panned hard R
'''

import soundfile as sf
import numpy as np
import sys
import pdb
import inquirer
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from matplotlib.widgets import RadioButtons

debug = False

def import_array(file):
	'''
	Import audio file as 64 bit float array
	file: audio file
	returns: a filename, number of channels, data (a 64 bit float numpy array of audio data), subtype: the files subtype, sample rate
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

def split(array, channels, name):
	'''
	splits 2d array of audio data into Left and Right or Mid and Side channels
	array: 2d numpy array of audio data
	channels: # of channels in signal, must be 2
	name: audio filename
	returns: Left and Right channels (or M/S)
	'''
	# mono case
	if channels == '1':
		return ('%s is mono, import 2 channel audio array for splitting.' % name)
	else:
		# divide array into stereo components
		array_list = np.hsplit(array, 2)
		left, right = array_list[0].flatten(order='F'), array_list[1].flatten(order='F')
		return left, right

def normalize(array):
	'''
	Performs peak normalization on array of audio data
	array: array of audio data 64 bit floating point
	returns: normalized version of array
	'''
	peak = np.amax(np.abs(array))
	#int64
	if peak in range(-2.0**31, 2.0**31-1.0):
		return ((2.0**31-1.0) / peak) * array
	# int32
	elif peak in range(-2.0**15, 2.0**15-1.0):
		return ((2.0**15-1.0) / peak) * array
	# float
	elif peak in range(-1.0, 1.0):
		# factors an array of audio data by the difference between 0 dbfs and the peak signal
		return (1 / peak) * array

def midside(array, channels, name, code=True):
	'''
	Encodes a stereo array of L/R audio data as mid/side data or mid/side data as L/R

	sum and difference matrix:
	mid: (L+R)-3dB or 1/2(L+R)
	side: (L-R)-3dB or 1/2(L-R)
	L: (M+S)-3dB or L = M + S = 1/2(L+R) + 1/2(L-R) = 1/2 * L + 1/2 * L
	R: (M-S)-3dB or R = M - S = 1/2(L+R) - 1/2(L-R) = 1/2 * R + 1/2 * R
	-3db accounts for the +6dB change to the output of a double encoding

	array: 2d numpy array of audio data (L/R, or M/S)
	channels: # of channels in audio signal (must be 2)
	name: name of audio signal
	code: True when encoding Mid/Side, False when decoding Mid/Side (default True)
	returns: given L/R: a 2d array of audio data encoded as mid/side, given M/S: a 2d array of audio data encoded as L/R
	'''
	# check for stereo or mid/side array
	if channels == '1':
		return print('%s is mono, import 2 channel audio array for mid side processing.' % name)
	else:
		# divide array into stereo components
		left, right = split(array, channels, name)

		if code:
			# Mid/Side encoding
			mid = 0.5 * (left + right)
			side = 0.5 * (left - right)

			encoded = np.stack((mid, side), axis=-1)

			midside = True

			return encoded, midside

		else:
			# Mid/Side decoding
			newleft = left + right # mid + side
			newright = left - right # mid - side

			decoded = np.stack((newleft, newright), axis=-1)

			midside = False

			return decoded, midside

def invert(array):
	'''
	Inverts the phase (polarity) of an array of audio data
	array: a numpy array of audio data
	returns: a version of array with the polarity inverted
	'''
	# inverts polarity of audio data
	return array * -1

def reverse(array):
	'''
	Reverses an array of audio data
	array: a numpy array of audio data
	returns: a reversed version of array
	'''
	return np.flip(array, 0)

def export_array(name, array, sample_rate, subtype):
	'''
	Export numpy array as audio file
	name: file to write to (truncates & overwrites if file exists) (str, int or file like object)
	array: soundfile object, numpy array of audio data as 64 bit float
	sample_rate: sample rate of the audio data
	subtype: subtype of the audio data
	returns: none
	'''
	sf.write(name, array, sample_rate, subtype)
	return None

# Visualization

def waveform(array, name, channels, sample_rate):
	'''
	array: numpy array of audio data
	name: file name
	channels: mono (1) or stereo (2) file
	sample_rate: sampling rate of audio file
	returns: waveform plot of intensity/time
	'''
	# Sliders for zoom
	# mono
	if channels == '1':
		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')
		fig, ax = plt.subplots()

		# labeling axes & title
		ax.set_title('%s Waveform' % name, fontsize='medium')
		ax.set_xlabel('Time (s)', fontsize='x-small')
		ax.set_ylabel('Amplitude', fontsize='x-small')
		ax.tick_params(axis='both', which='major', labelsize=6)
		ax.margins(0.001)

		# plot signal amplitude/time
		time = array.size / sample_rate # seconds in file
		ax.plot(np.arange(0.0, time, time / array.size), array, color='indigo')

		return plt.show()

	# stereo
	elif channels == '2':
		# divide array into stereo components
		left, right = split(array, channels, name)

		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')
		fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

		# labeling axes & title
		ax1.set_title('%s Waveform' % name, fontsize='medium')
		ax2.set_xlabel('Time (s)', fontsize='x-small')
		ax1.set_ylabel('Amplitude Left', fontsize='x-small')
		ax2.set_ylabel('Amplitude Right', fontsize='x-small')
		ax1.tick_params(axis='both', which='major', labelsize=6)
		ax2.tick_params(axis='both', which='major', labelsize=6)
		ax1.margins(0.001)
		ax2.margins(0.001)
		fig.subplots_adjust(hspace=0)

		# x axis on top
		ax1.xaxis.tick_top()

		# plot signal amplitude/time
		time = array.size / sample_rate
		ax1.plot(np.arange(0.0, time, time / left.size), left, color='indigo')
		ax2.plot(np.arange(0.0, time, time / right.size), right, color='indigo')

		# Multicursor
		multi = MultiCursor(fig.canvas, (ax1, ax2), horizOn=True, color='blueviolet', lw=0.5)

		return plt.show()

def magnitude(array, name, channels, sample_rate, side=None, scale=None):
	'''
	plots the log magnitude spectrum of an audio signal magnitude dB/frequency
	array: array of audio data
	name: audio file name
	channels: 1 mono or 2 stereo
	sample_rate: sampling rate of audio file
	side: l, r or both (l returns plot of left side, r returns right side, both sums sides and returns result) default None
	scale: scaling of values in the spectrum, 'dB' (amplitude (20 * log10)) or 'linear'
	returns: a plot of the log magnitude spectrum of an audio array
	'''
	# Radio Button for L, R, Sum
	# Radio button for Linear, Log
	# Radio Button for Mid, Side

	# adjust right side of plot to make room for button axis
	# create button axis, and buttons
	# create functions for buttons

	# L, R, Sum button
		# need a way to deal with mono

	# divide array into stereo components
	left, right = split(array, channels, name)

	# sum stereo channels
	sumsig = np.sum(array, axis=1)

	# dark background white text, initilize figure and axes
	plt.style.use('dark_background')
	fig, ax = plt.subplots()

	# plotting magnitude spectrum
	spec, fq, line = ax.magnitude_spectrum(sumsig, Fs=sample_rate, scale=scale, color='indigo')

	# labeling axes & title
	title = '%s Magnitude Spectrum' % name
	if scale == 'dB':
		title = '%s Log Magnitude Spectrum' % name
	ax.set_title(title, fontsize='medium')
	ax.set_xlabel('Frequency (hz)', fontsize='x-small')
	ax.set_ylabel('Magnitude (dB)', fontsize='x-small')
	ax.tick_params(axis='both', which='major', labelsize=6)

	# making room for button axis
	plt.subplots_adjust(left=0.225)

	# LRSUM button axis (left, bottom, width, height)
	rax = plt.axes([0.05, 0.7, 0.08, 0.15])

	# LRSUM button
	lrsum = RadioButtons(rax, ('L', 'R', 'Sum'))

	# Side function
	def side(label):
		sidedict = {'L': left, 'R': right, 'Sum': sumsig}
		ydata = sidedict[label]
		line.set_ydata(ydata)
		plt.draw()
	lrsum.on_clicked(side)

	return plt.show()

# Previously under channels == 2:

	# # left
	# if side == 'l':
	# 	magnitude(left, name + ' Left', '1', sample_rate, scale=scale)
	
	# # right
	# elif side == 'r':
	# 	magnitude(right, name + ' Right', '1', sample_rate, scale=scale)
	
	# # sum
	# elif side == 'both':
	# 	magnitude(sumsig, name + ' Sum', '1', sample_rate, scale=scale)

def spectrogram(array, name, channels, sample_rate):
	'''
	Creates a spectrogram given an array of audio data
	array: 1 or 2d numpy array of audio data
	channels: 1 mono or 2 stereo, number of channels in audio array
	name: name of the audio file
	returns a spectrogram with y: frequency decibel scale logarithmic, x: time (seconds)
	'''
	# Sliders for zoom
	# Mono case
	if channels == '1':
		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')
		fig, ax = plt.subplots()

		# labeling axes & title
		ax.set_xlabel('Time (s)', fontsize='x-small')
		ax.set_ylabel('Frequency (kHz)', fontsize='x-small')
		ax.set_title('%s Spectrogram' % name, fontsize='medium')
		ax.tick_params(axis='both', which='major', labelsize=6)
		# ax.grid(True, axis='y', ls=':')
		
		# plot spectrogram
		spec, fq, t, im = ax.specgram(array, Fs= sample_rate, cmap='magma', vmin=-120, vmax=0)
		
		# make space for colorbar
		fig.subplots_adjust(right=0.84)

		# colorbar
		cbar_ax = fig.add_axes([0.85, 0.1125, 0.01, 0.768])	# left, bottom, width, height
		fig.colorbar(im, ticks=np.arange(-120, 0 + 5, 5), cax=cbar_ax).set_label('Amplitude (dB)', fontsize='x-small')
		cbar_ax.tick_params(labelsize=5)
		
		# limit y axis to human hearing range
		ax.set_ylim([0, 20000])

		# fq in kHz
		ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
		ax.yaxis.set_major_formatter(ticks)

		return plt.show()

	# Stereo subplots fasceted
	elif channels == '2':
		# divide array into stereo components
		left, right = split(array, channels, name)
		
		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')
		fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
		
		# labeling axes & title
		ax2.set_xlabel('Time (s)', fontsize='x-small')
		ax1.set_ylabel('Left Frequency (kHz)', fontsize='x-small')
		ax2.set_ylabel('Right Frequency (kHz)', fontsize='x-small')
		ax1.set_title('%s Spectrogram' % name, fontsize='medium')
		ax1.tick_params(axis='both', which='major', labelsize=6)
		ax2.tick_params(axis='both', which='major', labelsize=6)
		# ax1.grid(True, axis='y', ls=':')
		# ax2.grid(True, axis='y', ls=':')

		# x axis on top
		ax1.xaxis.tick_top()
		
		# plot spectrograms
		specl, fql, tl, iml = ax1.specgram(left, Fs=sample_rate, cmap='magma', vmin=-120, vmax=0)
		specr, fqr, tr, imr = ax2.specgram(right, Fs=sample_rate, cmap='magma', vmin=-120, vmax=0)
		
		# make space for colorbar & stack plots snug
		fig.subplots_adjust(right=0.84, hspace=0)
		
		# colorbar
		cbar_ax = fig.add_axes([0.845, 0.11, 0.007, 0.77]) # left, bottom, width, height
		fig.colorbar(iml, ticks=np.arange(-120, 0 + 5, 5), cax=cbar_ax).set_label('Amplitude (dB)', fontsize='x-small')
		cbar_ax.tick_params(labelsize=6)
		
		# limit y axes to human hearing range
		ax1.set_ylim([0, 20000])
		ax2.set_ylim([0, 20000])

		# fq in kHz
		ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
		ax1.yaxis.set_major_formatter(ticks)
		ax2.yaxis.set_major_formatter(ticks)

		# multicursor
		multi = MultiCursor(fig.canvas, (ax1, ax2), horizOn=True, color='blueviolet', lw=0.5)

		return plt.show()

	else:
		return ('Invalid Array')

def vectorscope(array, name, code):
	'''
	A stereo vectorscope polar sample plot of audio data
	Side/Mid amplitudes as coordinates on X/Y 180 degree polar plot
	array: array of audio data
	name: audio datafile name
	code: boolean True if array is encoded as mid/side, false if encoded as L/R
	'''
	if code:
		# dark background white text, initilize polar figure and axes
		plt.style.use('dark_background')
		fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

		# converting cartesian coordinates to polar
		absarray = np.absolute(array)
		r = np.sqrt(np.sum(np.square(array), axis=1))
		theta = np.arctan2(absarray[:,0], array[:,1])
		
		# plotting
		ax.scatter(theta, r, s=0.25, c='indigo')
		ax.set_title('Polar Dot Per Sample Vectorscope of %s' % name)
		ax.set_thetamax(180)
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		ax.grid(False, axis='y')
		ax.set_thetagrids((135.0, 45.0))
		return plt.show()

	else:
		# midside encoding
		msarray, ms = midside(array, channels, name)
		vectorscope(msarray, name, True)

if __name__ == '__main__':
	questions = [inquirer.Checkbox('tests', message='Which tests to run?', choices=['Midside', 'Invert', 'Reverse', 'Waveform', 'Magnitude', 'Spectrogram', 'Vectorscope'],),]
	answers = inquirer.prompt(questions)

	if 'Midside' in answers['tests']:
		# midside encoding test mono
		name, channels, data, subtype, sample_rate = import_array('../binaries/Clap Innerworks 1.wav')
		midside(data, channels, name)

		# midside encoding test stereo
		name, channels, data, subtype, sample_rate = import_array('../binaries/Bottle.aiff')
		encoded, ms = midside(data, channels, name)
		print(encoded, ms)

		# midside decoding test stereo
		decoded, ms = midside(encoded, channels, name, code=False)
		print(decoded, ms)

	if 'Invert' in answers['tests']:
		# Polarity inversion test
		name, channels, data, subtype, sample_rate = import_array('../binaries/Bottle.aiff')
		print(data)
		print(invert(data))

	if 'Reverse' in answers['tests']:
		# Polarity inversion test
		name, channels, data, subtype, sample_rate = import_array('../binaries/Bottle.aiff')
		print(data)
		print(reverse(data))

	if 'Waveform' in answers['tests']:
		# Waveform plot test case mono file
		name, channels, data, subtype, sample_rate = import_array('../binaries/Clap Innerworks 1.wav')
		waveform(data, name, channels, sample_rate)

		# Waveform plot test case stereo file
		name, channels, data, subtype, sample_rate = import_array('../binaries/Bottle.aiff')
		waveform(data, name, channels, sample_rate)

	if 'Magnitude' in answers['tests']:
		# magnitude test mono file
		# name, channels, data, subtype, sample_rate = import_array('../binaries/Clap Innerworks 1.wav')
		# magnitude(data, name, channels, sample_rate)

		# magnitude test stereo file
		name, channels, data, subtype, sample_rate = import_array('../binaries/Bottle.aiff')
		magnitude(data, name, channels, sample_rate, side='both', scale='linear')

		# log magnitude test stereo file
		magnitude(data, name, channels, sample_rate, side='both', scale='dB')

	if 'Spectrogram' in answers['tests']:
		# spectrogram test case mono file
		name, channels, data, subtype, sample_rate = import_array('../binaries/Clap Innerworks 1.wav')
		# prevents divide by zero runtime exception
		data = trim(data)
		spectrogram(data, name, channels, sample_rate)

		# spectrogram test case stereo file
		name, channels, data, subtype, sample_rate = import_array('../binaries/Bottle.aiff')
		# prevents divide by zero runtime exception
		data = trim(data)
		spectrogram(data, name, channels, sample_rate)

	if 'Vectorscope' in answers['tests']:
		# vectorscope stereo test
		name, channels, data, subtype, sample_rate = import_array('../binaries/Bottle.aiff')
		vectorscope(data, name, False)

		# Stereo test left channel only
		name, channels, data, subtype, sample_rate = import_array('../binaries/Bottle.aiff')
		data[:,1] = 0
		vectorscope(data, name, False)