'''
Audio processing & visualization library
'''

import soundfile as sf
import numpy as np
import sys
import inquirer
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, RadioButtons, Button
import matplotlib.gridspec as gridspec

# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

def import_array(file):
	'''
	Import audio file as 64 bit float array
	file: audio file
	returns: a filename, number of channels, data (a 64 bit float numpy array of audio data), 
			the files subtype and sample rate of the file.
	'''
	# extracting the filename and subtype from soundfile's info object
	info = str(sf.info(file))

	# includes file path
	name_fp = info[:info.find('\n')]
	name = name_fp[name_fp.rfind('/')+1:]
	channels = info[info.find('channels:') + 10:info.find('channels:') + 11]
	subtype = info[info.find('subtype:') + 9:]
	subtype = subtype[subtype.find('['):]

	# reading the audio file as a soundfile numpy array
	data, sample_rate = sf.read(file)

	return name, channels, data, subtype, sample_rate

def bins(array, channels, sample_rate, bin_size=16):
	'''
	array: numpy array of audio data
	channels: 1 mono, 2 stereo
	sample_rate: sampling rate of the audio file
	bin_size: [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], default 32
	returns: downsampled array
	'''
	# either of len bin_size if array divisible by bin_size or length of the partial bin
	partial_bin = len(array) % bin_size
	to_fill = bin_size - partial_bin
	sample_rate = sample_rate / bin_size

	# array didn't need padding
	if partial_bin == 0:
		# separate array into bins of size bin_size, average the bins into new array return array
		# populate new array with averages of every (bin_size) samples
		return np.mean(array.reshape(-1, bin_size), axis=1), sample_rate

	# array needed padding for last bin
	else:
		# pad end of array with mean of last bin so array size divisible by bin_size
		if channels == '1':
			width = ((0, to_fill), )
			padded = np.pad(array, pad_width=width, mode='mean', stat_length=(partial_bin,))
			downsampled = np.mean(padded.reshape(-1, bin_size), axis=1)

		else:
			width = ((0, to_fill), (0, 0))
			padded = np.pad(array, pad_width=width, mode='mean', stat_length=(partial_bin,))
			downsampled = padded.reshape(-1, bin_size, padded.shape[1]).mean(axis=1)
		return downsampled, sample_rate

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
	mask: boolean array of non zeros (non epsilon) values in array
	invalid_value: marker for dimensions of only zeros
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
	mask: boolean array of non zeros (non epsilon) values in array
	invalid_value: marker for dimensions of only zeros, default argument is -1
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
	# FAILS BECAUSE FLOATS CANNOT BE INTERPRETED AS AN INTEGER
	peak = np.amax(np.abs(array))
	#int64
	if -2**31 <= peak <= 2**31-1:
		return ((2.0**31-1.0) / peak) * array
	# int32
	elif -2**15 <= peak <= 2**15-1:
		return ((2.0**15-1.0) / peak) * array
	# float
	elif -1.0 <= peak <= 1.0:
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
		# treat mono array as stereo array of 2 mono components (will sum to only mid data no side)
		left, right = array, array

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

#####################
### Visualization ###
#####################

def waveform(array, name, channels, sample_rate, fig=None, sub=False, gridspec=None):
	'''
	array: array of audio data
	name: file name
	channels: mono (1) or stereo (2) file
	sample_rate: sampling rate of audio file
	fig: external figure to plot onto if provided, default = None
	sub: boolean, True: plotting as subplot of larger figure, False: otherwise, default False
	gridspec: gridspec to plot onto if part of a larger figure otherwise None, default None

	returns: waveform plot of intensity/time either alone or as part of provided fig
	'''
	# Font
	mpl.rcParams['font.family'] = 'sans-serif'
	mpl.rcParams['font.sans-serif'] = 'Helvetica'

	# mono
	if channels == '1':
		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')
		
		# initializing figure and axes
		if fig is None:
			fig, ax = plt.subplots()

		# if plotting on external figure only adding subplot
		else:
			ax = fig.add_subplot(221)

		# labeling axes & title
		title = '%s WAVEFORM' % name
		if sub:
			title = 'WAVEFORM'
		ax.set_title(title, color='#F9A438', fontsize=10)
		ax.set_xlabel('TIME (S)', color='#F9A438', fontsize='x-small')
		ax.set_ylabel('AMPLITUDE', color='#F9A438', fontsize='x-small')
		ax.minorticks_on()
		ax.tick_params(axis='both', which='both', color='#F9A438', labelsize=6, labelcolor='#F9A438')

		# spine coloring
		spine_ls = ['top', 'bottom', 'left', 'right']
		for spine in spine_ls:
			ax.spines[spine].set_color('#F9A438')

		# adding gridline on 0 above data
		ax.axhline(0, color='#F9A438', linewidth=0.5, zorder=3)

		# plot signal amplitude/time
		time = array.size / sample_rate # seconds in file
		line = np.stack((np.linspace(0.0, time, array.size), array), axis=-1)
		col = mpl.collections.LineCollection([line], color='#16F9DA')
		ax.add_collection(col, autolim=True)

		ax.margins(0.001)

		# state variable dictionary of starting axis limits
		state = {'start_xlim': ax.get_xlim(), 'start_ylim': ax.get_ylim()}

		# zoom reset view button & axes
		if sub:
			reset_button_ax = fig.add_axes([0.463, 0.502, 0.0145, 0.01]) # left, bottom, width, height

			# reset button
			reset_button = Button(reset_button_ax, 'RESET', color='black', hovercolor='#7E0000')
			reset_button.label.set_size('x-small')
			reset_button.label.set_color('#F0191C')
			for spine in spine_ls:
				reset_button_ax.spines[spine].set_color('#F0191C')

			# callback function for zoom reset button
			def reset_button_on_clicked(mouse_event):
				ax.set_xlim(state['start_xlim'])
				ax.set_ylim(state['start_ylim'])
			reset_button.on_clicked(reset_button_on_clicked)

		# individual figure or as part of larger figure
		if sub:
			return fig, reset_button, reset_button_on_clicked
		else:
			return plt.show()

	# stereo
	elif channels == '2':
		# divide array into stereo components
		left, right = split(array, channels, name)

		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')
		
		# initializing figure and axes
		if fig is None:
			fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

		# if plotting on external figure only adding subplots
		else:
			ax1 = fig.add_subplot(gridspec[0, 0])
			ax2 = fig.add_subplot(gridspec[1, 0], sharex=ax1, sharey=ax1)

		# labeling axes & title
		title = '%s WAVEFORM' % name
		if sub:
			title = 'WAVEFORM'
		ax1.set_title(title, color='#F9A438', fontsize=10)
		ax2.set_xlabel('TIME (S)', color='#F9A438', fontsize='x-small')
		ax1.set_ylabel('AMPLITUDE LEFT', color='#F9A438', fontsize='x-small')
		ax2.set_ylabel('AMPLITUDE RIGHT', color='#F9A438', fontsize='x-small')
		ax1.minorticks_on()
		ax2.minorticks_on()
		ax1.tick_params(axis='both', which='both', color='#F9A438', labelsize=6, labelcolor='#F9A438')
		ax2.tick_params(axis='both', which='both', color='#F9A438', labelsize=6, labelcolor='#F9A438')

		# adding gridline on 0 above data
		ax1.axhline(0, color='#F9A438', linewidth=0.5, zorder=3)
		ax2.axhline(0, color='#F9A438', linewidth=0.5, zorder=3)

		# spine coloring
		spine_ls = ['top', 'bottom', 'left', 'right']
		for ax, spine in zip([ax1, ax2], spine_ls):
			plt.setp(ax.spines.values(), color='#F9A438')

		# snuggly fasceting subplots if plotting to external figure
		if not sub:
			fig.subplots_adjust(hspace=0)

		# x axis on top
		ax1.xaxis.tick_top()

		# plot signal amplitude/time
		time = left.size / sample_rate # only left size because otherwise will be double the amount of time
		line1 = np.stack((np.linspace(0.0, time, left.size), left), axis=-1)
		line2 = np.stack((np.linspace(0.0, time, left.size), right), axis=-1)
		col1 = mpl.collections.LineCollection([line1], color='#16F9DA')
		col2 = mpl.collections.LineCollection([line2], color='#16F9DA')
		ax1.add_collection(col1, autolim=True)
		ax2.add_collection(col2, autolim=True)

		ax1.margins(0.001)
		ax2.margins(0.001)

		# Multicursor
		multi = MultiCursor(fig.canvas, (ax1, ax2), horizOn=True, color='blueviolet', lw=0.5)

		# state variable dictionary for starting axis limits
		state = {'start_xlim1': ax1.get_xlim(), 'start_ylim1': ax1.get_ylim(), 'start_xlim2': ax2.get_xlim(), 
				'start_ylim2': ax2.get_ylim()}

		# zoom reset view button
		if sub: 
			reset_button_ax = fig.add_axes([0.463, 0.385, 0.0145, 0.01]) # axes left, bottom, width, height
			
			# reset button
			reset_button = Button(reset_button_ax, 'RESET', color='black', hovercolor='#7E0000')
			reset_button.label.set_size('x-small')
			reset_button.label.set_color('#F0191C')
			for spine in spine_ls:
				reset_button_ax.spines[spine].set_color('#F0191C')

			# callback function for zoom reset button
			def reset_button_on_clicked(mouse_event):
				ax1.set_xlim(state['start_xlim1'])
				ax2.set_xlim(state['start_xlim2'])
				ax1.set_ylim(state['start_ylim1'])
				ax2.set_ylim(state['start_ylim2'])
			reset_button.on_clicked(reset_button_on_clicked)

		# individual figure or as part of larger figure
		if sub:
			return fig, reset_button, reset_button_on_clicked
		else:
			return plt.show()

def magnitude(array, name, channels, sample_rate, fig=None, sub=False, gridspec=None):
	'''
	plots the log magnitude spectrum of an audio signal magnitude dB/frequency
	
	array: array of audio data
	name: audio file name
	channels: 1 mono or 2 stereo
	sample_rate: sampling rate of audio file
	fig: external figure to plot onto if provided, default = None
	sub: boolean, True: plotting as subplot of larger figure, False: otherwise, default False
	gridspec: gridspec to plot onto if part of a larger figure otherwise None, default None

	Radio buttons: 
		L: plots left channel, R: plots right channel, Sum: plots L+R, Mid: plots mid channel, Side: plots side channel
		Lin: plot with linear or or no scaling, dB: plot with dB scaling: amplitude (20 * log10)
	
	returns: a plot of the log magnitude spectrum of an audio array with radio buttons for signal array & fq scale
	'''
	# dictionary of state variables
	state = {'LIN': 'linear', 'dB': 'dB', 'scale': 'linear'}

	# dark background white text, initilize figure and axes
	plt.style.use('dark_background')

	# figure and axes init in case of subplot or singular
	if fig is None:
		fig, ax = plt.subplots()

	else:
		if channels == '1':
			ax = fig.add_subplot(223)
		else:
			ax = fig.add_subplot(gridspec[0, 0])

	# Font
	mpl.rcParams['font.family'] = 'sans-serif'
	mpl.rcParams['font.sans-serif'] = 'Helvetica'

	# labeling axes & title
	title = '%s MAGNITUDE SPECTRUM' % name
	if sub:
		title = 'MAGNITUDE SPECTRUM'
	ax.set_title(title, color='#F9A438', fontsize=10)
	ax.minorticks_on()
	ax.tick_params(axis='both', which='both', color='#F9A438', labelsize=6, labelcolor='#F9A438')

	# spine coloring
	spine_ls = ['top', 'bottom', 'left', 'right']
	for spine in spine_ls:
		ax.spines[spine].set_color('#F9A438')

	# mono
	if channels == '1':
		# initial ax
		sig, fq, line = ax.magnitude_spectrum(array, Fs=sample_rate, color='#FB636F')
		state['line'] = line

	# making room for LRSUM &/or Lindb button axes
	if not sub:
		plt.subplots_adjust(left=0.225)

	# adding data & ax state variables
	state.update({'ax': ax, 'data': array})

	# facecolor for button widgets
	button_face_color = 'black'
	
	# Stereo
	if channels == '2':
		# divide array into stereo components
		left, right = split(array, channels, name)

		# sum stereo channels
		sumsig = np.sum(array, axis=1)

		# encoding as midside
		msarray, code = midside(array, channels, name)

		# splitting midside array into mid and side components
		mid, side = split(msarray, channels, name)

		# initial axis
		sig, fq, line = ax.magnitude_spectrum(left, Fs=sample_rate, color='#FB636F')

		# state variable dictionary to keep track of plot status for button changes
		state.update({'L': left, 'R': right, 'SUM': sumsig, 'MID': mid, 'SIDE': side, 'data': left, 'line': line})

		# LRSUM button axis (left, bottom, width, height)
		if not sub:
			rax = plt.axes([0.08, 0.7, 0.08, 0.2], facecolor=button_face_color, frame_on=False)
		else:
			rax = plt.axes([0.08, 0.26, 0.04, 0.0835], facecolor=button_face_color, frame_on=False)

		# LRSUM button
		lrsums = RadioButtons(rax, ('L', 'R', 'SUM', 'MID', 'SIDE'), activecolor='#5C8BC6')

		# Side callback function for lrsums buttons
		def side(label):
			# clear previous data
			state['line'].remove()
			
			# plot
			sig, fq, line = ax.magnitude_spectrum(state[label], Fs=sample_rate, scale=state['scale'], color='#FB636F')
			
			# recompute axis limits
			ax.relim()

			# Set Labels
			ax.set_xlabel('FREQUENCY (HZ)', color='#F9A438', fontsize='x-small')
			ax.set_ylabel('MAGNITUDE (%s)' % state['scale'], color='#F9A438', fontsize='x-small')
			
			# update state variables to new line & data
			state['line'] = line
			state['data'] = state[label]
			fig.canvas.draw_idle()
		
		# connect button click event to side callback function
		lrsums.on_clicked(side)

		# labelsize & color for LRSUM buttons
		for label in lrsums.labels:
			label.set_fontsize('small')
			label.set_color('#F9A438')

		# dynamically resize radio button height with figure size & setting color and width of button edges
		rpos = rax.get_position().get_points()
		fig_height = fig.get_figheight()
		fig_width = fig.get_figwidth()
		rscale = (rpos[:,1].ptp() / rpos[:,0].ptp()) * (fig_height / fig_width)
		for circ in lrsums.circles:
			circ.height /= rscale
			circ.set_edgecolor('#F9A438')
			circ.set_lw(0.5)

	# Linear dB button axis (left, bottom, width, height)
	if not sub:
		rax = plt.axes([0.08, 0.4, 0.08, 0.15], facecolor=button_face_color, frame_on=False)
	else:
		rax = plt.axes([0.08, 0.2, 0.04, 0.05], facecolor=button_face_color, frame_on=False)

	# Linear dB buttons
	lindB = RadioButtons(rax, ('LIN', 'dB'), activecolor='#5C8BC6')

	# state variable dictionary of starting axis limits
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	state.update({'lin_xlim': xlim, 'lin_ylim': ylim, 'dB_xlim': xlim, 'dB_ylim': ylim})

	# scale callback function for lindB buttons
	def scale(label):
		# clear data
		state['line'].remove()
		
		# plot
		sig, fq, line = ax.magnitude_spectrum(state['data'], Fs=sample_rate, scale=state[label], color='#FB636F')
		
		# recompute axis limits
		ax.relim()

		# scale the ax
		ax.autoscale()

		# Set Labels
		ax.set_xlabel('FREQUENCY (HZ)', color='#F9A438', fontsize='x-small')
		ax.set_ylabel('MAGNITUDE (%s)' % label, color='#F9A438', fontsize='x-small')
		
		# update state variables to new line & scale
		state['line'] = line
		state['scale'] = state[label]
		fig.canvas.draw_idle()

	# connect button click event to scale callback function
	lindB.on_clicked(scale)

	# labelsize & color
	for label in lindB.labels:
		label.set_fontsize('small')
		label.set_color('#F9A438')

	# dynamically resize radio button height with figure size
	rpos = rax.get_position().get_points()
	fh = fig.get_figheight()
	fw = fig.get_figwidth()
	rscale = (rpos[:,1].ptp() / rpos[:,0].ptp()) * (fh / fw)
	for circ in lindB.circles:
		circ.height /= rscale
		circ.set_edgecolor('#F9A438')
		circ.set_lw(0.5)

	# Axis Labels
	ax.set_xlabel('FREQUENCY (HZ)', color='#F9A438', fontsize='x-small')
	ax.set_ylabel('MAGNITUDE (LIN)', color='#F9A438', fontsize='x-small')

	# zoom reset view button & axes
	if sub:
		reset_button_ax = fig.add_axes([0.463, 0.082, 0.0145, 0.01]) # left, bottom, width, height

		# zoom reset view button
		reset_button = Button(reset_button_ax, 'RESET', color='black', hovercolor='#7E0000')
		reset_button.label.set_size('x-small')
		reset_button.label.set_color('#F0191C')
		for spine in spine_ls:
			reset_button_ax.spines[spine].set_color('#F0191C')

		# callback function for zoom reset button
		def reset_button_on_clicked(mouse_event):
			# recompute axis limits
			ax.relim()

			# scale the ax
			ax.autoscale()
		reset_button.on_clicked(reset_button_on_clicked)

	# individual figure or as part of larger figure
	if sub:
		if channels == '2':
			return fig, lrsums, side, lindB, scale, reset_button, reset_button_on_clicked
		else:
			return fig, lindB, scale, reset_button, reset_button_on_clicked
	else:
		return plt.show()

def spectrogram(array, name, channels, sample_rate, fig=None, sub=False, gridspec=None):
	'''
	Creates a spectrogram given an array of audio data
	
	array: 1 or 2d numpy array of audio data
	channels: 1 mono or 2 stereo, number of channels in audio array
	name: name of the audio file
	fig: external figure to plot onto if provided, default = None
	sub: boolean, True: plotting as subplot of larger figure, False: otherwise, default False
	gridspec: gridspec to plot onto if part of a larger figure otherwise None, default None

	returns a spectrogram with y: frequency decibel scale logarithmic, x: time (seconds)
	'''
	# Font
	mpl.rcParams['font.family'] = 'sans-serif'
	mpl.rcParams['font.sans-serif'] = 'Helvetica'

	# Mono case
	if channels == '1':
		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')
		
		# figure and axes init in case of subplot or singular
		if fig is None:
			fig, ax = plt.subplots()

		else:
			ax = fig.add_subplot(222)

		# labeling axes & title
		title = '%s SPECTROGRAM' % name
		if sub:
			title = 'SPECTROGRAM'
		ax.set_xlabel('TIME (S)', color='#F9A438', fontsize='x-small')
		ax.set_ylabel('FREQUENCY (KHZ)', color='#F9A438', fontsize='x-small')
		ax.set_title(title, color='#F9A438', fontsize=10)
		ax.minorticks_on()
		ax.tick_params(axis='both', which='both', color='#F9A438', labelsize=6, labelcolor='#F9A438')

		# spine coloring
		spine_ls = ['top', 'bottom', 'left', 'right']
		for spine in ['top', 'bottom', 'left', 'right']:
			ax.spines[spine].set_color('#F9A438')
		
		# plot spectrogram
		spec, fq, t, im = ax.specgram(array, Fs= sample_rate, cmap='magma', vmin=-120, vmax=0)
		
		# make space for colorbar
		if not sub:
			fig.subplots_adjust(right=0.84)

		# colorbar
		if not sub:
			cbar_ax = fig.add_axes([0.85, 0.1125, 0.01, 0.768])	# left, bottom, width, height
		else:
			cbar_ax = fig.add_axes([0.905, 0.53, 0.003, 0.35])	# left, bottom, width, height
		fig.colorbar(im, ticks=np.arange(-120, 0 + 5, 5), cax=cbar_ax).set_label('AMPLITUDE (dB)', color='#F9A438', fontsize='x-small')
		cbar_ax.tick_params(color='#F9A438', labelsize=5, labelcolor='#F9A438')
		
		# limit y axis to human hearing range
		ax.set_ylim([0, 20000])

		# fq in kHz
		ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
		ax.yaxis.set_major_formatter(ticks)

		# state variable dictionary of starting axis limits
		state = {'start_xlim': ax.get_xlim(), 'start_ylim': ax.get_ylim()}

		# zoom reset view button & axes
		if sub:
			reset_button_ax = fig.add_axes([0.886, 0.502, 0.0145, 0.01]) # left, bottom, width, height

			# reset button
			reset_button = Button(reset_button_ax, 'RESET', color='black', hovercolor='#7E0000')
			reset_button.label.set_size('x-small')
			reset_button.label.set_color('#F0191C')
			for spine in spine_ls:
				reset_button_ax.spines[spine].set_color('#F0191C')
			
			# callback function for zoom reset button
			def reset_button_on_clicked(mouse_event):
				ax.set_xlim(state['start_xlim'])
				ax.set_ylim(state['start_ylim'])
			reset_button.on_clicked(reset_button_on_clicked)

		# individual figure or as part of larger figure
		if sub:
			return fig, reset_button, reset_button_on_clicked
		else:
			return plt.show()

	# Stereo subplots fasceted
	elif channels == '2':
		# divide array into stereo components
		left, right = split(array, channels, name)
		
		# dark background white text, initilize figure and axes
		plt.style.use('dark_background')

		# figure and axes init in case of subplot or singular
		if fig is None:
			fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

		else:
			ax1 = fig.add_subplot(gridspec[0, 1])
			ax2 = fig.add_subplot(gridspec[1, 1], sharex=ax1, sharey=ax1)
		
		# labeling axes & title
		title = '%s SPECTROGRAM' % name
		if sub:
			title = 'SPECTROGRAM'
		ax2.set_xlabel('TIME (S)', color='#F9A438', fontsize='x-small')
		ax1.set_ylabel('LEFT FREQUENCY (KHZ)', color='#F9A438', fontsize='x-small')
		ax2.set_ylabel('RIGHT FREQUENCY (KHZ)', color='#F9A438', fontsize='x-small')
		ax1.set_title(title, color='#F9A438', fontsize=10)
		ax1.minorticks_on()
		ax2.minorticks_on()
		ax1.tick_params(axis='both', which='both', color='#F9A438', labelsize=6, labelcolor='#F9A438')
		ax2.tick_params(axis='both', which='both', color='#F9A438', labelsize=6, labelcolor='#F9A438')

		# x axis on top
		ax1.xaxis.tick_top()

		# spine coloring
		spine_ls = ['top', 'bottom', 'left', 'right']
		for ax, spine in zip([ax1, ax2], spine_ls):
			plt.setp(ax.spines.values(), color='#F9A438')
		
		# plot spectrograms
		specl, fql, tl, iml = ax1.specgram(left, Fs=sample_rate, cmap='magma', vmin=-120, vmax=0)
		specr, fqr, tr, imr = ax2.specgram(right, Fs=sample_rate, cmap='magma', vmin=-120, vmax=0)
		
		# make space for colorbar & stack plots snug
		if not sub:
			fig.subplots_adjust(right=0.84, hspace=0)
		
		# colorbar
		if not sub:
			cbar_ax = fig.add_axes([0.845, 0.11, 0.007, 0.77]) # left, bottom, width, height
		else:
			cbar_ax = fig.add_axes([0.905, 0.414, 0.003, 0.466]) # left, bottom, width, height
		fig.colorbar(iml, ticks=np.arange(-120, 0 + 5, 5), cax=cbar_ax).set_label('AMPLITUDE (dB)', color='#F9A438', fontsize='x-small')
		cbar_ax.tick_params(color='#F9A438', labelsize=6, labelcolor='#F9A438')
		
		# limit y axes to human hearing range
		ax1.set_ylim([0, 20000])
		ax2.set_ylim([0, 20000])

		# fq in kHz
		ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
		ax1.yaxis.set_major_formatter(ticks)
		ax2.yaxis.set_major_formatter(ticks)

		# multicursor
		multi = MultiCursor(fig.canvas, (ax1, ax2), horizOn=True, color='blueviolet', lw=0.5)

		# state variable dictionary for starting axis limits
		state = {'start_xlim1': ax1.get_xlim(), 'start_ylim1': ax1.get_ylim(), 'start_xlim2': ax2.get_xlim(), 'start_ylim2': ax2.get_ylim()}

		# zoom reset view button
		if sub: 
			reset_button_ax = fig.add_axes([0.886, 0.385, 0.0145, 0.01]) # axes left, bottom, width, height
			
			# reset button
			reset_button = Button(reset_button_ax, 'RESET', color='black', hovercolor='#7E0000')
			reset_button.label.set_size('x-small')
			reset_button.label.set_color('#F0191C')
			for spine in spine_ls:
				reset_button_ax.spines[spine].set_color('#F0191C')
			
			# callback function for zoom reset button
			def reset_button_on_clicked(mouse_event):
				ax1.set_xlim(state['start_xlim1'])
				ax2.set_xlim(state['start_xlim2'])
				ax1.set_ylim(state['start_ylim1'])
				ax2.set_ylim(state['start_ylim2'])
			reset_button.on_clicked(reset_button_on_clicked)

		# individual figure or as part of larger figure
		if sub:
			return fig, reset_button, reset_button_on_clicked
		else:
			return plt.show()

def vectorscope(array, name, code, fig=None, sub=False, gridspec=None):
	'''
	A stereo vectorscope polar sample plot of audio data
	Side/Mid amplitudes as coordinates on X/Y 180 degree polar plot
	
	array: array of audio data
	name: audio datafile name
	code: boolean True if array is encoded as mid/side, false if encoded as L/R
	fig: external figure to plot onto if provided, default = None
	sub: boolean, True: plotting as subplot of larger figure, False: otherwise, default False
	gridspec: gridspec to plot onto if part of a larger figure otherwise None, default None
	
	returns: a vectorscope polar dot per sample plot of audio data
	'''
	if code:
		# dark background white text, initilize polar figure and axes
		plt.style.use('dark_background')

		if fig is None:
			fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

		else:
			if channels == '1':
				ax = fig.add_subplot(224, polar=True)
			else:
				ax = fig.add_subplot(gridspec[0, 1], polar=True)

		# Font
		mpl.rcParams['font.family'] = 'sans-serif'
		mpl.rcParams['font.sans-serif'] = 'Helvetica'

		# converting cartesian coordinates to polar
		absarray = np.absolute(array)
		r = np.sqrt(np.sum(np.square(array), axis=1))
		theta = np.arctan2(absarray[:,0], array[:,1])
		
		# plotting
		title = '%s POLAR DOT PER SAMPLE VECTORSCOPE' % name
		if sub:
			title = 'POLAR DOT PER SAMPLE VECTORSCOPE'
		ax.scatter(theta, r, s=0.25, c='#4B9D39')
		
		# set title & bring down close to top of plot
		if sub:
			if channels == '1':
				ax.set_title(title, color='#F9A438', fontsize=10, pad=-140)
			else:
				ax.set_title(title, color='#F9A438', fontsize=10, pad=-105)
		else:
			ax.set_title(title, color='#F9A438', fontsize='medium', pad=-70)

		# plotting 180 degrees
		ax.set_thetamax(180)
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		ax.grid(False, axis='y')
		ax.spines['polar'].set_color('#F9A438')

		# plotting only 2 theta grids
		ax.set_thetagrids((135.0, 45.0))

		# thetagrid color
		ax.xaxis.grid(color='#F9A438')

		# compensating for partial polar plot extra whitespace: left, bottom, width, height
		if sub is False:
			ax.set_position([0.1, 0.05, 0.8, 1])

		else:
			if channels == '1':
				ax.set_position([0.55, -0.735, 0.350, 2.023]) # left, bottom, width, height
			else:
				ax.set_position([0.6, -0.772, 0.245, 2])

		# individual figure or as part of larger figure
		if sub:
			return fig
		else:
			return plt.show()

	else:
		# midside encoding
		msarray, ms = midside(array, channels, name)
		vectorscope(msarray, name, True, fig, sub, gridspec)

def visualizer(array, name, channels, sample_rate, code):
	'''
	array: numpy array of audio data
	name: file name
	channels: mono (1) or stereo (2) file
	sample_rate: sampling rate of audio file
	code: boolean True if array is encoded as mid/side, false if encoded as L/R
	
	returns: fasceted subplots of waveform, magnitude, spectrogram & vectorscope
	'''
	# initialize figure with dark background and title
	plt.style.use('dark_background')
	fig = plt.figure() # figsize=(26, 13.5)
	figmanager = plt.get_current_fig_manager()
	figmanager.window.showMaximized()

	# Font
	mpl.rcParams['font.family'] = 'sans-serif'
	mpl.rcParams['font.sans-serif'] = 'Helvetica'
	
	# Title
	plt.suptitle('%s VISUALIZATION' % name, color='#F9A438', fontsize=17.5, fontweight=900)

	# gridspec to snugly fascet only stereo spectrogram and waveform plots
	# initialize for mono case
	gs1, gs2 = None, None
	if channels == '2':
		# snugly fascet stereo subplots
		fig.subplots_adjust(hspace=0)

		# outer gridspec, hspace separates waveform & spectrogram plots from magnitude & vectorscope
		outer = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, hspace = 0.2, height_ratios = [2, 1])

		# nested gridspecs
		gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec = outer[0])
		gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[1])

		# stereo mag plot with side button
		fig, lrsums, side, lindB, scale, reset_mag, reset_mag_click = magnitude(array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs2)
		
		# enabling mag buttons
		lrsums.on_clicked(side)

	else:
		# mono mag plot without side button
		fig, lindB, scale, reset_mag, reset_mag_click = magnitude(array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs2)
	
	# subplots currently multi_spec only shows
	fig, reset_wav, reset_wav_click = waveform(array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs1)
	fig, reset_spec, reset_spec_click = spectrogram(array, name, channels, sample_rate, fig=fig, sub=True, gridspec=gs1)
	fig = vectorscope(array, name, code, fig=fig, sub=True, gridspec=gs2)

	# enabling mag buttons
	lindB.on_clicked(scale)

	# enabling view reset buttons
	reset_wav.on_clicked(reset_wav_click)
	reset_spec.on_clicked(reset_spec_click)
	reset_mag.on_clicked(reset_mag_click)

	plt.show()

if __name__ == '__main__':
	# Test selector
	questions = [inquirer.Checkbox('tests', message='Which tests to run?', 
		choices=['Mono', 'Stereo', 'Downsample', 'Bins', 'Normalize', 'Midside', 'Invert', 'Reverse', 'Waveform', 'Magnitude', 'Spectrogram', 
		'Vectorscope', 'Visualizer'],),]

	answers = inquirer.prompt(questions)

	if 'Mono' in answers['tests']:
		# Waveform to perform tests on
		questions2 = [inquirer.List('waves', message='Which mono wave to test?', 
			choices=[('Silence', '../binaries/silence_44100_-infdBFS_Mono.aiff'), ('White Noise', '../binaries/white_88k_-3dBFS.wav'), 
			('Linear Chirp', '../binaries/hdchirp_88k_-3dBFS_lin.wav'), ('Sin 100Hz', '../binaries/sin_44100_100Hz_-3dBFS_1s.wav'), 
			('Sweep 1-44kHz', '../binaries/hdsweep_1Hz_44000Hz_-3dBFS_30s.wav')],
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
			data = normalize(data)
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
			# Reverse array test
			print(data)
			print(reverse(data))

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
			# prevents divide by zero runtime exception
			data = trim(data)
			spectrogram(data, name, channels, sample_rate)

		if 'Vectorscope' in answers['tests']:
			# vectorscope mono test
			vectorscope(data, name, False)

		if 'Visualizer' in answers['tests']:
			# visualizer mono plot
			visualizer(data, name, channels, sample_rate, code=False)

	if 'Stereo' in answers['tests']:
		# Waveform to perform tests on
		questions2 = [inquirer.List('waves', message='Which stereo wave to test?', 
			choices=[('Silence Stereo', '../binaries/silence_44100_-infdBFS_Stereo.aiff'), ('White Noise Stereo', '../binaries/whitenoise_44100_0dBFS_Stereo.aiff'), 
			('Chirp Stereo', '../binaries/hdchirp_88k_-3dBFS_lin_Stereo.aiff'), ('Sin 440Hz Stereo', '../binaries/sin_44100_440Hz_-.8dBFS_Stereo.aiff'), 
			('Lopez Song Stereo', '../binaries/Saija Original Mix.aiff')],
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
			data = normalize(data)
			# after normalization
			waveform(data, name, channels, sample_rate)

		if 'Midside' in answers['tests']:
			# midside encoding test stereo
			encoded, ms = midside(data, channels, name)
			print(encoded, ms)

			# midside decoding test stereo
			decoded, ms = midside(encoded, channels, name, code=False)
			print(decoded, ms)

		if 'Invert' in answers['tests']:
			# Polarity inversion test
			print(data)
			print(invert(data))

		if 'Reverse' in answers['tests']:
			# Reverse array test
			print(data)
			print(reverse(data))

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
			# prevents divide by zero runtime exception
			data = trim(data)
			spectrogram(data, name, channels, sample_rate)

		if 'Vectorscope' in answers['tests']:
			# vectorscope stereo test
			vectorscope(data, name, False)

			# Stereo test left channel only
			data[:,1] = 0
			vectorscope(data, name, False)

		if 'Visualizer' in answers['tests']:
			# visualizer stereo plot
			visualizer(data, name, channels, sample_rate, code=False)

