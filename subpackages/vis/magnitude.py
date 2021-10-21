'''
Audio processing & visualization library
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, RadioButtons, Button
import matplotlib.gridspec as gridspec
from subpackages.util.split import split
from subpackages.dsp.midside import midside

# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# ignore divide by 0 error in log
np.seterr(divide = 'ignore')

def magnitude(array, name, channels, sample_rate, fig=None, sub=False, gridspec=None, resize_ls=None):
	'''
	plots the log magnitude spectrum of an audio signal magnitude dB/frequency
	
	array: array of audio data
	name: audio file name
	channels: 1 mono or 2 stereo
	sample_rate: sampling rate of audio file
	fig: external figure to plot onto if provided, default = None
	sub: boolean, True: plotting as subplot of larger figure, False: otherwise, default False
	gridspec: gridspec to plot onto if part of a larger figure otherwise None, default None
	resize_ls: list of text objects to be resized on window resize events when plotting inside visualizer, default None

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
	title_mag = ax.set_title(title, color='#F9A438', fontsize=10)
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
			rax = plt.axes([0.06, 0.26, 0.04, 0.0835], facecolor=button_face_color, frame_on=False)

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
			xlabel = ax.set_xlabel('FREQUENCY (HZ)', color='#F9A438', fontsize=7)
			ylabel = ax.set_ylabel('MAGNITUDE (%s)' % state['scale'], color='#F9A438', fontsize=7)
			
			# update state variables to new line & data
			state['line'] = line
			state['data'] = state[label]
			fig.canvas.draw_idle()
		
		# connect button click event to side callback function
		lrsums.on_clicked(side)

		# labelsize & color for LRSUM buttons
		for label in lrsums.labels:
			label.set_fontsize(8)
			label.set_color('#F9A438')

			if resize_ls is not None:
				# add to resize list for resizing in visualizer 
				resize_ls.append(label)

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
		rax = plt.axes([0.06, 0.2, 0.04, 0.05], facecolor=button_face_color, frame_on=False)

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
		xlabel = ax.set_xlabel('FREQUENCY (HZ)', color='#F9A438', fontsize=7)
		ylabel = ax.set_ylabel('MAGNITUDE (%s)' % label, color='#F9A438', fontsize=7)
		
		# update state variables to new line & scale
		state['line'] = line
		state['scale'] = state[label]
		fig.canvas.draw_idle()

	# connect button click event to scale callback function
	lindB.on_clicked(scale)

	# labelsize & color
	for label in lindB.labels:
		label.set_fontsize(8)
		label.set_color('#F9A438')

		if resize_ls is not None:
			# add to resize list for resizing in visualizer 
			resize_ls.append(label)

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
	xlabel = ax.set_xlabel('FREQUENCY (HZ)', color='#F9A438', fontsize=7)
	ylabel = ax.set_ylabel('MAGNITUDE (LIN)', color='#F9A438', fontsize=7)

	# zoom reset view button & axes
	if sub:
		# store initial figure dimesions
		fig_width, fig_height = fig.get_size_inches() * fig.dpi

		# reset button axis size based on figure size to look correct on multiple screens
		if fig_height <= 1700:
			reset_button_ax = fig.add_axes([0.455, 0.07, 0.022, 0.015])

		else:
			reset_button_ax = fig.add_axes([0.463, 0.07, 0.0145, 0.01]) # left, bottom, width, height

		# zoom reset view button
		reset_button = Button(reset_button_ax, 'RESET', color='black', hovercolor='#7E0000')
		
		# small screen, smaller label
		if fig_height <= 1700:
			reset_button.label.set_size(6)

		# big screen, big label
		else:
			reset_button.label.set_size(7)
		
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

	if resize_ls is not None:
		# store text to be resized
		resize_ls.extend([title_mag, xlabel, ylabel, reset_button.label])

	# individual figure or as part of larger figure
	if sub:
		# only return lrsums button if stereo array
		if channels == '2':
			return fig, lrsums, side, lindB, scale, reset_button, reset_button_on_clicked, resize_ls
		else:
			return fig, lindB, scale, reset_button, reset_button_on_clicked, resize_ls
	else:
		return plt.show()