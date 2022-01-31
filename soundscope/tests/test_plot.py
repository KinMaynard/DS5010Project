'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import os
import soundfile as sf
import numpy as np
from soundscope.vis.waveform import waveform
from soundscope.vis.magnitude import magnitude
from soundscope.vis.spectrogram import spectrogram
from soundscope.vis.vectorscope import vectorscope
from soundscope.vis.visualizer import visualizer

# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# ignore divide by 0 error in log
np.seterr(divide = 'ignore')

class TestPlot(unittest.TestCase):

	def setup(self):
		sf.write('tmp.wav', np.arange(-1, 1, .5).reshape(2, 2), 44100, 'PCM_24')
	
	def test_bins(self):

	def test_magnitude(self):

	def test_spectrogram(self):

	def test_vectorscope(self):

	def test_waveform(self):

	def test_vectorscope(self):

	def tearDown(self):
		os.remove('tmp.wav')