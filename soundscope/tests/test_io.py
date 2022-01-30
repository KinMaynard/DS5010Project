'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import soundfile as sf
import numpy as np
import os
from soundscope.io.import_array import import_array
from soundscope.io.export_array import export_array

class TestIO(unittest.TestCase):
	def setup(self):
		sf.write('tmp.wav', np.arange(-1, 1, .5).reshape(2, 2), 44100, 'PCM_24')
		metadata = ('tmp.wav', '2', array([[-1. , -0.5], [ 0. ,  0.5]]), '[PCM_24]', 44100)

	def test_import_array(self):
		self.assertEqual(import_array('tmp.wav'), metadata)

	def test_export_array(self):
		export_array('tmp.wav', array([[-1. , -0.5], [ 0. ,  0.5]]), 48000, '[PCM_24]')
		name, channels, data, subtype, sample_rate = import_array('tmp.wav')
		self.assertEqual(sample_rate, 48000)

	def tearDown(self):
		os.remove('tmp.wav')