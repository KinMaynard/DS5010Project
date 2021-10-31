'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import numpy as np
from subpackages.dsp.normalize import normalize
from subpackages.dsp.midside import midside

class TestDsp(unittest.TestCase):
	
	def test_midside(self):
		self.assertEqual(midside(array, '1'), )
		self.assertEqual(midside(array, '2'), )
		self.assertEqual(midside(array, '2', False), )

	def test_normalize(self):
		self.assertEqual(normalize(np.arange(-50, 75, 25)), (np.array(-1., -0.5, 0., 0.5, 1.), True))