'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import numpy as np
from subpackages.dsp.normalize import normalize
from subpackages.dsp.midside import midside

sign_arr = np.arange(-1, 2)
p_arr = np.arange(6)
n_arr = np.arange(-4, 0)
dub_sign_arr = np.arange(-1, 3).reshape(2, 2)
dub_p_arr = p_arr.reshape(3, 2)
dub_n_arr = n_arr.reshape(2, 2)

class TestDsp(unittest.TestCase):
	
	def test_midside(self):

	def test_normalize(self):