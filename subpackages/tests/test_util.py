'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import numpy as np
from subpackages.util.trim import trim
from subpackages.util.split import split
from subpackages.util.invert import invert
from subpackages.util.reverse import reverse

sign_arr = np.arange(-1, 2)
p_arr = np.arange(6)
n_arr = np.arange(-4, 0)
dub_sign_arr = np.arange(-1, 3).reshape(2, 2)
dub_p_arr = p_arr.reshape(2, 2)
dub_n_arr = n_arr.reshape(2, 2)

class TestUtil(unittest.TestCase):
	
	def test_invert(self):
		self.assertEqual(invert(sign_arr), np.negative(sign_arr))
		self.assertEqual(invert(p_arr, np.negative(p_arr)))
		self.assertEqual(invert(n_arr, np.negative(n_arr)))
		self.assertEqual(invert(dub_sign_arr), np.negative(dub_sign_arr))
		self.assertEqual(invert(dub_p_arr, np.negative(dub_p_arr)))
		self.assertEqual(invert(dub_n_arr, np.negative(dub_n_arr)))

	def test_reverse(self):
		self.assertEqual(reverse(p_arr, channels='1'), p_arr[::-1])
		self.assertEqual(reverse(n_arr, channels='1', subdivision=2), np.array([-3, -4, -1, -2]))
		self.assertEqual(reverse(p_arr, channels='1', subdivision=3), np.array([1, 0, 3, 2, 5, 4]))
		