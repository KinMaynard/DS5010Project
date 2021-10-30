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

class TestUtil(unittest.TestCase):
	
	def test_invert(self):
		self.assertEqual(invert(np.arange(-1, 2)), np.negative(np.arange(-1, 2)))
		self.assertEqual(invert(np.arange(4), np.negative(np.arange(4))))
		self.assertEqual(invert(np.arange(-4, 4), np.negative(np.arange(-4, 4))))
		self.assertEqual(invert(np.arange(-1, 3).reshape(2, 2)), np.negative(np.arange(-1, 3).reshape(2, 2)))
		self.assertEqual(invert(np.arange(4).reshape(2, 2), np.negative(np.arange(4).reshape(2, 2))))
		self.assertEqual(invert(np.arange(-4, 0).reshape(2, 2), np.negative(np.arange(-4, 0).reshape(2, 2))))

	def test_reverse(self):
		