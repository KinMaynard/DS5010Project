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

array = np.arange(-1, 1)

class TestUtil(unittest.TestCase):
	
	def test_invert(self):
		self.assertEqual(invert(array), np.array([1, 0, 1]))