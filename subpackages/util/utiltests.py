'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import numpy as np
from subpackages.io.import_array import import_array
from subpackages.util.trim import trim
from subpackages.util.split import split
from subpackages.util.invert import invert
from subpackages.util.reverse import reverse

class TestInvert(unittest.TestCase):
	