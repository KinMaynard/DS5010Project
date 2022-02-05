"""
Audio processing & visualization library

Unit Tests
"""

import unittest

import numpy as np

from soundscope.util.trim import mask
from soundscope.util.trim import last_nonzero
from soundscope.util.trim import first_nonzero
from soundscope.util.trim import trim
from soundscope.util.split import split
from soundscope.util.invert import invert
from soundscope.util.reverse import reverse


sign_arr = np.arange(-1, 2)
p_arr = np.arange(6)
n_arr = np.arange(-4, 0)
dub_sign_arr = np.arange(-1, 3).reshape(2, 2)
dub_p_arr = p_arr.reshape(3, 2)
dub_n_arr = n_arr.reshape(2, 2)


class TestUtil(unittest.TestCase):
    
    def test_invert(self):
        self.assertTrue((invert(sign_arr) == np.negative(sign_arr)).all())
        self.assertTrue((invert(p_arr) == np.negative(p_arr)).all())
        self.assertTrue((invert(n_arr) == np.negative(n_arr)).all())
        self.assertTrue((invert(dub_sign_arr)
                         == np.negative(dub_sign_arr)).all())
        self.assertTrue((invert(dub_p_arr) == np.negative(dub_p_arr)).all())
        self.assertTrue((invert(dub_n_arr) == np.negative(dub_n_arr)).all())

    def test_reverse(self):
        self.assertTrue((reverse(p_arr, channels='1') == p_arr[::-1]).all())
        self.assertTrue((reverse(n_arr, channels='1', subdivision=2)
                        == np.array([-3, -4, -1, -2])).all())
        self.assertTrue((reverse(p_arr, channels='1', subdivision=3)
                        == np.array([1, 0, 3, 2, 5, 4])).all())
        self.assertTrue((reverse(dub_p_arr, channels='2')
                        == np.array([[4, 5], [2, 3], [0, 1]])).all())

    def test_split(self):
        left, right = split(dub_n_arr, channels='2')
        self.assertTrue((left == np.array([-4, -2])).all())
        self.assertTrue((right == np.array([-3, -1])).all())

    def test_mask(self):
        self.assertTrue((mask(sign_arr)
                        == np.array([True, False, True])).all())

    def test_lastnonzero(self):
        self.assertEqual(last_nonzero(sign_arr, 0, mask(sign_arr)), 2)

    def test_firstnonzero(self):
        self.assertEqual(first_nonzero(sign_arr, 0, mask(sign_arr)), 0)

    def test_trim(self):
        self.assertEqual(trim(np.array([0, 1, 0])), np.array([1]))
        self.assertTrue((trim(np.array([[0, 0], [1, 0]]))
                        == np.array([[1, 0]])).all())