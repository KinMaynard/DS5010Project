'''
Audio processing & visualization library

Unit Tests
'''

import unittest

import numpy as np

from soundscope.dsp.normalize import normalize
from soundscope.dsp.midside import midside

encoded = np.array([[ 0.5,  0.5], [ 0.5, -0.5]])

class TestDsp(unittest.TestCase):
    
    def test_midside(self):
        coded, ms = midside(np.arange(2), '1')
        self.assertTrue((coded == np.array([[0., 0.], [1., 0.]])).all())
        self.assertEqual(ms, True)
        coded2, ms2 = midside(np.identity(2), '2')
        self.assertTrue((coded2 == encoded).all())
        self.assertEqual(ms2, True)
        coded3, ms3 = midside(encoded, '2', False)
        self.assertTrue((coded3 == np.identity(2)).all())
        self.assertEqual(ms3, False)

    def test_normalize(self):
        array, normal = normalize(np.arange(-50, 75, 25))
        self.assertTrue((array == np.array([-1., -0.5, 0., 0.5, 1.])).all())
        self.assertEqual(normal, True)