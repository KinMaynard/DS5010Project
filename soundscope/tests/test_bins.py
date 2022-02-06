"""
Audio processing & visualization library

Unit Tests
"""

import unittest

import numpy as np

from soundscope.vis.bins import bins


class TestBins(unittest.TestCase):
    
    def test_bins(self):
        """Test bins module."""
        # Len divisible by bin
        downsampled, sample_rate = bins(np.arange(128).reshape(-1, 2), '2',
                                        128)
        self.assertTrue((downsampled == np.array([7.5, 23.5, 39.5, 55.5, 71.5,
                                                  87.5, 103.5, 119.5])).all())
        self.assertEqual(sample_rate, 8.0)
        # Len indivisible by bin, 1d
        down, rate = bins(np.arange(120), '1', 128)
        self.assertTrue((down == np.array([7.5, 23.5, 39.5, 55.5, 71.5, 87.5,
                                          103.5, 115.75])).all())
        self.assertEqual(rate, 8.0)
        # Len indivisible by bin, 2d
        dwn, rte = bins(np.arange(120).reshape(-1, 2), '2', 128)
        self.assertTrue((dwn == np.array([[15., 16.], [47., 48.], [79., 80.],
                                         [107., 108.]])).all())
        self.assertEqual(rte, 8.0)