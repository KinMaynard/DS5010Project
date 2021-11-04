'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import numpy as np
from subpackages.vis.waveform import waveform
from subpackages.vis.magnitude import magnitude
from subpackages.vis.spectrogram import spectrogram
from subpackages.vis.vectorscope import vectorscope
from subpackages.vis.visualizer import visualizer

# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# ignore divide by 0 error in log
np.seterr(divide = 'ignore')

