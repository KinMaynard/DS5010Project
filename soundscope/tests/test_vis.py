'''
Audio processing & visualization library

Unit Tests
'''

import unittest
import numpy as np
from soundscope.vis.waveform import waveform
from soundscope.vis.magnitude import magnitude
from soundscope.vis.spectrogram import spectrogram
from soundscope.vis.vectorscope import vectorscope
from soundscope.vis.visualizer import visualizer

# use backend that supports animation, blitting & figure window resizing
mpl.use('Qt5Agg')

# ignore divide by 0 error in log
np.seterr(divide = 'ignore')

