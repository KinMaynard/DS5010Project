import unittest

# import test modules

from soundscope.tests import test_util
from soundscope.tests import test_bins
from soundscope.tests import test_dsp
from soundscope.tests import test_io


# initialize test suite

loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to suite

suite.addTest(loader.loadTestsFromModule(test_util))
suite.addTest(loader.loadTestsFromModule(test_bins))
suite.addTest(loader.loadTestsFromModule(test_io))
suite.addTest(loader.loadTestsFromModule(test_dsp))

# initialize test runner, run the suite

runner = unittest.TextTestRunner()
result = runner.run(suite)