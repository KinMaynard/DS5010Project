import unittest

# import test modules

from subpackages.tests import test_util
from subpackages.tests import test_bins
from subpackages.tests import test_dsp
from subpackages.tests import test_io
from subpackages.tests import test_plot

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