import unittest

# import test modules

from subpackages.tests import test_util

# initialize test suite

loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to suite

suite.addTest(loader.loadTestsFromModule(test_util))

# initialize test runner, run the suite

runner = unittest.TextTestRunner()
result = runner.run(suite)