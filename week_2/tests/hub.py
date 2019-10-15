import sys
import unittest

from week_2.tests.model_test import TestLinearRegressionModel
from week_2.tests.sgd_test import TestSGD


def model_test():
    """
    Tests for Programming Assignment 2.9
    :return: number of error and failures
    """
    return run_test(TestLinearRegressionModel)


def sgd_test():
    """
    Tests for Programming Assignment 2.13
    :return: number of error and failures
    """
    return run_test(TestSGD)


def run_test(test_cls):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_cls)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return len(result.errors) + len(result.failures) == 0


if __name__ == '__main__':
    if model_test() and sgd_test():
        sys.exit(0)
    sys.exit(1)
