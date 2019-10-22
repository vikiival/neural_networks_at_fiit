import sys
import unittest

from week_3.tests.accuracy_test import TestAccuracy
from week_3.tests.mlp_test import TestMultilayerPerceptron


def mlp_test():
    """
    Tests for Programming Assignment 3.4
    :return: number of error and failures
    """
    return run_test(TestMultilayerPerceptron)


def accuracy_test():
    """
    Tests for Programming Assignment 3.6
    :return: number of error and failures
    """
    return run_test(TestAccuracy)


def run_test(test_cls):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_cls)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return len(result.errors) + len(result.failures) == 0


if __name__ == '__main__':
    if mlp_test() and accuracy_test():
        sys.exit(0)
    sys.exit(1)
