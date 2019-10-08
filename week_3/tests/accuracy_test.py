import unittest

import numpy as np

from week_3.mlp import MultilayerPerceptron


class TestAccuracy(unittest.TestCase):

    def setUp(self):
        self.model = MultilayerPerceptron(2, 10, 5)
        np.random.seed(0)
        self.model.b_1 = np.random.rand(10)
        self.model.w_1 = np.random.rand(10, 2)
        self.model.b_2 = np.zeros(5)
        self.model.w_2 = np.random.rand(5, 10)

        self.xs = np.random.rand(200, 2)
        self.ys = np.zeros((200, 5))
        self.ys[np.arange(200), np.random.randint(0, 5, 200)] = 1

    def test_accuracy(self):
        self.assertAlmostEqual(0.22, self.model.accuracy(self.xs, self.ys))
