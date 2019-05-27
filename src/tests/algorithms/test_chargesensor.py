""" Tests for analyse effect of sensing dot on fitting of tunnel barrier."""

import unittest
import numpy as np
from qtt.algorithms.chargesensor import DataLinearizer


class TestOneDot(unittest.TestCase):

    @staticmethod
    def test_data_linearizer():
        x = np.arange(0, 10, .1)
        y = x + .05 * x**2
        dl = DataLinearizer(x, y)
        _ = dl.forward([1])
        _ = dl.forward_curve([1])
