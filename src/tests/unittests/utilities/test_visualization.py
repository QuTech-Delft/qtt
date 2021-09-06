import unittest
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

import qtt.utilities.visualization


class TestUtilities(unittest.TestCase):

    def test_combine_legends(self):
        plt.figure(1)
        plt.clf()
        ax1 = plt.gca()
        ax1.plot([1, 2], [3, 4], label='a')
        ax2 = ax1.twinx()
        ax2.plot([1, 2], [4, 3], 'r', label='b')
        qtt.utilities.visualization.combine_legends([ax1, ax2], target_ax=ax2)
        plt.close(1)

    def test_combine_legends_empty_input(self):
        plt.figure(1)
        plt.clf()
        ax1 = plt.gca()
        ax1.plot([1, 2], [3, 4], label='a')
        qtt.utilities.visualization.combine_legends([], target_ax=ax1)
        plt.close(1)


class TestVerticalHorizontalLine(unittest.TestCase):

    def test_plot_vertical_line(self):
        mock = Mock()
        mock_ax = Mock()
        mock_ax.axvline = Mock(return_value=mock)
        vline = qtt.utilities.visualization.plot_vertical_line(3., color='r', alpha=.1, label='hi', ax=mock_ax)

        mock_ax.axvline.assert_called_once_with(3., label='hi')
        self.assertEqual(vline, mock)

    def test_plot_horizontal_line(self):
        mock = Mock()
        mock_ax = Mock()
        mock_ax.axhline = Mock(return_value=mock)
        vline = qtt.utilities.visualization.plot_horizontal_line(4., color='r', alpha=.1, label='hi', ax=mock_ax)

        mock_ax.axhline.assert_called_once_with(4., label='hi')
        self.assertEqual(vline, mock)
