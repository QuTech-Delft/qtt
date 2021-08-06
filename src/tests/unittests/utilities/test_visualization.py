import unittest
from unittest.mock import Mock, patch

import qtt.utilities.visualization


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
