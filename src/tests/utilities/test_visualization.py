import unittest
from unittest.mock import patch, Mock

import qtt.utilities.visualization


class TestVerticalHorizontalLine(unittest.TestCase):

    def test_plot_vertical_line(self):
        mock = Mock()
        with patch('matplotlib.pyplot.axvline', return_value=mock) as mock_axvline:
            vline = qtt.utilities.visualization.plot_vertical_line(3., color='r', alpha=.1, label='hi')

            mock_axvline.assert_called_once_with(3., label='hi')

        self.assertIsEqual(vline, mock)

    def test_plot_horizontal_line(self):
        mock = Mock()
        with patch('matplotlib.pyplot.axhline', return_value=mock) as mock_axhline:
            vline = qtt.utilities.visualization.plot_horizonal_line(3., color='r', alpha=.1, label='hi')

            mock_axhline.assert_called_once_with(3., label='hi')

        self.assertIsEqual(vline, mock)
