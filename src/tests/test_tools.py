import sys
import unittest
from unittest.mock import MagicMock

from qtt.utilities import tools


class TestMethods(unittest.TestCase):

    def test_get_python_version(self):
        result = tools.get_python_version()
        self.assertIn(sys.version, result)

    def test_code_version(self):
        code_version = tools.code_version()
        self.assertIsInstance(code_version, dict)
        self.assertIn('python', code_version)
        self.assertIn('version', code_version)
        self.assertIn('numpy', code_version['version'])

    def test_rgb_tuple_to_integer(self):
        white_color = (0, 0, 0)
        self.assertEqual(0, tools._convert_rgb_color_to_integer(white_color))

        light_red_color = (100, 0, 0)
        self.assertEqual(100, tools._convert_rgb_color_to_integer(light_red_color))

        light_green_color = (0, 100, 0)
        self.assertEqual(256 * 100, tools._convert_rgb_color_to_integer(light_green_color))

        light_blue_color = (0, 0, 100)
        self.assertEqual(256 * 256 * 100, tools._convert_rgb_color_to_integer(light_blue_color))

        black_color = (255, 255, 255)
        self.assertEqual(256 * 256 * 256 - 1, tools._convert_rgb_color_to_integer(black_color))

    def test_rgb_tuple_to_integer_raises_error(self):
        self.assertRaises(ValueError, tools._convert_rgb_color_to_integer, 'r')
        self.assertRaises(ValueError, tools._convert_rgb_color_to_integer, (0, 0, 0, 0))
        self.assertRaises(ValueError, tools._convert_rgb_color_to_integer, (-1, 0, 100))
        self.assertRaises(ValueError, tools._convert_rgb_color_to_integer, (511, 0, 100))

    def test_integer_to_rgb_tuple(self):
        white_color = 0
        self.assertEqual((0, 0, 0), tools._covert_integer_to_rgb_color(white_color))

        red_color = 256 - 1
        self.assertEqual((255, 0, 0), tools._covert_integer_to_rgb_color(red_color))

        green_color = (256 - 1) * 256
        self.assertEqual((0, 255, 0), tools._covert_integer_to_rgb_color(green_color))

        blue_color = (256 - 1) * 256 * 256
        self.assertEqual((0, 0, 255), tools._covert_integer_to_rgb_color(blue_color))

        black_color = 256 * 256 * 256 - 1
        self.assertEqual((255, 255, 255), tools._covert_integer_to_rgb_color(black_color))

    def test_integer_to_rgb_tuple_raises_error(self):
        self.assertRaises(ValueError, tools._convert_rgb_color_to_integer, -1)
        self.assertRaises(ValueError, tools._convert_rgb_color_to_integer, 256 * 256 * 256)

    def test_covert_rgb_and_back(self):
        expected_green_color = 256 * 256 * 256 - 1
        rgb_green = tools._covert_integer_to_rgb_color(expected_green_color)
        actual_color = tools._convert_rgb_color_to_integer(rgb_green)
        self.assertEqual(expected_green_color, actual_color)

        expected_black_color = 256 * 256 * 256 - 1
        rgb_black = tools._covert_integer_to_rgb_color(expected_black_color)
        actual_color = tools._convert_rgb_color_to_integer(rgb_black)
        self.assertEqual(expected_black_color, actual_color)

    def test_set_ppt_slide_background(self):
        slide = MagicMock()
        slide.Background.Fill.ForeColor.RGB = 0
        color = (255, 0, 0)

        tools.set_ppt_slide_background(slide, color)
        self.assertEqual(0, slide.FollowMasterBackground)
        self.assertEqual(255, slide.Background.Fill.ForeColor.RGB)

if __name__ == '__main__':
    unittest.main()
