import sys
import unittest
from unittest.mock import MagicMock, patch
import warnings
import numpy as np
import qcodes
from qcodes.data.data_array import DataArray
from qcodes.tests.legacy.data_mocks import DataSet2D
from qtt.utilities import tools
from qtt.utilities.tools import resampleImage, diffImage, diffImageSmooth, reshape_metadata, get_python_version, \
    get_module_versions, get_git_versions, code_version, rdeprecated, in_ipynb, pythonVersion
import qtt.measurements.scans


class TestTools(unittest.TestCase):

    def test_python_code_modules_and_versions(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="qupulse")
            _ = get_python_version()
            _ = get_module_versions(['numpy'])
            _ = get_git_versions(['qtt'])
            c = code_version()
            self.assertTrue('python' in c)
            self.assertTrue('timestamp' in c)
            self.assertTrue('system' in c)

    @staticmethod
    def test_rdeprecated():
        @rdeprecated('hello')
        def dummy():
            pass

        @rdeprecated('hello', expire='1-1-2400')
        def dummy2():
            pass

    def test_array(self):
        # DataSet with one 2D array with 4 x 6 points
        yy, xx = np.meshgrid(np.arange(0, 10, .5), range(3))
        zz = xx ** 2 + yy ** 2
        # outer setpoint should be 1D
        xx = xx[:, 0]
        x = DataArray(name='x', label='X', preset_data=xx, is_setpoint=True)
        y = DataArray(name='y', label='Y', preset_data=yy, set_arrays=(x,),
                      is_setpoint=True)
        z = DataArray(name='z', label='Z', preset_data=zz, set_arrays=(x, y))
        self.assertTrue(z.size, 60)
        self.assertTupleEqual(z.shape, (3, 20))
        return z

    def test_image_operations(self, verbose=0):
        if verbose:
            print('testing resampleImage')
        ds = DataSet2D()
        _, _ = resampleImage(ds.z)

        z = self.test_array()
        _, _ = resampleImage(z)
        if verbose:
            print('testing diffImage')
        _ = diffImage(ds.z, dy='x')

    def test_in_ipynb(self):
        running_in_ipynb = in_ipynb()
        self.assertFalse(running_in_ipynb)

    def test_pythonVersion(self):
        with patch('builtins.print') as mock_print:
            pythonVersion()

        y = str(mock_print.call_args)
        self.assertIn('python', y)
        self.assertIn('ipython', y)
        self.assertIn('notebook', y)

    def test_diffImageSmooth(self):
        image = np.arange(12.).reshape(4, 3)
        diff_image = diffImageSmooth(image, dy='x')
        np.testing.assert_array_almost_equal(diff_image, np.array(
            [[0.346482, 0.390491, 0.346482], [0.346482, 0.390491, 0.346482], [0.346482, 0.390491, 0.346482],
             [0.346482, 0.390491, 0.346482]]))

        with self.assertRaises(Exception):
            diffImageSmooth(np.arange(4), dy='x')

    def test_reshape_metadata(self, quiet=True):
        param = qcodes.ManualParameter('dummy')
        data_set = qcodes.loops.Loop(param[0:1:10]).each(param).run(quiet=quiet)

        metadata = reshape_metadata(data_set, printformat='dict')
        self.assertTrue(metadata.startswith('dataset'))

        data_set.metadata['scanjob'] = {'scanjobdict': True}
        metadata = reshape_metadata(data_set, printformat='dict')
        self.assertIn('scanjobdict', metadata)

    def test_reshape_metadata_station(self):
        instr = qcodes.Instrument(qtt.measurements.scans.instrumentName('_dummy_test_reshape_metadata_123'))
        st = qcodes.Station(instr)
        result = reshape_metadata(st, printformat='dict')
        instr.close()
        self.assertTrue('_dummy_test_reshape_metadata_123' in result)
        self.assertTrue(isinstance(result, str))
        self.assertFalse('(' in result)
        self.assertFalse(')' in result)

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
