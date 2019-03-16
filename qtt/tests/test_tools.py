import sys
import unittest


import qtt.utilities.tools

# %%


class TestMethods(unittest.TestCase):

    def test_rgb_tuple_to_ppt_color(self):
        self.assertEqual(qtt.utilities.tools._rgb_tuple_to_ppt_color( (0,0,0)), 0)
        self.assertEqual(qtt.utilities.tools._rgb_tuple_to_ppt_color( (100,0,0)), 100)
        self.assertEqual(qtt.utilities.tools._rgb_tuple_to_ppt_color( (0,100,0)), 255*100)
        self.assertEqual(qtt.utilities.tools._rgb_tuple_to_ppt_color( (0,0,100)), 255*255*100)

        with self.assertRaises(Exception):
            qtt.utilities.tools._rgb_tuple_to_ppt_color( (-1,0,100))
        with self.assertRaises(Exception):
            qtt.utilities.tools._rgb_tuple_to_ppt_color( (511,0,100))
        with self.assertRaises(Exception):
            qtt.utilities.tools._rgb_tuple_to_ppt_color( 'r')

    def test_get_python_version(self):
        result = qtt.utilities.tools.get_python_version()
        self.assertIn(sys.version, result)


    def test_code_version(self):
        code_version = qtt.utilities.tools.code_version()
        self.assertIsInstance(code_version, dict)
        self.assertIn('python', code_version)
        self.assertIn('version', code_version)
        self.assertIn('numpy', code_version['version'])

    
if __name__ == '__main__':
    unittest.main()

