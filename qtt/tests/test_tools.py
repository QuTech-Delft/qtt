import unittest


import qtt.utilities.tools
from qtt.utilities.tools import _rgb_tuple_to_ppt_color

# %%


class TestMethods(unittest.TestCase):

    def test_rgb_tuple_to_ppt_color(self):
        self.assertEqual(qtt.utilities.tools._rgb_tuple_to_ppt_color( (0,0,0)), 0)
        self.assertEqual(qtt.utilities.tools._rgb_tuple_to_ppt_color( (100,0,0)), 100)
        self.assertEqual(qtt.utilities.tools._rgb_tuple_to_ppt_color( (0,100,0)), 255*100)
        self.assertEqual(qtt.utilities.tools._rgb_tuple_to_ppt_color( (0,0,100)), 255*255*100)

        with self.assertRaises(Exception):
            qtt.utilities.tools._rgb_tuple_to_ppt_color( (-1,0,100))

if __name__ == '__main__':
    unittest.main()

