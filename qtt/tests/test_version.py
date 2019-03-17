import unittest

import qtt


# %%


class TestVersion(unittest.TestCase):

    def test_version(self):
        version = qtt.__version__

        self.assertIsInstance(version, str)


if __name__ == '__main__':
    unittest.main()

