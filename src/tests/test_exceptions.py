import unittest
import qtt.exceptions

class TestExceptions(unittest.TestCase):

    def test_calibration_exception(self):
        self.assertIsInstance(qtt.exceptions.CalibrationException, BaseException)

    def test_PackageVersionWarning(self):
        self.assertIsInstance(qtt.exceptions.PackageVersionWarning, UserWarning)

    def test_MissingOptionalPackageWarning(self):
        self.assertIsInstance(qtt.exceptions.MissingOptionalPackageWarning, UserWarning)

