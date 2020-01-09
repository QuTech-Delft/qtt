import unittest
import qtt.exceptions


class TestExceptions(unittest.TestCase):

    def test_calibration_exception(self):
        self.assertTrue(issubclass(qtt.exceptions.CalibrationException, Exception))

    def test_PackageVersionWarning(self):
        self.assertTrue(issubclass(qtt.exceptions.PackageVersionWarning, UserWarning))

    def test_MissingOptionalPackageWarning(self):
        self.assertTrue(issubclass(qtt.exceptions.MissingOptionalPackageWarning, UserWarning))
