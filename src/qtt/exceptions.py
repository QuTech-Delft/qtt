class MissingOptionalPackageWarning(UserWarning, ValueError):
    """ An optional package is missing """
    pass

# %%


class PackageVersionWarning(UserWarning):
    """ An package has the incorrect version """
    pass


class CalibrationException(BaseException):
    """ Exception thrown for a bad calibration """
    pass