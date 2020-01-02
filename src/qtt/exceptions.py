class MissingOptionalPackageWarning(UserWarning, ValueError):
    """ An optional package is missing """
    pass


class PackageVersionWarning(UserWarning):
    """ A package has the incorrect version """
    pass


class CalibrationException(Exception):
    """ Exception thrown for a bad calibration """
    pass
