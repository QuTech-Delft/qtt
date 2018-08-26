class MissingOptionalPackageWarning(UserWarning, ValueError):
    """ An optional package is missing """
    pass

#%%

class PackageVersionWarning(UserWarning):
    """ An package has the incorrect version """
    pass