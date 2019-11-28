from typing import Union
import enum

from qcodes import Parameter


class Role(enum.Enum):
    GAIN = 1
    DIVISION = 2


class ParameterScaler(Parameter):
    """ Deprecated class, use qcodes.ScaledParameter instead
    """

    def __init__(self,
                 output: Parameter,
                 division: Union[int, float, Parameter] = None,
                 gain: Union[int, float, Parameter] = None,
                 name: str = None,
                 label: str = None,
                 unit: str = None) -> None:
        raise Exception('Use qcodes.ScaledParameter instead')
