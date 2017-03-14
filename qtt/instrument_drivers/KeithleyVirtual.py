# KeithleyVirtual.py driver for Keithley testing
#
# Pieter Eendebak <pieter.eendebak@gmail.com>

import time
import logging
from functools import partial
import numpy as np

import qtt.tools

from qcodes import Instrument
from qcodes.utils.validators import Numbers as NumbersValidator


@qtt.tools.deprecated
class KeithleyVirtual(Instrument):
    '''
    This is the qcodes driver for the Keithley_2700 Multimeter

    Usage: Initialize with
    <name> =  = KeithleyVirtual(<name>, address='<GPIB address>', ... )

    '''
    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('dummy',
                           get_cmd=self.get_dummy,
                           set_cmd=self.set_dummy,
                           vals=NumbersValidator())

        self.add_function('readnext', call_cmd=self.readnext_function, unit='arb.unit')

    def readnext_function(self, **kwargs):
        val = np.random.rand() - .5
        print('readnext_function: val %f: %s' % (val, str(kwargs)))
        return val

    def get_dummy(self, **kwargs):
        val = np.random.rand() - .5
        print('get_dummy: %s' % (str(kwargs)))
        return val
    def set_dummy(self, value, **kwargs):
        print('set_dummy: %s: %s' % (value, str(kwargs)))
