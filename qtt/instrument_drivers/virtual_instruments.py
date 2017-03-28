# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages

from functools import partial
from qcodes.utils.validators import Numbers

import qcodes as qc
from qcodes import Instrument   # , Parameter, Loop, DataArray


#%%


class VirtualMeter(Instrument):

    shared_kwargs = ['model']

    def __init__(self, name, model=None, **kwargs):
        """ Virtual instrument with a parameter
        
        Args:
            name (str)
            model (object): class that provides a `.get` function
            
        """
        super().__init__(name, **kwargs)
        self.model = model

        g = 'amplitude'
        self.add_parameter(g,
                           label='%s amplitude' % name,
                           unit='a.u.',
                           get_cmd=partial(self.get_gate, g),
                           )
        self.add_parameter('readnext', get_cmd=partial(self.get, 'amplitude'), label=name)

    def get_gate(self, gate):
        return self.model.get(self.name + '_' + gate)

    def set_gate(self, gate, value):
        self.model.set(self.name + '_' + gate, value)
        return

    def get_idn(self):
        """ Overrule because the default get_idn yields a warning """
        IDN = {'vendor': 'QuTech', 'model': self.name,
               'serial': None, 'firmware': None}
        return IDN
    
class VirtualSource(Instrument):
    """A virtual instrument with a settable paramater
    """
    shared_kwargs = ['gates']

    def __init__(self, name, source, parameterParent, parameterChild=None, gain=1.0, unit=None, **kwargs):
        """Initialize a virtual gates object.

        Args:
            name (string): The name of the object
            source (Instrument): the physical source instrument
            parameterParent (string): the name of the parameter on the parent instrument
            parameterChild (string, optional): the name of the parameter on the child instrument. If absent, equal to
                                                the parent one
            gain (float, optional): the gain of the virtual parameter. Default equal to one.
            unit (str, optional): the unit of the virtual parameter. Default equal to the parent parameter.
        """
        super().__init__(name, **kwargs)
        self.name = name
        self.gain = gain

        if not parameterChild:
            parameterChild = parameterParent
        print(parameterChild)

        if not unit:
            unit = source[parameterParent].unit

        self.add_parameter(parameterChild,
                           label='%s' % parameterChild,
                           unit=unit,
                           get_cmd=lambda: source.get(parameterParent)*self.gain,
                           set_cmd=lambda val: source.set(parameterParent, val/self.gain),
                           vals=Numbers())  # TODO: Adjust the validator range based on that of the gates

