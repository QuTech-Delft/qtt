# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages

import numpy as np
from functools import partial

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
    
