# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

# %% Load packages
import numpy as np
import logging
from functools import partial

from qcodes import Instrument
from qcodes.utils.validators import Numbers

import qtt.utilities.tools

logger = logging.getLogger(__name__)

# %%


class VirtualMeter(Instrument):

    def __init__(self, name, model=None, **kwargs):
        """ Virtual instrument with a parameter

        Args:
            name (str)
            model (object or None): Class that provides a `.get` function. If the model is None, then the amplitude
                                    parameter returns a random value

        """
        super().__init__(name, **kwargs)
        self.model = model

        g = 'amplitude'
        self.add_parameter(g,
                           label='%s amplitude' % name,
                           unit='a.u.',
                           get_cmd=partial(self._get_gate, g),
                           )
        self.add_parameter('readnext', get_cmd=partial(self.get, 'amplitude'), label=name)

    def _get_gate(self, gate):
        if self.model is None:
            return np.random.rand()
        return self.model.get(self.name + '_' + gate)

    def get_idn(self):
        """ Overrule because the default get_idn yields a warning """
        IDN = {'vendor': 'QuTech', 'model': self.name,
               'serial': None, 'firmware': None}
        return IDN

# %%


class VirtualIVVI(Instrument):

    def __init__(self, name, model, gates=['dac%d' % i for i in range(1, 17)], dac_unit='a.u.', **kwargs): # type: ignore
        """ Virtual instrument representing a DAC

        Args:
            name (str)
            model (object): the model should implement functions get and set 
                  which can get and set variables of the form INSTR_PARAM
                  Here INSTR is the name of the VirtualIVVI, PARAM is the name
                  of the gate
            gates (list of gate names)
            dac_unit (str): unit to set for the dac parameters
        """
        super().__init__(name, **kwargs)

        self.model = model
        self._gates = gates
        logger.debug('add gates')
        for i, g in enumerate(gates):
            logger.debug('VirtualIVVI: add gate %s' % g)
            if model is None:
                self.add_parameter(g,
                                   set_cmd=None,
                                   initial_value=0,
                                   label='Gate {} (arb. units)'.format(g),
                                   unit=dac_unit,
                                   vals=Numbers(-800, 400))
            else:
                self.add_parameter(g,
                                   label='Gate {} (mV)'.format(g),
                                   get_cmd=partial(self._get_gate, g),
                                   set_cmd=partial(self._set_gate, g),
                                   unit=dac_unit,
                                   vals=Numbers(-800, 400))

        self.add_function('reset', call_cmd='rst')

        logger.debug('add legacy style gates to VirtualIVVI')
        for i, g in enumerate(gates):
            self.add_function(
                'get_{}'.format(g), call_cmd=partial(self.get, g))
            logger.debug('add gates function %s: %s' % (self.name, g))

        self._get_all()

    def get_idn(self):
        """
        Overwrites the get_idn function using constants as the virtual device
        does not have a proper `*IDN` function.
        """
        return {'firmware': None, 'model': None, 'serial': None, 'vendor': 'QuTech'}

    def _get_gate(self, gate):
        if self.model is None:
            return 0
        value = self.model.get(self.name + '_' + gate)
        return value

    def _set_gate(self, gate, value):
        if self.model is None:
            return
        value = float(value)
        self.model.set(self.name + '_' + gate, value)
        return

    def allvalues(self):
        """ Return all DAC values 

        Returns:
            dict: dictionary with all DAC values
        """
        return dict([(g, self.get(g)) for g in self.parameters])

    def _get_all(self):
        """ Get all parameters in instrument """
        for g in self._gates:
            logger.debug('_get_all %s: %s' % (self.name, g))
            self.get(g)

    def __repr__(self):
        """ Return string description instance """
        return 'VirtualIVVI: %s' % self.name
