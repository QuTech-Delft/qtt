# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
import logging
from functools import partial

import qcodes as qc
from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Numbers

logger = logging.getLogger(__name__)

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

#%%


class VirtualIVVI(Instrument):

    shared_kwargs = ['model']

    def __init__(self, name, model, gates=['dac%d' % i for i in range(1, 17)], mydebug=False, **kwargs):
        """ Virtual instrument representing an IVVI 

        Args:
            name (str)
            model (object): the model should implement functions get and set 
                  which can get and set variables of the form INSTR_PARAM
                  Here INSTR is the name of the VirtualIVVI, PARAM is the name
                  of the gate
            gates (list of gate names)
        """
        super().__init__(name, **kwargs)

        self.model = model
        self._gates = gates
        logger.debug('add gates')
        for i, g in enumerate(gates):
            logger.debug('VirtualIVVI: add gate %s' % g)
            if model is None:
                self.add_parameter(g,
                                   parameter_class=ManualParameter,
                                   initial_value=0,
                                   label='Gate {} (arb. units)'.format(g),
                                   unit='arb. units',
                                   vals=Numbers(-800, 400))
            else:
                self.add_parameter(g,
                                   label='Gate {} (mV)'.format(g),
                                   get_cmd=partial(self.get_gate, g),
                                   set_cmd=partial(self.set_gate, g),
                                   unit='arb.units',
                                   vals=Numbers(-800, 400))

        self.add_function('reset', call_cmd='rst')

        logger.debug('add legacy style gates to VirtualIVVI')
        for i, g in enumerate(gates):
            self.add_function(
                'get_{}'.format(g), call_cmd=partial(self.get, g))
            logger.debug('add gates function %s: %s' % (self.name, g))

        if not mydebug:
            self.get_all()

    def get_idn(self):
        """
        Overwrites the get_idn function using constants as the virtual device
        does not have a proper `*IDN` function.
        """
        # not all IVVI racks support the version command, so return a dummy
        return {'firmware': None, 'model': None, 'serial': None, 'vendor': None}

    def get_gate(self, gate):
        if self.model is None:
            return 0
        value = self.model.get(self.name + '_' + gate)
        logger.debug('%s: get_gate %s' % (self.name, gate))
        return value

    def set_gate(self, gate, value):
        if self.model is None:
            return
        value = float(value)
        self.model.set(self.name + '_' + gate, value)
        logger.debug('set_gate %s: %s' % (gate, value))
        return

    def allvalues(self):
        return dict([(g, self.get(g) ) for g in self.parameters])
    
    def get_all(self):
        ''' Get all parameters in instrument '''
        for g in self._gates:
            logger.debug('get_all %s: %s' % (self.name, g))
            self.get(g)

    def __repr__(self):
        ''' Return string description instance '''
        return 'VirtualIVVI: %s' % self.name
