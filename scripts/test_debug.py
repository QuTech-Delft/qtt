# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
import numpy as np

import qcodes as qc
from qcodes.utils.validators import Numbers

import logging
l = logging.getLogger()
l.setLevel(logging.DEBUG)

print('# clean up old processes')
[ x.terminate() for x in qc.active_children() if x.name in ['dummymodel2', 'ivvi1', 'ivvi2'] ]


#%% Virtual model
from qcodes import MockModel, MockInstrument

class DummyModel(MockModel):

    ''' Dummy model for testing '''
    def __init__(self, name, **kwargs):
        self._data = dict()
        self._data['ivvi1'] = dict()
        super().__init__(name=name)

    def compute(self):
        ''' Compute output of the model '''

        logging.debug('compute')
        self._data['ivvi1']['c1']=np.random.rand()
        return 
         
    def ivvi1_get(self, param):
        logging.debug('ivvi1_get: %s' % param)
        if not param in self._data['ivvi1']:
            self._data['ivvi1'][param]=0

        return self._data['ivvi1'][param]
        
    def ivvi1_set(self, param, value):
        logging.debug('ivvi1_set: %s value %s' % ( param, value) )
        self._data['ivvi1'][param]=value


#%%
from functools import partial

class VirtualIVVI(MockInstrument):

    ''' Virtual instrument representing an IVVI '''

    def __init__(self, name, model, gates=['c%d' % i for i in range(1, 17)], **kwargs):
        super().__init__(name, model=model, **kwargs)

        self._gates = gates
        logging.debug('add gates')
        for i, g in enumerate(gates):
            cmdbase = g  # 'c{}'.format(i)
            logging.debug('add gate %s' % g )
            self.add_parameter(g,
                               label='Gate {} (mV)'.format(g),
                               get_cmd=cmdbase + '?',
                               set_cmd=cmdbase + ':{:.4f}',
                               get_parser=float,
                               vals=Numbers(-2000, 2000))

        self.add_function('reset', call_cmd='rst')

        logging.debug('add gates function')
        for i, g in enumerate(gates):
            self.add_function(
                'get_{}'.format(g), call_cmd=partial(self.get, g))
            logging.debug('add gates function %s: %s' % (self.name, g) )


    def get_all(self):
        ''' Get all parameters in instrument '''
        for g in self._gates:
            logging.debug('get_all %s: %s' % (self.name, g) )
            self.get(g)

    def __repr__(self):
        ''' Return string description instance '''
        return 'VirtualIVVI: %s' % self.name


#%% Create a virtual model for testing

print('# import qtt_toymodel')

print('# make dummymodel')
model = DummyModel(name='dummymodel', server_name=None)

print('# make ivvi1')
ivvi1 = VirtualIVVI(name='ivvi1', model=model, server_name=None)


#%%
print('# set ivvi1: fine under unix')
ivvi1.c1.set(200)


print('get c1: %f '  % (ivvi1.c1.get(),) )


