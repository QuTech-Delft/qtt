# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:37:36 2016

@author: diepencjv
"""

#%%
from qcodes import Instrument
from functools import partial
from qcodes.utils.validators import Numbers
import numpy as np


class eff_gates(Instrument):
    '''
    Create effective gates, which are linear combinations of the physical gates.
    They are defined, such that when changing one of the effective gates, the
    others are (almost) not influenced. The effective gates are developed in 
    a way such that they correspond to the chemical potentials in mV's.
    
    They do not (yet?) have an offset relative to the physical gates.
    
    Input:
        name (string):
        gates (object):
        eff_gate_map ():
    '''
    shared_kwargs = ['gates']
    
    def __init__(self, name, gates, eff_gate_map, **kwargs):
        super().__init__(name, **kwargs)

        self.gates = gates
        self._map = eff_gate_map
        self._gates_list = sorted(list(self._map[list(self._map.keys())[0]].keys()))
        self._eff_gates_list = sorted(list(self._map.keys()))
        self._matrix = np.array([[self._map[x][y] for y in self._gates_list] for x in self._eff_gates_list])
        self._matrix_inv = np.linalg.inv(self._matrix)
        
        for g in self._eff_gates_list:
            self.add_parameter(g,
                               label='%s (mV)' % g,
                               units='mV',
                               get_cmd=partial(self._get, gate=g),
                               set_cmd=partial(self._set, gate=g),
                               vals=Numbers(-2000, 2000))

    def _get(self, gate):
        gateval = sum([self._map[gate][g]*self.gates[g].get() for g in self._map[gate]])
        return gateval
        
    def _set(self, value, gate):
        gate_vec = np.zeros(len(self._eff_gates_list))
        increment = value - self.get(gate)
        gate_vec[self._eff_gates_list.index(gate)] = increment
        set_matrix = np.dot(self._matrix_inv, gate_vec)
        for idx, g in enumerate(self._gates_list):
            self.gates.set(g, self.gates.get(g) + set_matrix[idx])
        return