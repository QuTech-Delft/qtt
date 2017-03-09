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


class virtual_gates(Instrument):
    """A virtual instrument, which has linear combinations of gates.

    The virtual gates are defined, such that when changing one of the
    virtual gates, the others are not influenced. The virtual gates
    can be used for changing only one physical parameter, e.g. a chemical 
    potential or a tunnel coupling. Note: They do not (yet?) have an offset 
    relative to the physical parameters.
    The sweepmap describes a submatrix of the inverse of the virt_gate_map.

    Attributes:
        name (string): The name of the object
        gates (Instrument): the physical gates
        virt_map (dict): The transformation matrix. Note: this matrix
                      describes the influence of every gate on the physical
                      parameter, hence to get and set the physical parameter we
                      actually need the inverse transormation.
        map_inv (dict): 
    """
    shared_kwargs = ['gates']

    def __init__(self, name, gates, virt_map, **kwargs):
        """Initialize a virtual gates object.

        Args:
            name (string): The name of the object
            gates (Instrument): the physical gates
            virt_map (dict): The transformation matrix. Note: this matrix
                      describes the influence of every gate on the physical
                      parameter, hence to get and set the physical parameter we
                      actually need the inverse transormation.
        """
        super().__init__(name, **kwargs)
        self.name = name
        self.gates = gates
        self.virt_map = virt_map
        self._gates_list = sorted(list(self.virt_map[list(self.virt_map.keys())[0]].keys()))
        self._virt_gates_list = sorted(list(self.virt_map.keys()))
        self._matrix = np.array([[self.virt_map[x].get(y, 0) for y in self._gates_list] for x in self._virt_gates_list])
        self._matrix_inv = np.linalg.inv(self._matrix)
        self.map_inv = dict()
        for idvirt, virtg in enumerate(self._virt_gates_list):
            self.map_inv[virtg] = dict()
            for idg, g in enumerate(self._gates_list):
                self.map_inv[virtg][g] = self._matrix_inv[idg][idvirt]

        for g in self._virt_gates_list:
            self.add_parameter(g,
                               label='%s (mV)' % g,
                               unit='mV',
                               get_cmd=partial(self._get, gate=g),
                               set_cmd=partial(self._set, gate=g),
                               vals=Numbers())  # TODO: Adjust the validator range based on that of the gates

        for vg in self.map_inv:
            self.parameters[vg].comb_map = []
            for g in self.map_inv[vg]:
                self.parameters[vg].comb_map.append((self.gates.parameters[g], self.map_inv[vg][g]))

    def _get(self, gate):
        gateval = sum([self.virt_map[gate][g] * self.gates[g].get() for g in self.virt_map[gate]])
        return gateval

    def _set(self, value, gate):
        gate_vec = np.zeros(len(self._virt_gates_list))
        increment = value - self.get(gate)
        gate_vec[self._virt_gates_list.index(gate)] = increment
        set_vec = np.dot(self._matrix_inv, gate_vec)

        for idx, g in enumerate(self._gates_list):
            self.gates.parameters[g].validate(self.gates.get(g) + set_vec[idx])

        for idx, g in enumerate(self._gates_list):
            self.gates.set(g, self.gates.get(g) + set_vec[idx])

    def allvalues(self):
        """Return all virtual gates values in a dict."""
        vals = [(gate, self.get(gate)) for gate in self._virt_gates_list]
        return dict(vals)

    def resetgates(self, activegates, basevalues=None, verbose=2):
        """Reset a set of gates to new values.

        If no new values are specified the virtual gates will be reset to zero.

        Args:
            activegates (list or dict): gates to reset
            basevalues (dict): new values for the gates
            verbose (int): output level
        """
        if verbose:
            print('resetgates: setting gates to default values')
        for g in activegates:
            if basevalues == None:
                val = 0
            else:
                if g in basevalues:
                    val = basevalues[g]
                else:
                    val = 0
            if verbose >= 2:
                print('  setting gate %s to %.1f [mV]' % (g, val))
            self.set(g, val)
