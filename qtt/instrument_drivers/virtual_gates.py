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
    """A virtual gate instrument to control linear combinations of gates.

    The virtual gates can be defined, such that when changing one of the
    virtual gates, the others are not influenced. The virtual gates
    can be used for changing only one physical parameter, e.g. a chemical 
    potential or a tunnel coupling.
    
    Note: They do not (yet?) have an offset relative to the physical parameters.
    The sweepmap describes a submatrix of the inverse of the virt_gate_map.

    Attributes:
        name (string): The name of the virtual gate object
        gates_instr (Instrument): The instrument of physical gates
        crosscap_map (dict): The dictionary of a cross capacitance matrix
                    between dot parameters and gates. Here defines the name of
                    dot parameters.
                    Note: this matrix describes the influence of each physical
                    gate on the dotparameters, hence to get and set the dot
                    parameter using a combination of physical gates we need
                    the inverse transormation.
        crosscap_map_inv (dict): The dictionary of an inversed cross capacitance matrix.
    """
    shared_kwargs = ['gates']

    def __init__(self, name, gates_instr, crosscap_map, **kwargs):
        """Initialize a virtual gates object.

        Args:
            name (string): The name of the object (used for?)
            gates_instr (Instrument): The instrument of physical gates
            crosscap_map (dict): Full map of cross capacitance matrix defined
                    as a dictionary labeled between dot parameters and gates.
                    Name of dot parameters are initially defined in this dict.
        """
        super().__init__(name, **kwargs)
        self.name = name
        self.gates = gates_instr
        self.crosscap_map = crosscap_map
        self._gates_list = sorted(list(self.crosscap_map[list(self.crosscap_map.keys())[0]].keys()))
        self._virts_list = sorted(list(self.crosscap_map.keys()))
        self._crosscap_matrix = np.array([[self.crosscap_map[x].get(y, 0) for y in self._gates_list] for x in self._virts_list])
        self._crosscap_matrix_inv = np.linalg.inv(self._crosscap_matrix)
        self.crosscap_map_inv = dict()
        for idvirt, virtg in enumerate(self._virts_list):
            self.crosscap_map_inv[virtg] = dict()
            for idg, g in enumerate(self._gates_list):
                self.crosscap_map_inv[virtg][g] = self._crosscap_matrix_inv[idg][idvirt]

        for g in self._virts_list:
            self.add_parameter(g,
                               label='%s' % g,
                               unit='mV',
                               get_cmd=partial(self._get, gate=g),
                               set_cmd=partial(self._set, gate=g),
                               vals=Numbers())  # TODO: Adjust the validator range based on that of the gates

        # comb_map: crosscap_map defined using gate Parameters
        for vg in self.crosscap_map_inv:
            self.parameters[vg].comb_map = []
            for g in self.crosscap_map_inv[vg]:
                self.parameters[vg].comb_map.append((self.gates.parameters[g], self.crosscap_map_inv[vg][g]))

    def _get(self, gate):
        gateval = sum([self.crosscap_map[gate][g] * self.gates[g].get() for g in self.crosscap_map[gate]])
        return gateval

    def _set(self, value, gate):
        gate_vec = np.zeros(len(self._virts_list))
        increment = value - self.get(gate)
        gate_vec[self._virts_list.index(gate)] = increment
        set_vec = np.dot(self._crosscap_matrix_inv, gate_vec)

        for idx, g in enumerate(self._gates_list):
            self.gates.parameters[g].validate(self.gates.get(g) + set_vec[idx])

        for idx, g in enumerate(self._gates_list):
            self.gates.set(g, self.gates.get(g) + set_vec[idx])

    def allvalues(self):
        """Return all virtual gates values in a dict."""
        vals = [(gate, self.get(gate)) for gate in self._virts_list]
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

    def update_crosscap(self, crosscap_dict, verbose=2):
        """Update cross capacitance values for the specified gate cobinations.
        
        Args:
            crosscap_dict (dict): Map of new cross capacitance values. Uses an
                    arbitrary part inside the full map.
                    
                    Example: {'VP1': {'P2': 0.4}, 'VP2': {'P1': 0.4, 'P3': 0.1}}
        """
        if verbose:
            print('update_crosscap: updating cross-capacitance map')
        for vg in crosscap_dict:
            for g in crosscap_dict[vg]:
                val = crosscap_dict[vg][g]
                if verbose >= 2:
                    print('  setting cross capacitance between %s and %s from %.3f to %.3f' % (vg, g, self.crosscap_map[vg][g], val))
                self.crosscap_map[vg][g] = val
        if verbose >= 2:
            print('',*self._gates_list,sep='\t')
            for vg in self._virts_list:
                print(vg,*(self.crosscap_map[vg][g] for g in self.crosscap_map[vg]),sep='\t')
        self._update_rest(self.crosscap_map)

    def _update_rest(self, updated_var):
        """Updates rest of the virtual gate variables"""
        if updated_var == self.crosscap_map:
            self._crosscap_matrix = self._update_crosscap_matrix(self.crosscap_map)
            self._crosscap_matrix_inv = np.linalg.inv(self._crosscap_matrix)
            self.crosscap_map_inv = self._update_crosscap_map_inv(self._crosscap_matrix_inv)
            self._update_virt_parameters(self.crosscap_map_inv)
            self.allvalues()

    def _update_crosscap_matrix(self, crosscap_map, verbose=1):
        """Internal update of cc_matrix from cc_map"""
        crosscap_matrix = np.array([[crosscap_map[x].get(y, 0) for y in self._gates_list] for x in self._virts_list])
        if verbose >= 2:
            print('  updating crosscap_matrix')
        return crosscap_matrix

    def _update_crosscap_map_inv(self, crosscap_matrix_inv, verbose=2):
        """Internal update of cc_map_inv"""
        crosscap_map_inv = dict()
        for idvirt, virtg in enumerate(self._virts_list):
            crosscap_map_inv[virtg] = dict()
            for idg, g in enumerate(self._gates_list):
                crosscap_map_inv[virtg][g] = crosscap_matrix_inv[idg][idvirt]
        if verbose:
            print('  updating crosscap_map_inv')
            if verbose >= 2:
                print('',*self._gates_list,sep='\t')
                for vg in self._virts_list:
                    print(vg,*(crosscap_map_inv[vg][g] for g in crosscap_map_inv[vg]),sep='\t')
        return crosscap_map_inv

    def _update_virt_parameters(self, crosscap_map_inv, verbose=1):
        """Redefining the cross capacitance values in the virts Parameter"""
        for vg in crosscap_map_inv:
            self.parameters[vg].comb_map = []
            for g in crosscap_map_inv[vg]:
                self.parameters[vg].comb_map.append((self.gates.parameters[g], crosscap_map_inv[vg][g]))
        if verbose >= 2:
            print('  updating virt parameters')



