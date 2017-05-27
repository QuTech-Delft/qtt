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
from collections import OrderedDict


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

    Functions:
        
    """
    shared_kwargs = ['gates']

    def __init__(self, name, gates_instr, crosscap_map, **kwargs):
        """Initialize a virtual gates object.

        Args:
            name (string): The name of the object (used for?)
            gates_instr (Instrument): The instrument of physical gates
            crosscap_map (OrderedDict/dict): Full map of cross capacitance matrix defined
                    as a dictionary labeled between dot parameters and gates.
                    Name of dot parameters are initially defined in this dict.
                    Use OrderedDict form when the order is important.
                    
                    Example:
                        crosscap_map = OrderedDict((
                    ('VP1', OrderedDict((('P1', 1), ('P2', 0.6), ('P3', 0)))),
                    ('VP2', OrderedDict((('P1', 0.5), ('P2', 1), ('P3', 0)))),
                    ('VP3', OrderedDict((('P1', 0), ('P2', 0), ('P3', 1))))
                    ))

                    Note: this matrix describes the influence of each physical
                    gate on the dotparameters, hence to get and set the dot
                    parameter using a combination of physical gates we need
                    the inverse transormation.

        """
        super().__init__(name, **kwargs)
        self.name = name
        self.gates = gates_instr
        if isinstance(crosscap_map, OrderedDict):
            self._crosscap_map = crosscap_map
            for vg in list(crosscap_map.keys()):
                for g in list(crosscap_map[list(crosscap_map.keys())[0]].keys()):
                    try:
                        self._crosscap_map[vg][g]
                    except:
                        raise NameError('missing physical gate "%s" in virtual gate "%s"' %(g, vg))
        elif isinstance(crosscap_map, dict):
            self._crosscap_map = OrderedDict()
            for vg in sorted(list(crosscap_map.keys())):
                self._crosscap_map[vg] = OrderedDict()
                for g in sorted(list(crosscap_map[list(crosscap_map.keys())[0]].keys())):
                    self._crosscap_map[vg][g] = crosscap_map[vg][g]
        else:
            raise ValueError('cross-capacitance map must be in an OrdereDict or dict form')

        self._gates_list = list(self._crosscap_map[list(self._crosscap_map.keys())[0]].keys())
        self._virts_list = list(self._crosscap_map.keys())
#        self._crosscap_matrix = np.array([[self._crosscap_map[x].get(y, 0) for y in self._gates_list] for x in self._virts_list])
#        self._crosscap_matrix_inv = np.linalg.inv(self._crosscap_matrix)

        self._crosscap_map_inv = OrderedDict()
        for idvirt, virtg in enumerate(self._virts_list):
            self._crosscap_map_inv[virtg] = OrderedDict()
            for idg, g in enumerate(self._gates_list):
                self._crosscap_map_inv[virtg][g] = np.linalg.inv(self.get_crosscap_matrix())[idg][idvirt]

        for g in self._virts_list:
            self.add_parameter(g,
                               label='%s' % g,
                               unit='mV',
                               get_cmd=partial(self._get, gate=g),
                               set_cmd=partial(self._set, gate=g),
                               vals=Numbers())  # TODO: Adjust the validator range based on that of the gates

        # comb_map: crosscap_map defined using gate Parameters
        for vg in self._crosscap_map_inv:
            self.parameters[vg].comb_map = []
            for g in self._crosscap_map_inv[vg]:
                self.parameters[vg].comb_map.append((self.gates.parameters[g], self._crosscap_map_inv[vg][g]))

    def _get(self, gate):
        """Get the value of virtual gate voltage in mV
        
        Args:
            gate (string): Name of virtual gate.

        """
        gateval = sum([self._crosscap_map[gate][g] * self.gates[g].get() for g in self._crosscap_map[gate]])
        return gateval

    def _set(self, value, gate):
        """Set the value of virtual gate voltage in mV
        
        Args:
            value (float): Value to set.
            gate (string): Name of virtual gate.

        """
        gate_vec = np.zeros(len(self._virts_list))
        increment = value - self.get(gate)
        gate_vec[self._virts_list.index(gate)] = increment
        set_vec = np.dot(self._crosscap_matrix_inv, gate_vec)

        for idx, g in enumerate(self._gates_list):
            self.gates.parameters[g].validate(self.gates.get(g) + set_vec[idx])

        for idx, g in enumerate(self._gates_list):
            self.gates.set(g, self.gates.get(g) + set_vec[idx])

    def allvalues(self):
        """Return all virtual gate voltage values in a dict."""
        vals = [(gate, self.get(gate)) for gate in self._virts_list]
        return dict(vals)

    def resetgates(self, activegates, basevalues=None, verbose=0):
        """Reset a set of gates to new values.

        If no new values are specified the virtual gates will be reset to zero.

        Args:
            activegates (list or dict): Virtual gates to reset
            basevalues (dict): New values for the gates
            verbose (int): Output level

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

    def get_crosscap_map(self):
        """Gets the current cross-capacitance map."""
        return self._crosscap_map

    def get_crosscap_map_inv(self):
        """Gets the current inverse of cross-capacitance map."""
        return self._crosscap_map_inv

    def set_crosscap_map(self, replace_map, verbose=0):
        """Sets the cross-capacitance map by replacing the specified map. Then
        updates the connected parameters in memory.
        
        Args:
            replace_map (dict): Map containing replacing values. Uses an
                    arbitrary part of the dict inside the full map.
                    
                    Example: {'VP1': {'P2': 0.4}, 'VP2': {'P1': 0.4, 'P3': 0.1}}

        """
        self._crosscap_map = self._update_map(replace_map, self._crosscap_map, verbose)
        self._update_rest(self._crosscap_map, verbose)
        return self._crosscap_map

    def set_crosscap_map_inv(self, replace_map, verbose=0):
        """Sets the inverse of the cross-capacitance map by replacing the specified map.
        Then updates the connected parameters in memory.
        
        Args:
            replace_map (dict): Map containing replacing values. Uses an
                    arbitrary part of the dict inside the full map.
                    
                    Example: {'VP1': {'P2': -0.4}, 'VP2': {'P1': -0.4, 'P3': -0.1}}

        """
        self._crosscap_map_inv = self._update_map(replace_map, self._crosscap_map_inv, verbose)
        self._update_rest(self._crosscap_map_inv, verbose)
        return self._crosscap_map_inv

    def _update_map(self, replace_map, base_map, verbose=0):
        """Update map values by replacing the values for specified gate cobinations.

        Args:
            replace_map (dict): Map containing values for replacement. Uses an
                    arbitrary part of the dict inside the full map.
            base_map (dict): Base full map that will be replaced. Either a crosscap_map
                    or a crosscap_map_inv.

        """
        updated_map = base_map
        for vg in replace_map:
            for g in replace_map[vg]:
                preivious_val = base_map[vg][g]
                updated_map[vg][g] = replace_map[vg][g]
                if verbose >= 2:
                    print('  setting %s-%s, %.3f to %.3f' % (vg, g, preivious_val, updated_map[vg][g]))
        if verbose >= 2:
            self.print_map(updated_map)
        return updated_map

    def print_map(self, base_map):
        """Show map as table.
        
        Args:
            base_map (dict): Map of what to show. Either a crosscap_map
                    or a crosscap_map_inv.

        """
        print('',*self._gates_list,sep='\t')
        for vg in self._virts_list:
            print('\t'.join([vg] + [('%.3f' % value).rstrip('0').rstrip('.') for g, value in base_map[vg].items() ] ) )

    def _update_rest(self, base_map, verbose=0):
        """Updates rest of the virtual gate variables
        
        Args:
            base_map (dict): Base full map that was replaced. Either a crosscap_map
                    or a crosscap_map_inv.

        """
        if base_map == self._crosscap_map:
#            self._crosscap_matrix = self._update_crosscap_matrix(self._crosscap_map, verbose)
#            self._crosscap_matrix_inv = np.linalg.inv(self._crosscap_matrix)
            crosscap_map_inv = OrderedDict()
            for idvirt, virtg in enumerate(self._virts_list):
                crosscap_map_inv[virtg] = OrderedDict()
                for idg, g in enumerate(self._gates_list):
                    crosscap_map_inv[virtg][g] = np.linalg.inv(self.get_crosscap_matrix())[idg][idvirt]
            if verbose:
                print('  updating crosscap_map_inv')
                if verbose >= 2:
                    self.print_map(crosscap_map_inv)
            self._crosscap_map_inv = crosscap_map_inv
            self._update_virt_parameters(self._crosscap_map_inv, verbose)
            self.allvalues()
        elif base_map == self._crosscap_map_inv:
            crosscap_map = OrderedDict()
            for idvirt, virtg in enumerate(self._virts_list):
                crosscap_map[virtg] = OrderedDict()
                for idg, g in enumerate(self._gates_list):
                    crosscap_map[virtg][g] = np.linalg.inv(self.get_crosscap_matrix_inv())[idg][idvirt]
            if verbose:
                print('  updating crosscap_map')
                if verbose >= 2:
                    self.print_map(crosscap_map)
            self._crosscap_map = crosscap_map
            self._update_virt_parameters(self._crosscap_map_inv, verbose)
            self.allvalues()

    def get_crosscap_matrix(self):
        """Gets the current cross-capacitance matrix."""
        return np.array([[self._crosscap_map[x].get(y, 0) for y in self._gates_list] for x in self._virts_list])

    def get_crosscap_matrix_inv(self):
        """Gets the current inverse of cross-capacitance matrix."""
        return np.array([[self._crosscap_map_inv[x].get(y, 0) for y in self._gates_list] for x in self._virts_list])

    def _update_virt_parameters(self, crosscap_map_inv, verbose=0):
        """Redefining the cross capacitance values in the virts Parameter.
        
        Needs the crosscap_map_inv information as an input.

        """
        for vg in crosscap_map_inv:
            self.parameters[vg].comb_map = []
            for g in crosscap_map_inv[vg]:
                self.parameters[vg].comb_map.append((self.gates.parameters[g], crosscap_map_inv[vg][g]))
        if verbose >= 2:
            print('  updating virt parameters')
