# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:37:36 2016

@author: diepencjv, eendebakpt
"""

# %% Load packages
from qcodes import Instrument
from functools import partial
from qcodes.utils.validators import Numbers
import numpy as np
ordered_dict = dict

import warnings
import matplotlib.pyplot as plt

import qtt.measurements.scans


def set_distance_matrix(virt_gates, dists):
    """ Update the cross capacitance matrix for a virtual_gate matrix

    Args:
        virt_gates (VirtualGates): virtual gates object
        dists (list): list of distances between dots
    """
    cc = virt_gates.get_crosscap_matrix()
    dists = list(dists) + [0] * cc.shape[0]
    for ii in range(cc.shape[0]):
        for jj in range(cc.shape[0]):
            cc[ii, jj] = dists[np.abs(ii - jj)]
    virt_gates.set_crosscap_matrix(cc)


def create_virtual_matrix_dict(virt_basis, physical_gates, c=None, verbose=1):
    """ Converts the virtual gate matrix into a virtual gate mapping

    Args:
        virt_basis (list): containing all the virtual gates in the setup
        physical_gates (list): containing all the physical gates in the setup
        c (array or None): virtual gate matrix
    Returns:
        virtual_matrix (dict): dictionary, mapping of the virtual gates
    """
    virtual_matrix = ordered_dict()
    for ii, vname in enumerate(virt_basis):
        if verbose:
            print('create_virtual_matrix_dict: adding %s ' % (vname,))
        if c is None:
            v = np.zeros(len(physical_gates))
            v[ii] = 1
        else:
            v = c[ii, :]
        tmp = ordered_dict(zip(physical_gates, v))
        virtual_matrix[vname] = tmp
    return virtual_matrix

class VirtualGates(Instrument):
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

    def __init__(self, name, gates_instr, crosscap_map, **kwargs):
        """Initialize a virtual gates object.

        Args:
            name (string): The name of the object (used for?)
            gates_instr (Instrument): The instrument of physical gates
            crosscap_map (ordered_dict/dict): Full map of cross capacitance matrix defined
                    as a dictionary labeled between dot parameters and gates.
                    Name of dot parameters are initially defined in this dict.
                    Use ordered_dict form when the order is important.

                    Example:
                        crosscap_map = ordered_dict((
                    ('VP1', ordered_dict((('P1', 1), ('P2', 0.6), ('P3', 0)))),
                    ('VP2', ordered_dict((('P1', 0.5), ('P2', 1), ('P3', 0)))),
                    ('VP3', ordered_dict((('P1', 0), ('P2', 0), ('P3', 1))))
                    ))

                    Note: this matrix describes the influence of each physical
                    gate on the dotparameters, hence to get and set the dot
                    parameter using a combination of physical gates we need
                    the inverse transormation.

        """
        super().__init__(name, **kwargs)
        self.gates = gates_instr
        self._fast_readout = False
        if isinstance(crosscap_map, ordered_dict):
            self._crosscap_map = crosscap_map
            for vg in list(crosscap_map.keys()):
                for g in list(crosscap_map[list(crosscap_map.keys())[0]].keys()):
                    try:
                        self._crosscap_map[vg][g]
                    except:
                        raise NameError('missing physical gate "%s" in virtual gate "%s"' % (g, vg))
        elif isinstance(crosscap_map, dict):
            self._crosscap_map = ordered_dict()
            for vg in sorted(list(crosscap_map.keys())):
                self._crosscap_map[vg] = ordered_dict()
                for g in sorted(list(crosscap_map[list(crosscap_map.keys())[0]].keys())):
                    self._crosscap_map[vg][g] = crosscap_map[vg][g]
        else:
            raise ValueError('cross-capacitance map must be in an OrdereDict or dict form')

        self._gates_list = list(self._crosscap_map[list(self._crosscap_map.keys())[0]].keys())
        self._virts_list = list(self._crosscap_map.keys())
        self._crosscap_map_inv = self.convert_matrix_to_map(np.linalg.inv(
            self.get_crosscap_matrix()), gates=self._virts_list, vgates=self._gates_list)
        self._fast_readout = True

        self._create_parameters()
        self.allvalues()

    def to_dictionary(self):
        """ Convert a virtual gates object to a dictionary for storage """
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}
        d = without_keys(self.__dict__, ['parameters', 'log','_crosscap_map', '_crosscap_map_inv'])
        d['gates'] = str(d['gates'])
        d['crosscap_matrix'] = self.get_crosscap_matrix()
        return d

    @staticmethod
    def from_dictionary(vgdict, gates, name=None):
        """ Convert dictionary to virtual gate matrix object """
        if name is None:
            name = qtt.measurements.scans.instrumentName(vgdict['_name'])
        pgates = vgdict['_gates_list']
        vgates = vgdict['_virts_list']
        virt_map = create_virtual_matrix_dict(vgates, pgates, c=vgdict['crosscap_matrix'], verbose=0)

        return VirtualGates(name, gates, virt_map)

    def _create_parameters(self):
        for g in self._virts_list:
            self.add_parameter(g,
                               label='%s' % g,
                               unit='mV',
                               get_cmd=partial(self._get, gate=g),
                               set_cmd=partial(self._set, gate=g),
                               vals=Numbers())

        self.add_parameter('virtual_matrix', get_cmd=self.get_crosscap_matrix)

        self._update_virt_parameters()

    def vgates(self):
        """ Return the names of the virtual gates """
        return self._virts_list

    def pgates(self):
        """ Return the names of the physical gates """
        return self._gates_list

    def _get(self, gate):
        """Get the value of virtual gate voltage in mV

        Args:
            gate (string): Name of virtual gate.

        """
        if self._fast_readout:
            gateval = sum([self._crosscap_map[gate][g] * self.gates[g].get_latest() for g in self._crosscap_map[gate]])
        else:
            gateval = sum([self._crosscap_map[gate][g] * self.gates[g].get() for g in self._crosscap_map[gate]])
        return gateval

    def multi_set(self, increment_map):
        """ Update multiple parameters at once

        Args:
            increment_map (dict): dictionary with keys the gate names and values the increments
        """

        # get current gate values
        gatevalue = [None] * len(self._gates_list)
        for idx, g in enumerate(self._gates_list):
            if self._fast_readout:
                gatevalue[idx] = self.gates.parameters[g].get_latest()
            else:
                gatevalue[idx] = self.gates.parameters[g].get()

        gate_vec = np.zeros(len(self._virts_list))
        for g, increment in increment_map.items():
            gate_vec[self._virts_list.index(g)] += increment
        set_vec = np.dot(self.get_crosscap_matrix_inv(), gate_vec)

        # check the values
        for idx, g in enumerate(self._gates_list):
            self.gates.parameters[g].validate(gatevalue[idx] + set_vec[idx])

        # update the values
        for idx, g in enumerate(self._gates_list):
            self.gates.set(g, gatevalue[idx] + set_vec[idx])

    def _set(self, value, gate):
        """Set the value of virtual gate voltage in mV

        Args:
            value (float): Value to set.
            gate (string): Name of virtual gate.

        """

        increment = value - self.get(gate)
        # self.multi_set({gate: increment})
        gate_vec = np.zeros(len(self._virts_list))
        gate_vec[self._virts_list.index(gate)] = increment
        set_vec = np.dot(self.get_crosscap_matrix_inv(), gate_vec)

        if self._fast_readout:
            gatevalue = [None] * len(self._gates_list)
            for idx, g in enumerate(self._gates_list):
                gatevalue[idx] = self.gates.parameters[g].get_latest()

                self.gates.parameters[g].validate(gatevalue[idx] + set_vec[idx])
        else:
            gatevalue = [None] * len(self._gates_list)
            for idx, g in enumerate(self._gates_list):
                gatevalue[idx] = self.gates.get(g)

                self.gates.parameters[g].validate(gatevalue[idx] + set_vec[idx])

        for idx, g in enumerate(self._gates_list):
            self.gates.set(g, gatevalue[idx] + set_vec[idx])

    def allvalues(self, get_latest=False):
        """Return all virtual gate voltage values in a dict."""
        if get_latest:
            vals = [(gate, self.parameters[gate].get_latest()) for gate in self._virts_list]
        else:
            vals = [(gate, self.get(gate)) for gate in self._virts_list]
        return dict(vals)

    def set_distances(self, dists):
        """ Update the cross-capacitance matrix based on a list of distances """
        set_distance_matrix(self, dists)

    def setgates(self, values, verbose=0):
        """ Set gates to new values.

        Args:
            values (dict): keys are gate names, values are values to be set
            verbose (int): Output level
        """
        if verbose:
            print('resetgates: setting gates to default values')
        for g, val in values.items():
            if verbose >= 2:
                print('  setting gate %s to %.1f [mV]' % (g, val))
            self.set(g, val)

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
            if basevalues is None:
                val = 0
            else:
                if g in basevalues:
                    val = basevalues[g]
                else:
                    val = 0
            if verbose >= 2:
                print('  setting gate %s to %.1f [mV]' % (g, val))
            self.set(g, val)

    def set_crosscap_matrix(self, cc):
        """Sets the cross-capacitance matrix. Update the dependent variables """
        m = self.convert_matrix_to_map(cc)
        self.set_crosscap_map(m)

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
                    arbitrary part of the dict inside the full map. Order of
                    gates does not matter.

                    Example: {'VP2': {'P2': 0.4}, 'VP2': {'P1': 0.4, 'P3': 0.1}}

        """
        self._crosscap_map = self._update_map(replace_map, self._crosscap_map, verbose)
        self._update_rest(self._crosscap_map, verbose)
        return self._crosscap_map

    def set_crosscap_map_inv(self, replace_map, verbose=0):
        """Sets the inverse of the cross-capacitance map by replacing the specified map.
        Then updates the connected parameters in memory.

        Args:
            replace_map (dict): Map containing replacing values. Uses an
                    arbitrary part of the dict inside the full map. Order of
                    gates does not matter.

                    Example: {'P1': {'VP2': -0.4}, 'P2': {'VP1': -0.4, 'VP3': -0.1}}

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
        import copy
        updated_map = copy.deepcopy(base_map)
        for vg in replace_map:
            for g in replace_map[vg]:
                try:
                    previous_val = base_map[vg][g]
                    new_val = replace_map[vg][g]
                    updated_map[vg][g] = new_val
                except:
                    warnings.warn('are you sure you want this?')
                    previous_val = base_map[g][vg]
                    new_val = replace_map[g][vg]
                    updated_map[g][vg] = new_val
                if verbose >= 2:
                    print('  setting %s-%s, %.3f to %.3f' % (vg, g, previous_val, new_val))
        if verbose >= 2:
            self.print_map(updated_map)
        return updated_map

    def ratio(self, target, g1, g2, laplace=0):
        """ Return ratio of influence of two gates

        Args:
            target (str): target gate
            g1, g2 (str): gates
            laplace (float): parameter for Laplacian smoothing

        Returns
            ratio (float)

        """
        m = self.get_crosscap_map()
        ratio = (m[target][g2] + laplace) / (m[target][g1] + laplace)
        return ratio

    def print_matrix(self):
        self.print_map(self.get_crosscap_map())

    def print_inverse_matrix(self):
        self.print_map(self.get_crosscap_map_inv())

    def normalize_matrix(self):
        """ Normalize the rows of the matrix by dividing each row by the diagonal coefficient """
        normalized_matrix = self.get_crosscap_matrix()
        normalized_matrix = normalized_matrix * (1.0 / normalized_matrix.diagonal()).reshape(-1, 1)
        self.set_crosscap_matrix(normalized_matrix)

    @staticmethod
    def print_map(base_map):
        """Show map as table.

        Args:
            base_map (dict): Map of what to show. Either a crosscap_map
                    or a crosscap_map_inv.

        """
        print('', *list(list(base_map.values())[0].keys()), sep='\t')
        for vg in list(base_map.keys()):
            print('\t'.join([vg] + [('%.3f' % value).rstrip('0').rstrip('.') for g, value in base_map[vg].items()]))

    def plot_matrix(self, fig=10, inverse=False):
        """ Plot the cross-capacitance matrix as a figure

        Args:
            fig (int): number of figure window
            inverse (bool): If True then plot the inverse matrix
        """
        if inverse:
            m = self.get_crosscap_matrix_inv()
            xlabels = self.vgates()
            ylabels = self.pgates()
        else:
            m = self.get_crosscap_matrix()
            xlabels = self.pgates()
            ylabels = self.vgates()
        x = range(0, len(xlabels))
        y = range(0, len(ylabels))

        plt.figure(fig)
        plt.clf()
        plt.imshow(m, interpolation='nearest')
        ax = plt.gca()
        plt.tick_params(
            axis='y',
            left=False,)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax.xaxis.set_tick_params(labeltop=True)
        plt.xticks(x, xlabels, rotation='vertical')
        plt.yticks(y, ylabels)  # , rotation='vertical')

    def _update_rest(self, base_map, verbose=0):
        """Updates rest of the virtual gate variables

        Args:
            base_map (dict): Base full map that was replaced. Either a crosscap_map
                    or a crosscap_map_inv.

        """
        if base_map == self._crosscap_map:
            cmatrix = self.get_crosscap_matrix()
            cmatrix_inv = np.linalg.inv(cmatrix)
            crosscap_map_inv = self.convert_matrix_to_map(cmatrix_inv, gates=self._virts_list, vgates=self._gates_list)
            if verbose:
                print('  updating crosscap_map_inv')
                if verbose >= 2:
                    self.print_map(crosscap_map_inv)
            self._crosscap_map_inv = crosscap_map_inv
            self._update_virt_parameters(self._crosscap_map_inv, verbose)
            self.allvalues()
        elif base_map == self._crosscap_map_inv:
            cmatrix = np.linalg.inv(self.get_crosscap_matrix_inv())
            crosscap_map = self.convert_matrix_to_map(cmatrix)
            if verbose:
                print('  updating crosscap_map')
                if verbose >= 2:
                    self.print_map(crosscap_map)
            self._crosscap_map = crosscap_map
            self._update_virt_parameters(self._crosscap_map_inv, verbose)
            self.allvalues()

    def get_crosscap_matrix(self):
        """Gets the current cross-capacitance matrix."""
        return self.convert_map_to_matrix(self._crosscap_map)

    def get_crosscap_matrix_inv(self):
        """Gets the current inverse of cross-capacitance matrix."""
        return self.convert_map_to_matrix(self._crosscap_map_inv, gates=self._virts_list, vgates=self._gates_list)

    def convert_map_to_matrix(self, base_map, gates=None, vgates=None):
        """Convert map of the crosscap form to matrix

        Args:
            base_map (ordered_dict): Crosscap map or its inverse.
             gates (list or None): list of gate names (columns of matrix)
             vgates (list or None): list of virtual gate names (rows of matrix)
        Return:
            converted_matrix (array): Matrix with its elements orderd with given gate order.

        """
        if gates is None:
            gates = self._gates_list
        if vgates is None:
            vgates = self._virts_list
        return np.array([[base_map[x].get(y, 0) for y in gates] for x in vgates])

    def convert_matrix_to_map(self, base_matrix, gates=None, vgates=None):
        """Convert ordered matrix to map.

        Args:
            base_matrix (array): Matrix with its elements ordered with given gate order.
             gates (list or None): list of gate names (columns of matrix)
             vgates (list or None): list of virtual gate names (rows of matrix)

        Return:
            converted_map (ordered_dict): Map after conversion.

        """
        if gates is None:
            gates = self._gates_list
        if vgates is None:
            vgates = self._virts_list
        converted_map = ordered_dict()
        for idvirt, virtg in enumerate(vgates):
            converted_map[virtg] = ordered_dict()
            for idg, g in enumerate(gates):
                converted_map[virtg][g] = base_matrix[idvirt][idg]
        return converted_map

    def _update_virt_parameters(self, crosscap_map_inv=None, verbose=0):
        """ Redefining the cross capacitance values in the virts Parameter.

        Needs the crosscap_map_inv information as an input.

        """
        if crosscap_map_inv is None:
            crosscap_map_inv = self.get_crosscap_map_inv()
        for vg in self._virts_list:
            self.parameters[vg].comb_map = []
            for g in self._gates_list:
                self.parameters[vg].comb_map.append((self.gates.parameters[g], crosscap_map_inv[g][vg]))
        if verbose >= 2:
            print('  updating virt parameters')


def extend_virtual_gates(vgates, pgates, virts, name='vgates', verbose=0):
    """ Create a new virtual gates object based on another virtual gates object """
    vgates0 = virts.vgates()
    pgates0 = virts.pgates()

    map0 = virts.get_crosscap_map()
    cc = np.eye(len(vgates), len(pgates))

    for ii, v in enumerate(vgates0):
        for jj, p in enumerate(pgates0):
            pass
            if(p in pgates) and v in vgates:
                i = vgates.index(v)
                j = pgates.index(p)
                cc[i, j] = map0[v][p]
                if verbose:
                    print('extend_virtual_gates: %s %s = %s' % (v, p, cc[i, j]))
    crosscap_map = create_virtual_matrix_dict(vgates, pgates, cc, verbose=0)
    virts = VirtualGates(qtt.measurements.scans.instrumentName(name), virts.gates, crosscap_map)
    return virts


def update_cc_matrix(virt_gates, update_cc, old_cc=None, verbose=1):
    """ Create a new virtual gates object using an update matrix

    Args:
        virt_gates (VirtualGates): virtual gates object
        update_cc (array): update to cc matrix
        old_cc (array or None): if None, then get the old cc matrix from the virt_gates
        verbose (int): verbosity level
    Returns:
        new_virt_gates (virtual gates):
        new_cc (array):
        results (dict): dictionary with additional results
    """
    physical_gates = virt_gates.pgates()
    vgates = virt_gates.vgates()

    if old_cc is None:
        old_cc = virt_gates.get_crosscap_matrix()
    new_cc = update_cc.dot(old_cc)

    if verbose:
        print('old matrix')
        print(old_cc)
        print('update matrix')
        print(update_cc)
        print('new matrix')
        print(new_cc)

    virt_map = create_virtual_matrix_dict(vgates, physical_gates, new_cc, verbose)
    new_virt_gates = VirtualGates(qtt.measurements.scans.instrumentName('virt_gates'), virt_gates.gates, virt_map)
    if verbose >= 2:
        new_virt_gates.print_map(virt_map)
        print(virt_gates.get_crosscap_matrix_inv())

    return new_virt_gates, new_cc, {'old_cc': old_cc}


virtual_gates=qtt.utilities.tools.deprecated(VirtualGates)
