# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:36:01 2017

@author: diepencjv
"""

# %%
import warnings
import time
import logging
from functools import partial
import numpy as np

from qcodes import Instrument
from qcodes.utils.validators import Numbers
from qcodes.data.data_set import load_data

try:
    import graphviz
except:
    pass

# %%

class VirtualDAC(Instrument):
    """ This class maps the dacs of IVVI('s) to named gates.

    The main functionality is the renaming of numbered dacs of one or multiple
    DAC instruments (for example an IVVI or SPI D5a module) to gates, which in 
    general have names describing their main purpose or position on the sample.

    Attributes:
        name (str): The name given to the VirtualDAC object
        instruments (list): a list of qcodes instruments
        gate_map (dict): the map between IVVI dac and gate
        rc_times (dict): dictionary with rc times for the gates
    """

    def __init__(self, name, instruments, gate_map, rc_times=None, **kwargs):
        """ Virtual instrument providing a translation to physical instruments
        
        Args:
            name (str): The name given to the VirtualDAC object
            instruments (list): a list of qcodes instruments
            gate_map (dict): the map between the names gates and the physical instrument channels
            rc_times (None or dict): dictionary with rc times for the gates
        
        The format of the `gate_map` to map the gate 'P1' to dac4 of the first instrument and
        'B0' to dac3 of the second instrument is: `{'P1': (0, 4), 'B0': (0, 3)}`.
 
        If the DAC gates are connected to the sample with a bias-T then there is a typical RC time
        for a voltage change to arrive at the sample. These RC times can be provides to the instument 
        with the `rc_times` argument.
        
        """
        super().__init__(name, **kwargs)
        self._instrument_list = instruments
        self._gate_map = gate_map
        self._direct_gate_map = {}  # fast access to parameters
        self._fast_readout = True

        if rc_times is None:
            rc_times = {}
        self._rc_times = rc_times

        self.add_parameter('rc_times', get_cmd=partial(
            self._get_variable, '_rc_times'), set_cmd=False)

        # Create all functions for the gates as defined in self._gate_map
        for gate in self._gate_map.keys():
            logging.debug('gates: make gate %s' % gate)
            self._make_gate(gate)

            gatemap = self._gate_map[gate]
            i = self._instrument_list[gatemap[0]]
            igate = 'dac%d' % gatemap[1]
            self._direct_gate_map[gate] = getattr(i, igate)
        self.get_all()

    def _get_variable(self, v):
        return getattr(self, v)

    def get_idn(self):
        """ Overrule because the default VISA command does not work. """
        IDN = {'vendor': 'QuTech', 'model': 'gates',
               'serial': None, 'firmware': None}
        return IDN

    def get_all(self, verbose=0):
        """ Returns all gate values. """
        for gate in sorted(self._gate_map.keys()):
            self.get(gate)
            if verbose:
                print('%s: %f' % (gate, self.get(gate)))

    def _get(self, gate, fast_readout=False):
        if self._direct_gate_map is not None:
            param = self._direct_gate_map[gate]
            if fast_readout:
                return param.get_latest()
            else:
                return param.get()

        gatemap = self._gate_map[gate]
        gate = 'dac%d' % gatemap[1]
        if fast_readout:
            return self._instrument_list[gatemap[0]].get_latest(gate)
        else:
            return self._instrument_list[gatemap[0]].get(gate)

    def _set(self, value, gate):
        value = float(value)

        if self._direct_gate_map is not None:
            param = self._direct_gate_map[gate]
            param.set(value)
            return

        gatemap = self._gate_map[gate]
        i = self._instrument_list[gatemap[0]]
        gate = 'dac%d' % gatemap[1]
        i.set(gate, value)

    def _set_wrap(self, value, gate):
        self.set(param_name=gate, value=value)

    def _make_gate(self, gate):
        self.add_parameter(gate,
                           label='%s' % gate,  # (\u03bcV)',
                           unit='mV',
                           get_cmd=partial(self._get, gate=gate),
                           set_cmd=partial(self._set, gate=gate))
        self.add_function(
            'get_{}'.format(gate), call_cmd=partial(self.get, param_name=gate))
        self.add_function('set_{}'.format(gate), call_cmd=partial(
            self._set_wrap, gate=gate), args=[Numbers()])

    def get_instrument_parameter(self, g):
        """ Returns the dac parameter mapped to this gate. """
        gatemap = self._gate_map[g]
        return getattr(self._instrument_list[gatemap[0]], 'dac%d' % gatemap[1])

    def set_boundaries(self, gate_boundaries):
        """ Set boundaries on the values that can be set on the gates. 

        Assigns a range of values to the validator of a parameter.

        Args:
            gate_boundaries (dict): a range of allowed values per parameter.
        """
        for g, bnds in gate_boundaries.items():
            logging.debug('gate %s: %s' % (g, bnds))

            gx = self._gate_map.get(g, None)
            if gx is None:
                # gate is not connected
                continue
            instrument = self._instrument_list[gx[0]]
            param = self.get_instrument_parameter(g)
            param.vals = Numbers(bnds[0], max_value=bnds[1])
            if hasattr(instrument, 'adjust_parameter_validator'):
                instrument.adjust_parameter_validator(param)

    def __repr__(self):
        gm = getattr(self, '_gate_map', [])
        s = 'gates: %s (%d gates)' % (self.name, len(gm))

        return s

    def allvalues(self):
        """ Return all gate values in a simple dict. """
        if self._fast_readout:
            vals = [(gate, self.parameters[gate].get_latest())
                    for gate in sorted(self._gate_map)]
        else:
            vals = [(gate, self.get(gate)) for gate in sorted(self._gate_map)]
        return dict(vals)

    def allvalues_string(self, fmt='%.3f'):
        """ Return all gate values in string format. """
        vals = self.allvalues()
        s = '{' + ','.join(['\'%s\': ' % (x,) + fmt % (vals[x], )
                            for x in vals]) + '}'
        return s

    def resetgates(gates, activegates, basevalues=None, verbose=2):
        """ Reset a set of gates to new values.

        If no new values are specified the gates will be reset to zero.

        Args:
            activegates (list or dict): gates to reset
            basevalues (dict): new values for the gates
            verbose (int): output level
        """
        if verbose:
            print('resetgates: setting gates to default values')
        for g in (activegates):
            if basevalues == None:
                val = 0
            else:
                if g in basevalues.keys():
                    val = basevalues[g]
                    if isinstance(val, np.ndarray):
                        val = float(val)
                else:
                    val = 0
            if verbose >= 2:
                print('  setting gate %s to %.1f [mV]' % (g, val))
            gates.set(g, val)

    def set_overshoot(self, gate, value, extra_delay=0.02, overshoot=4):
        """ Set gate to a value with overshoot

        This function can be used for gates with a slow RC value on the bias-T. 
        The actual overshoot is determined by the rc_times in the object.

        Args:
            gate (str): gate to set
            value (float): value to set at the gate
            extra_delay (float): ...
            overshoot (float): overshoot factor

        """
        gateparam = getattr(self, gate)
        rc = self._rc_times.get(gate, None)
        value0 = gateparam.get_latest()
        dv = value-value0

        if rc is None:
            warnings.warn('could not find rc time for gate %s' % gate)
            self.set(gate, value)
            time.sleep(1.)
        elif np.abs(dv) < 5:
            # hack for small steps
            self.set(gate, value)
            time.sleep(extra_delay)
        else:
            try:
                self.set(gate, value0+4*dv)
                time.sleep(rc/3)
            except ValueError:
                # outside safe boundaries
                warnings.warn('set outside boundaries?')
                self.set(gate, value)
                time.sleep(rc)

            self.set(gate, value)
            time.sleep(extra_delay)

    def resettodataset(self, dataset):
        """ Reset gates to the values from a previous dataset
        Args:
            dataset (qcodes.DataSet or str): the dataset or location to load from.
        """
        if isinstance(dataset, str):
            dataset = load_data(dataset)
        gatevals = dataset.metadata['allgatevalues']
        self.resetgates(gatevals, gatevals)

    def visualize(self, fig=1):
        """ Create a graphical representation of the system (needs graphviz). """
        gates = self
        dot = graphviz.Digraph(name=self.name)

        inames = [x.name for x in gates._instrument_list]

        cgates = graphviz.Digraph('cluster_gates')
        cgates.body.append('color=lightgrey')
        cgates.attr('node', style='filled', color='seagreen1')
        cgates.body.append('label="%s"' % 'gates')

        iclusters = []
        for i, iname in enumerate(inames):
            c0 = graphviz.Digraph(name='cluster_%d' % i)
            c0.body.append('style=filled')
            c0.body.append('color=grey80')

            c0.node_attr.update(style='filled', color='white')
            iclusters.append(c0)

        for g in gates._gate_map:
            xx = gates._gate_map[g]
            cgates.node(str(g), label='%s' % g)

            ix = inames[xx[0]] + '%d' % xx[1]
            ixlabel = 'dac%d' % xx[1]
            icluster = iclusters[xx[0]]
            icluster.node(ix, label=ixlabel, color='lightskyblue')

        for i, iname in enumerate(inames):
            iclusters[i].body.append('label="%s"' % iname)

        dot.subgraph(cgates)
        for c0 in iclusters:
            dot.subgraph(c0)

        for g in gates._gate_map:
            xx = gates._gate_map[g]
            ix = inames[xx[0]] + '%d' % xx[1]
            dot.edge(str(g), str(ix))

        return dot

virtual_IVVI = VirtualDAC