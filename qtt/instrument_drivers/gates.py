# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:36:01 2017

@author: diepencjv
"""

#%%
from qcodes import Instrument
import logging
from functools import partial
from qcodes.utils.validators import Numbers
try:
    import graphviz
except:
    pass

#%%


class gates(Instrument):
    """ Class to map named gates to gates on an instrument.

    A typical example would be to map the names of gates to the
    dacs of several IVVI's.

    Attributes:
        name (str)
        instruments (list of qcodes instruments):
        gate_map (dict)
    """
    shared_kwargs = ['instruments']

    # instruments will be a list of RemoteInstrument objects, which can be
    # given to a server on creation but not later on, so it needs to be
    # listed in shared_kwargs

    def __init__(self, name, instruments, gate_map, **kwargs):
        """ Inits gates with gate_map. """
        super().__init__(name, **kwargs)
        self._instrument_list = instruments
        self._gate_map = gate_map
        # Create all functions for the gates as defined in self._gate_map
        for gate in self._gate_map.keys():
            logging.debug('gates: make gate %s' % gate)
            self._make_gate(gate)

        self.get_all()

    def get_idn(self):
        ''' Overrule because the default VISA command does not work '''
        IDN = {'vendor': 'QuTech', 'model': 'gates',
               'serial': None, 'firmware': None}
        return IDN

    def get_all(self, verbose=0):
        for gate in sorted(self._gate_map.keys()):
            self.get(gate)
            if verbose:
                print('%s: %f' % (gate, self.get(gate)))

    def _get(self, gate):
        gatemap = self._gate_map[gate]
        gate = 'dac%d' % gatemap[1]
        logging.debug('_get: %s %s' % (gatemap[0], gate))
        return self._instrument_list[gatemap[0]].get(gate)

    def _set(self, value, gate):
        logging.debug('gate._set: gate %s, value %s' % (gate, value))
        value = float(value)
        gatemap = self._gate_map[gate]
        i = self._instrument_list[gatemap[0]]
        gate = 'dac%d' % gatemap[1]
        logging.debug('gate._set: instrument %s, param %s: value %s' %
                      (i.name, gate, value))
        i.set(gate, value)

    def _set_wrap(self, value, gate):
        self.set(param_name=gate, value=value)

    def _make_gate(self, gate):
        self.add_parameter(gate,
                           label='%s (mV)' % gate,  # (\u03bcV)',
                           units='mV',
                           get_cmd=partial(self._get, gate=gate),
                           set_cmd=partial(self._set, gate=gate),
                           vals=Numbers(-2000, 2000), )
        self.add_function(
            'get_{}'.format(gate), call_cmd=partial(self.get, param_name=gate))
        self.add_function('set_{}'.format(gate), call_cmd=partial(
            self._set_wrap, gate=gate), args=[Numbers()])

    def get_instrument_parameter(self, g):
        gatemap = self._gate_map[g]
        return getattr(self._instrument_list[gatemap[0]], 'dac%d' % gatemap[1])

    def set_boundaries(self, gate_boundaries):
        for g, bnds in gate_boundaries.items():
            logging.debug('gate %s: %s' % (g, bnds))

            param = self.get_instrument_parameter(g)
            param._vals = Numbers(bnds[0], max_value=bnds[1])

    def __repr__(self):
        gm = getattr(self, '_gate_map', [])
        s = 'gates: %s (%d gates)' % (self.name, len(gm))

        return s

    def allvalues(self):
        """ Return all gate values in a simple dict """
        vals = [(gate, self.get(gate)) for gate in sorted(self._gate_map)]
        return dict(vals)
    def allvalues_string(self):
        """ Return all gate values in string format """
        vals = self.allvalues()
        s = '{' + ','.join(['\'%s\': %.2f' % (x, vals[x]) for x in vals]) + '}'
        return s

    def resetgates(gates, activegates, basevalues=None, verbose=2):
        """ Reset a set of gates to default values

        Arguments
        ---------
            activegates : list or dict
                list of gates to reset
            basevalues: dict
                new values for the gates
            verbose : integer
                output level

        """
        if verbose:
            print('resetgates: setting gates to default values')
        for g in (activegates):
            if basevalues == None:
                val = 0
            else:
                if g in basevalues.keys():
                    val = basevalues[g]
                else:
                    val = 0
            if verbose >= 2:
                print('  setting gate %s to %.1f [mV]' % (g, val))
            gates.set(g, val)

    def visualize(self, fig=1):
        ''' Create a graphical representation of the system (needs graphviz) '''
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
            # c0.attr('node', style='filled', color='lightblue')
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
