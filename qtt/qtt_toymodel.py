# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
# pylint: disable=invalid-name

import logging
import numpy as np

import qcodes as qc
from qcodes import Instrument, MockInstrument, MockModel # , Parameter, Loop, DataArray
from qcodes.utils.validators import Numbers

#import sys
#sys.stdout.write=logging.info

import pyqtgraph
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui # as QtGui
from functools import partial

#import PyQt4 as Qt; import PyQt4.QtGui as QtGui


import qtt

logger = logging.getLogger('qtt')

#%%

def logTest():
    logging.info('info')
    logging.warning('warning')
    logging.debug('debug')

def partiala(method, **kwargs):
  ''' Function to perform functools.partial on named arguments '''
  def t(x):
    return method(x, **kwargs)
  return t


class ModelError(Exception):
    ''' Error in Model '''
    pass


import qtt.simulation.dotsystem
from qtt.simulation.dotsystem import DotSystem, TripleDot, FourDot, GateTransform

import traceback

class MockModelLocal:
    ''' Same as MockModel, but without a server    '''
    def __init__(self, name='Model-{:.7s}'):
        pass

    def ask(self, *query):
        return self.do_query(*query)

    def write(self, *query):
        self.do_query(*query)

    def do_query(self, *query):
        fullquery=query
        query=query[1:]
        query_args = query[1:]
        query = query[0].split(':')

        instrument = query[0]

        param = query[1]
        if param[-1] == '?' and len(query) == 2:
            getter = getattr(self, instrument + '_get')
            return getter(param[:-1])
        elif len(query) <= 3:
            value = query[2] if len(query) == 3 else None
            setter = getattr(self, instrument + '_set')
            setter(param, value)
        else:
            raise ValueError

from qtt.simulation.dotsystem import *

def dotConductance(self, index=0, verbose=0, T=6):
    ''' Calculate conductance in dot due to Coulomb peak tunneling '''

    cond=0
    E0=self.energies[0]
    for i, e in enumerate(self.energies):
        fac=np.exp(-e/T)/np.exp(-E0/T)
        if verbose:
            print('energy: %f: factor %f'  % (e, fac))
        v=self.stateoccs[i]-self.stateoccs[0]
        v[index]
        cond+=np.abs(v[index])*fac
    return cond
            
class DummyModel(MockModelLocal):

    ''' Dummy model for testing '''
    def __init__(self, name, gate_map, **kwargs):
        self._data = dict()
        #self.name = name

        self.gate_map=gate_map
        for instr in [ 'ivvi1', 'ivvi2']: # , 'keithley1', 'keithley2']:
            if not instr in self._data:
                self._data[instr] = dict()
            if 1:
                setattr(self, '%s_get' % instr, partial(self._generic_get, instr ) )
                setattr(self, '%s_set' % instr, partial( self._generic_set, instr) )
            else:
                setattr(self, '%s_get' % instr, self._dummy_get )
                setattr(self, '%s_set' % instr, self._dummy_set )

        for instr in [ 'keithley1', 'keithley2']:
            if not instr in self._data:
                self._data[instr] = dict()
            if 1:
                setattr(self, '%s_set' % instr, partial( self._generic_set, instr) )

        #self._data['gates']=dict()
        self._data['keithley3']=dict()
        self._data['keithley3']['amplitude']=.5

        # initialize a 4-dot system
        if 0:
            self.ds=FourDot(use_tunneling=False)

            self.targetnames=['det%d' % (i+1) for i in range(4)]
            self.sourcenames=['P%d' % (i+1) for i in range(4)]

            self.sddist1 = [6,4,2,1]
            self.sddist2 = [1,2,4,6]
        else:
            self.ds=TripleDot(maxelectrons=2)

            self.targetnames=['det%d' % (i+1) for i in range(3)]
            self.sourcenames=['P%d' % (i+1) for i in range(3)]

            self.sddist1 = [6,4,2]
            self.sddist2 = [2,4,6]

        for ii in range(self.ds.ndots):
            setattr(self.ds, 'osC%d' % ( ii+1), 55)
        for ii in range(self.ds.ndots-1):
            setattr(self.ds, 'isC%d' % (ii+1), 3)

        Vmatrix = qtt.simulation.dotsystem.defaultVmatrix(n=self.ds.ndots)
        Vmatrix[0:3,-1]=[100,100,100]
        self.gate_transform = GateTransform(Vmatrix, self.sourcenames, self.targetnames)

        # coulomb model for sd1
        self.sd1ds=qtt.simulation.dotsystem.OneDot()
        defaultDotValues(self.sd1ds)
        Vmatrix=np.matrix([[.1, 1, .1, 300.],[0,0,0,1]])
        self.gate_transform_sd1 = GateTransform(Vmatrix, ['SD1a','SD1b','SD1c'], ['det1'])
        
        super().__init__(name=name)

    def _dummy_get(self, param):
        return 0
    def _dummy_set(self, param, value):
        pass

    def gate2ivvi(self,g):
        i, j = self.gate_map[g]
        return 'ivvi%d' % (i+1), 'c%d'  % j

    def gate2ivvi_value(self,g):
        i, j = self.gate2ivvi(g)
        value= self._data[i][j]
        return value

    def get_gate(self, g):
        return self.gate2ivvi_value(g)

    def _calculate_pinchoff(self, gates, offset=-200., random=0):
        c = 1
        for jj, g in enumerate(gates):
            i, j = self.gate2ivvi(g)
            v = float(self._data[i][j] )
            c=c*qtt.logistic(v, offset+jj*5, 1 / 40.)
        val = c
        if random:
            val=c + (np.random.rand()-.5) * random
        return val

    def computeSD(self, usediag=True, verbose=0):
        logger.debug('start SD computation')

        # main contribution
        val1 = self._calculate_pinchoff(['SD1a', 'SD1b', 'SD1c'], offset=-50, random=.1)
        val2 = self._calculate_pinchoff(['SD2a','SD2b', 'SD2c'], offset=-50, random=.1)

        # coulomb system for dot1
        ds=self.sd1ds
        gate_transform_sd1=self.gate_transform_sd1
        gv=[float(self.get_gate(g)) for g in gate_transform_sd1.sourcenames ]
        setDotSystem(ds, gate_transform_sd1, gv)        
        ds.makeHsparse()
        ds.solveH(usediag=True)        
        _=ds.findcurrentoccupancy()
        cond1=.75*dotConductance(ds, index=0, T=3)  
        if verbose>=2:
            print('k1 %f, cond %f' % (k1, cond) )

        # contribution of charge from bottom dots
        gv=[self.get_gate(g) for g in self.sourcenames ]
        tv=self.gate_transform.transformGateScan(gv)
        ds=self.ds
        for k, val in tv.items():
            if verbose:
                print('compudateSD: %d, %f'  % (k,val) )
            setattr(ds, k, val)
        ds.makeHsparse()
        ds.solveH(usediag=usediag)
        ret = ds.OCC

        sd1=(1/np.sum(self.sddist1))*(ret*self.sddist1).sum()
        sd2=(1/np.sum(self.sddist1))*(ret*self.sddist2).sum()

        #c1=self._compute_pinchoff(['SD1b'], offset=-200.)
        #c2=self._compute_pinchoff(['SD1b'], offset=-200.)

        return [val1+sd1+cond1, val2+sd2]


    def compute(self, random=0.02):
        ''' Compute output of the model '''

        try:
            logger.debug('DummyModel: compute values')
            # current through keithley1, 2 and 3

            #v = float(self._data['ivvi1']['c11'])
            c = 1
            if 1:
                c*=self._calculate_pinchoff(['P1', 'P2', 'P3', 'P4'], offset=-200.)
                c*=self._calculate_pinchoff(['D1', 'D2', 'D3', 'L', 'R'], offset=-150.)
            instrument = 'keithley3'
            if not instrument in self._data:
                self._data[instrument] = dict()
            val=c + random*(np.random.rand()-.5) 
            logging.debug('compute: value %f' % val)

            self._data[instrument]['amplitude'] = val


        except Exception as ex:
            #print(ex)
            #raise ex
            msg = traceback.format_exc(ex)
            logging.warning('compute failed! %s' % msg)
        return c

    def _generic_get(self, instrument, parameter):
        if not parameter in self._data[instrument]:
            self._data[instrument][parameter]=0
        return self._data[instrument][parameter]

    def _generic_set(self, instrument, parameter, value):
        logging.debug('_generic_set: param %s, val %s' % (parameter, value))
        self._data[instrument][parameter] = float(value)

    def keithley1_get(self, param):
        sd1, sd2 = self.computeSD()
        self._data['keithley1']['amplitude'] = sd1
        self._data['keithley2']['amplitude'] = sd2
        #print('keithley1_get: %f %f' % (sd1, sd2))
        return  self._generic_get('keithley1', param)

    def keithley2_get(self, param):
        sd1, sd2 = self.computeSD()
        self._data['keithley1']['amplitude'] = sd1
        self._data['keithley2']['amplitude'] = sd2
        return  self._generic_get('keithley2', param)

    def keithley3_get(self, param):
        logging.debug('keithley3_get: %s' % param)
        self.compute()
        return self._generic_get('keithley3', param)

    def keithley3_set(self, param, value):
        #self.compute()
        pass
        print('huh?')
        #return self._generic_get('keithley3', param)


class VirtualIVVI(MockInstrument):

    ''' Virtual instrument representing an IVVI '''

    def __init__(self, name, model, gates=['c%d' % i for i in range(1, 17)], mydebug=False, **kwargs):
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
                               vals=Numbers(-800, 400))

        self.add_function('reset', call_cmd='rst')

        logger.debug('add gates function')
        for i, g in enumerate(gates):
            self.add_function(
                'get_{}'.format(g), call_cmd=partial(self.get, g))
            logger.debug('add gates function %s: %s' % (self.name, g) )
            #model.write('%s:%s %f' % (self.name, g, 0) )

        if not mydebug:
            self.get_all()

    def get_all(self):
        ''' Get all parameters in instrument '''
        for g in self._gates:
            logging.debug('get_all %s: %s' % (self.name, g) )
            self.get(g)

    def __repr__(self):
        ''' Return string description instance '''
        return 'VirtualIVVI: %s' % self.name


#%%


class MockSource(MockInstrument):

    def __init__(self, name, model, **kwargs):
        ''' Dummy source object '''
        super().__init__(name, model=model, **kwargs)

        # this parameter uses built-in sweeping to change slowly
        self.add_parameter('amplitude',
                           label='Source Amplitude (\u03bcV)',
                           get_cmd='amplitude?',
                           set_cmd='amplitude {:.4f}',
                           get_parser=float,
                           vals=Numbers(0, 10),
                           sweep_step=0.1,
                           sweep_delay=0.05)


class MockMeter(MockInstrument):

    def __init__(self, name, model, **kwargs):
        super().__init__(name, model=model, **kwargs)

        self.add_parameter('amplitude',
                           label='%s Current (nA)' % name,
                           get_cmd='amplitude?',
                           get_parser=float)

        #self.add_function('readnext', call_cmd=partial(self.get, 'amplitude'))
        self.add_parameter('readnext', get_cmd=partial(self.get, 'amplitude'), label=name)


#%%

class MyInstrument(Instrument):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

try:
    import graphviz
except:
        pass

# from qcodes.utils.validators import Validator
class virtual_gates(Instrument):

    def __init__(self, name, instruments, gate_map, **kwargs):
        super().__init__(name, **kwargs)
        self._instrument_list = instruments
        self._gate_map = gate_map
        # Create all functions for the gates as defined in self._gate_map
        for gate in self._gate_map.keys():
            logging.debug('virtual_gates: make gate %s' % gate)
            self._make_gate(gate)

        self.get_all()

    def get_all(self):
        for gate in self._gate_map.keys():
            self.get(gate)

    def _get(self, gate):
        gatemap = self._gate_map[gate]
        gate = 'c%d' % gatemap[1]
        logging.debug('_get: %s %s'  % (gatemap[0], gate) )
        return self._instrument_list[gatemap[0]].get(gate)

    def _set(self, value, gate):
        logging.debug('virtualgate._set: gate %s, value %s' % (gate, value))
        gatemap = self._gate_map[gate]
        i = self._instrument_list[gatemap[0]]
        gate = 'c%d' % gatemap[1]
        logging.debug('virtualgate._set: instrument %s, param %s: value %s' %
                      (i.name, gate, value))
        i.set(gate, value)

    def _set_wrap(self, value, gate):
        self.set(param_name=gate, value=value)

    def _make_gate(self, gate):
        self.add_parameter(gate,
                           label='%s (mV)' % gate,  # (\u03bcV)',
                           get_cmd=partial(self._get, gate=gate),
                           set_cmd=partial(self._set, gate=gate),
                           #get_parser=float,
                           vals=Numbers(-2000, 2000), )
                           # sweep_step=0.1,
                           # sweep_delay=0.05)
        self.add_function(
            'get_{}'.format(gate), call_cmd=partial(self.get, param_name=gate))
        self.add_function('set_{}'.format(gate), call_cmd=partial(
            self._set_wrap, gate=gate), args=[Numbers()])

    def get_instrument_parameter(self, g):
        gatemap = self._gate_map[g]
        return getattr(self._instrument_list[gatemap[0]], 'c%d' % gatemap[1] )


    def set_boundaries(self, gate_boundaries):
        for g, bnds in gate_boundaries.items():
            logging.debug('gate %s: %s' % (g, bnds))

            param = self.get_instrument_parameter(g)
            param._vals=Numbers(bnds[0], max_value=bnds[1])


    def __repr__(self):
        s = 'virtual_gates: %s (%d gates)' % (self.name, len(self._gate_map))

        return s
        # func = lambda voltage: self._do_set_gate(voltage, gate)
        # setattr(self, '_do_set_%s' %gate, func)


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
        gates=self
        dot=graphviz.Digraph(name=self.name)

        inames = [x.name for x in gates._instrument_list]

        cgates=graphviz.Digraph('cluster_gates')
        cgates.body.append('color=lightgrey')
        cgates.attr('node', style='filled', color='seagreen1')
        cgates.body.append('label="%s"' % 'Virtual gates')

        iclusters=[]
        for i, iname in enumerate(inames):
            c0=graphviz.Digraph(name='cluster_%d' % i)
            c0.body.append('style=filled')
            c0.body.append('color=grey80')

            c0.node_attr.update(style='filled', color='white')
            #c0.attr('node', style='filled', color='lightblue')
            iclusters.append(c0)

        for g in gates._gate_map:
            xx=gates._gate_map[g]
            cgates.node(str(g), label='%s' % g)

            ix = inames[xx[0]] + '%d' % xx[1]
            ixlabel='c%d' % xx[1]
            icluster=iclusters[xx[0]]
            icluster.node(ix, label=ixlabel, color='lightskyblue')

        for i, iname in enumerate(inames):
            iclusters[i].body.append('label="%s"' % iname)

        dot.subgraph(cgates)
        for c0 in iclusters:
            dot.subgraph(c0)
        # group

        for g in gates._gate_map:
            xx=gates._gate_map[g]
            ix = inames[xx[0]] + '%d' % xx[1]
            dot.edge(str(g), str(ix))

        return dot

