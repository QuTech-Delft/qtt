# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
# pylint: disable=invalid-name

import os
import logging
import numpy as np

import qcodes as qc
from qcodes import Instrument   # , Parameter, Loop, DataArray
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import ManualParameter

import pyqtgraph
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui 
from functools import partial
import threading

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

#import traceback



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
            


class FourdotModel(Instrument):

    ''' Dummy model for testing '''
    def __init__(self, name, gate_map, **kwargs):
        super().__init__(name, **kwargs)

        self._data = dict()
        #self.name = name
        self.lock = threading.Lock() # lock to be able to use the simulation with threading

        # make parameters for all the gates...
        gateset= set(gate_map.values())
        gateset= [ (i, a) for a in range(1,17) for i in range(2)]
        for i, idx in gateset:
            g='ivvi%d_dac%d' % (i+1,idx)
            logging.debug('add gate %s' % g )
            self.add_parameter(g,
                               label='Gate {} (mV)'.format(g),
                               get_cmd=partial(self._data_get, g),
                               set_cmd=partial(self._data_set, g),
                               get_parser=float,
                               )


        self.sdrandom = .001 # random noise in SD
        self.biasrandom = .01 # random noise in bias current
        
        self.gate_map=gate_map
        for instr in [ 'ivvi1', 'ivvi2']: # , 'keithley1', 'keithley2']:
            #if not instr in self._data:
            #    self._data[instr] = dict()
            setattr(self, '%s_get' % instr, self._dummy_get )
            setattr(self, '%s_set' % instr, self._dummy_set )

        for instr in [ 'keithley1', 'keithley2', 'keithley3']:
            if not instr in self._data:
                self._data[instr] = dict()
            if 0:
                setattr(self, '%s_set' % instr, partial( self._generic_set, instr) )
            g=instr+'_amplitude'
            self.add_parameter(g,
                               #label='Gate {} (mV)'.format(g),
                               #get_cmd=getattr(self, instr+'_get' ),
                               get_cmd=partial(getattr(self, instr+'_get' ),'amplitude'),
                               #set_cmd=partial(self._data_set, g),
                               get_parser=float,
                               )

        if 1:
            # initialize a 4-dot system
            self.ds=FourDot(use_tunneling=False)

            self.targetnames=['det%d' % (i+1) for i in range(4)]
            self.sourcenames=['P%d' % (i+1) for i in range(4)]

            self.sddist1 = [6,4,2,1]
            self.sddist2 = [1,2,4,6]
        else:
            # altenative: 3-dot (faster calculations...)

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
        Vmatrix=np.matrix([[.23, 1, .23, 300.],[0,0,0,1]])
        self.gate_transform_sd1 = GateTransform(Vmatrix, ['SD1a','SD1b','SD1c'], ['det1'])
        
        #super().__init__(name=name)


    def _dummy_get(self, param):
        return 0
    def _dummy_set(self, param, value):
        pass

    def _data_get(self, param):
        return self._data.get(param, 0)
    def _data_set(self, param, value):
        self._data[param]=value
        return

    def gate2ivvi(self,g):
        i, j = self.gate_map[g]
        return 'ivvi%d' % (i+1), 'dac%d'  % j

    def gate2ivvi_value(self,g):
        i, j = self.gate2ivvi(g)
        #value= self._data[i].get(j, 0)
        value = self._data.get(i+'_'+j, 0)
        return value

    def get_gate(self, g):
        return self.gate2ivvi_value(g)

    def _calculate_pinchoff(self, gates, offset=-200., random=0):
        c = 1
        for jj, g in enumerate(gates):
            v = self.gate2ivvi_value(g)
            #i, j = self.gate2ivvi(g)
            #v = float( self._data[i].get(j, 0) )
            c=c*qtt.logistic(v, offset+jj*5, 1 / 40.)
        val = c
        if random:
            val=c + (np.random.rand()-.5) * random
        return val

    def computeSD(self, usediag=True, verbose=0):
        logger.debug('start SD computation')

        # main contribution
        val1 = self._calculate_pinchoff(['SD1a', 'SD1b', 'SD1c'], offset=-150, random= self.sdrandom)
        val2 = self._calculate_pinchoff(['SD2a','SD2b', 'SD2c'], offset=-150, random= self.sdrandom)

        val1x = self._calculate_pinchoff(['SD1a', 'SD1c'], offset=-50, random=0)

        # coulomb system for dot1
        ds=self.sd1ds
        gate_transform_sd1=self.gate_transform_sd1
        gv=[float(self.get_gate(g)) for g in gate_transform_sd1.sourcenames ]
        setDotSystem(ds, gate_transform_sd1, gv)        
        ds.makeHsparse()
        ds.solveH(usediag=True)        
        _=ds.findcurrentoccupancy()
        cond1=.75*dotConductance(ds, index=0, T=3)  
        
        cond1=cond1*np.prod( (1-val1x) )
        if verbose>=2:
            print('k1 %f, cond %f' % (k1, cond) )

        # contribution of charge from bottom dots
        gv=[self.get_gate(g) for g in self.sourcenames ]
        tv=self.gate_transform.transformGateScan(gv)
        ds=self.ds
        for k, val in tv.items():
            if verbose:
                print('computeSD: %d, %f'  % (k,val) )
            setattr(ds, k, val)
        ds.makeHsparse()
        ds.solveH(usediag=usediag)
        ret = ds.OCC

        sd1=(1/np.sum(self.sddist1))*(ret*self.sddist1).sum()
        sd2=(1/np.sum(self.sddist1))*(ret*self.sddist2).sum()

        #c1=self._compute_pinchoff(['SD1b'], offset=-200.)
        #c2=self._compute_pinchoff(['SD1b'], offset=-200.)

        return [val1+sd1+cond1, val2+sd2]


    def compute(self, biasrandom=None):
        ''' Compute output of the model '''

        if biasrandom is None:
            biasrandom = self.biasrandom
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
            val=c + self.biasrandom*(np.random.rand()-.5) 
            logging.debug('compute: value %f' % val)

            self._data[instrument]['amplitude'] = val


        except Exception as ex:
            print(ex)
            logging.warning(ex)
            #raise ex
            #msg = traceback.format_exc(ex)
            #logging.warning('compute failed! %s' % msg)
        return c

    def _generic_get(self, instrument, parameter):
        if not parameter in self._data[instrument]:
            self._data[instrument][parameter]=0
        return self._data[instrument][parameter]

    def _generic_set(self, instrument, parameter, value):
        logging.debug('_generic_set: param %s, val %s' % (parameter, value))
        self._data[instrument][parameter] = float(value)

    def keithley1_get(self, param):
        self.lock.acquire()

        sd1, sd2 = self.computeSD()
        self._data['keithley1']['amplitude'] = sd1
        self._data['keithley2']['amplitude'] = sd2
        #print('keithley1_get: %f %f' % (sd1, sd2))
        val= self._generic_get('keithley1', param)
        self.lock.release()
        return val

    def keithley2_get(self, param):
        self.lock.acquire()
        sd1, sd2 = self.computeSD()
        self._data['keithley1']['amplitude'] = sd1
        self._data['keithley2']['amplitude'] = sd2
        val = self._generic_get('keithley2', param)
        self.lock.release()
        return val

    def keithley3_get(self, param):
        self.lock.acquire()
        logging.debug('keithley3_get: %s' % param)
        self.compute()
        val= self._generic_get('keithley3', param)
        self.lock.release()
        return val

#    def keithley3_set(self, param, value):
#        print('huh?')
        
class VirtualIVVI(Instrument):

    shared_kwargs = ['model']
    
    ''' Virtual instrument representing an IVVI '''
    def __init__(self, name, model, gates=['dac%d' % i for i in range(1, 17)], mydebug=False, **kwargs):
        super().__init__(name, **kwargs)

        self.model = model
        self._gates = gates
        logging.debug('add gates')
        for i, g in enumerate(gates):
            logging.debug('VirtualIVVI: add gate %s' % g )
            if model is None:
                self.add_parameter(g,
                               parameter_class=ManualParameter,
                               initial_value=0,
                               label='Gate {} (arb. units)'.format(g),
                               vals=Numbers(-800, 400))
            else:
                self.add_parameter(g,
                               label='Gate {} (mV)'.format(g),
                               get_cmd=partial(self.get_gate, g),
                               set_cmd=partial(self.set_gate, g),
                               #get_parser=float,
                               vals=Numbers(-800, 400))

        self.add_function('reset', call_cmd='rst')

        logger.debug('add legacy gates to VirtualIVVI')
        for i, g in enumerate(gates):
            self.add_function(
                'get_{}'.format(g), call_cmd=partial(self.get, g))
            logger.debug('add gates function %s: %s' % (self.name, g) )

        if not mydebug:
            self.get_all()

    def get_gate(self, gate):
        if self.model is None:
            return 0
        value = self.model.get(self.name+'_'+gate)
        logging.debug('%s: get_gate %s' % (self.name,gate) )
        return value

    def set_gate(self, gate, value):
        if self.model is None:
            return
        value=float(value)
        self.model.set(self.name+'_'+gate, value)
        logging.debug('set_gate %s: %s' % (gate, value))
        return
        
    def get_all(self):
        ''' Get all parameters in instrument '''
        for g in self._gates:
            logging.debug('get_all %s: %s' % (self.name, g) )
            self.get(g)

    def __repr__(self):
        ''' Return string description instance '''
        return 'VirtualIVVI: %s' % self.name

"""        
class VirtualIVVI2(MockInstrument):

    ''' Virtual instrument representing an IVVI '''

    def __init__(self, name, model, gates=['dac%d' % i for i in range(1, 17)], mydebug=False, **kwargs):
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

"""
#%%


class VirtualMeter(Instrument):

    shared_kwargs = ['model']

    def __init__(self, name, model = None, **kwargs):
        super().__init__(name, **kwargs)
        self.model = model
        
        g='amplitude'
        self.add_parameter(g,
                        label='%s Current (nA)' % name,
                        get_cmd=partial(self.get_gate, g),
                        #set_cmd=partial(self.set_gate, g),
                        get_parser=float)

        #self.add_function('readnext', call_cmd=partial(self.get, 'amplitude'))
        self.add_parameter('readnext', get_cmd=partial(self.get, 'amplitude'), label=name)

    def get_gate(self, gate):
        # need a remote get...
        return self.model.get(self.name+'_'+gate)
        #self.name+'_'+gate
        
        #logging.debug('%s: get_gate %s' % (self.name,gate) )
       # return 0

    def set_gate(self, gate, value):
        self.model.set(self.name+'_'+gate, value)
        #logging.info('set_gate %s: %s' % (gate, value))
        return
        
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

    # instruments will be a list of RemoteInstrument objects, which can be
    # given to a server on creation but not later on, so it needs to be
    # listed in shared_kwargs
    shared_kwargs = ['instruments']
    
    def __init__(self, name, instruments, gate_map, **kwargs):
        super().__init__(name, **kwargs)
        self._instrument_list = instruments
        self._gate_map = gate_map
        # Create all functions for the gates as defined in self._gate_map
        for gate in self._gate_map.keys():
            logging.debug('virtual_gates: make gate %s' % gate)
            self._make_gate(gate)

        if 0:
            # debugging
            print('pid %d: add to q.txt' % os.getpid())
            with open('/home/eendebakpt/tmp/q.txt', 'at') as fid:
                l=logging.getLogger()
                fid.write('virtual_gates on pid %d\n' % os.getpid() )
                fid.write('  handlers: %s\n' % (str(l.handlers)) )
                logging.info('hello from %d ' % os.getpid() )
        self.get_all()

    def get_idn(self):
        ''' Overrule because the default VISA command does not work '''
        IDN = {'vendor': 'QuTech', 'model': 'virtual_gates',
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
        logging.debug('_get: %s %s'  % (gatemap[0], gate) )
        return self._instrument_list[gatemap[0]].get(gate)

    def _set(self, value, gate):
        logging.debug('virtualgate._set: gate %s, value %s' % (gate, value))
        value=float(value)
        gatemap = self._gate_map[gate]
        i = self._instrument_list[gatemap[0]]
        gate = 'dac%d' % gatemap[1]
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
        return getattr(self._instrument_list[gatemap[0]], 'dac%d' % gatemap[1] )


    def set_boundaries(self, gate_boundaries):
        for g, bnds in gate_boundaries.items():
            logging.debug('gate %s: %s' % (g, bnds))

            param = self.get_instrument_parameter(g)
            param._vals=Numbers(bnds[0], max_value=bnds[1])


    def __repr__(self):
        gm=getattr(self, '_gate_map', [])
        s = 'virtual_gates: %s (%d gates)' % (self.name, len(gm) )

        return s

    def allvalues(self):
        """ Return all gate values in a simple dict """
        vals = [ (gate, self.get(gate) ) for gate in self._gate_map ]
        return dict(vals)

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
            ixlabel='dac%d' % xx[1]
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

