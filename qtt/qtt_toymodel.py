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

import pyqtgraph
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui # as QtGui
from functools import partial

#import PyQt4 as Qt; import PyQt4.QtGui as QtGui


import qtt

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


class DummyModel(MockModel):

    ''' Dummy model for testing '''
    def __init__(self, name, gate_map, **kwargs):
        self._data = dict()
        #self.name = name

        self.gate_map=gate_map
        for instr in [ 'ivvi1', 'ivvi2', 'keithley1', 'keithley2']:
            if not instr in self._data:
                self._data[instr] = dict()
            if 1:
                setattr(self, '%s_get' % instr, partial(self._generic_get, instr ) )
                setattr(self, '%s_set' % instr, partial( self._generic_set, instr) )
            else:
                setattr(self, '%s_get' % instr, self._dummy_get )
                setattr(self, '%s_set' % instr, self._dummy_set )
            
        #self._data['gates']=dict()
        self._data['keithley3']=dict()
        self._data['keithley3']['amplitude']=.5
        
        super().__init__(name=name)

    def _dummy_get(self, param):
        return 0
    def _dummy_set(self, param, value):
        pass
        
    def gate2ivvi(self,g):
        i, j = self.gate_map[g]
        return 'ivvi%d' % (i+1), 'c%d'  % j
    def compute(self):
        ''' Compute output of the model '''

        logging.debug('compute')
        # current through keithley1, 2 and 3

        # FIXME: loop over the gates instead of the dacs...
        v = float(self._data['ivvi1']['c11'])
        c = qtt.logistic(v, -200., 1 / 40.)
        if 1:
            for jj, g in enumerate(['P1', 'P2', 'P3', 'P4']):
                i, j = self.gate2ivvi(g)
                v = float(self._data[i][j] )
                c=c*qtt.logistic(v, -200.+jj*5, 1 / 40.)
            for jj, g in enumerate(['D1', 'D2', 'D3', 'L', 'R']):
                i, j = self.gate2ivvi(g)
                v = float(self._data[i][j] )
                c=c*qtt.logistic(v, -200.+jj*5, 1 / 40.)
        instrument = 'keithley3'
        if not instrument in self._data:
            self._data[instrument] = dict()
        val=c + (np.random.rand()-.5) / 20.
        logging.debug('compute: value %f' % val)
        self._data[instrument]['amplitude'] = val

        return c

    def _generic_get(self, instrument, parameter):
        if not parameter in self._data[instrument]:
            self._data[instrument][parameter]=0
        return self._data[instrument][parameter]

    def _generic_set(self, instrument, parameter, value):
        logging.debug('_generic_set: param %s, val %s' % (parameter, value))
        self._data[instrument][parameter] = float(value)

     
    def keithley3_get(self, param):
        logging.debug('keithley3_get: %s' % param)
        self.compute()        
        return self._generic_get('keithley3', param)

    def keithley3_set(self, param, value):
        self.compute()        
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

        logging.debug('add gates function')
        for i, g in enumerate(gates):
            self.add_function(
                'get_{}'.format(g), call_cmd=partial(self.get, g))
            logging.debug('add gates function %s: %s' % (self.name, g) )
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

    def __init__(self, name, instruments, gate_map, model=None, **kwargs):
        super().__init__(name, model=model, **kwargs)
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

#%%

import time
import threading


class QCodesTimer(threading.Thread):

    def __init__(self, fn, dt=2, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.dt = dt

    def run(self):
        while 1:
            logging.debug('QCodesTimer: start sleep')
            time.sleep(self.dt)
            # do something
            logging.debug('QCodesTimer: run!')
            self.fn()
            
class ParameterViewer(Qt.QtGui.QTreeWidget):

    ''' Simple class to show qcodes parameters '''
    def __init__(self, station, instrumentnames=['gates'], name='QuTech Parameter Viewer', **kwargs):
        super().__init__(**kwargs)
        w = self
        w.setGeometry(1700, 50, 300, 600)
        w.setColumnCount(3)
        header = QtGui.QTreeWidgetItem(["Parameter", "Value"])
        w.setHeaderItem(header)
                        # Another alternative is
                        # setHeaderLabels(["Tree","First",...])
        w.setWindowTitle(name)

        self._instrumentnames=instrumentnames
        self._itemsdict = dict()
        self._itemsdict['gates'] = dict()
        self._timer = None
        self._station = station
        self.init()
        self.show()

        self.callbacklist=[]
        
    def init(self):
        ''' Initialize parameter viewer '''
        if self._station==None:
            return
        dd = self._station.snapshot()
        #x = 
        for iname in self._instrumentnames:
            pp = dd['instruments'][iname]['parameters']
            gatesroot = QtGui.QTreeWidgetItem(self, [iname])
            for g in pp:
                # ww=['gates', g]
                value = pp[g]['value']
                A = QtGui.QTreeWidgetItem(gatesroot, [g, str(value)])
                self._itemsdict['gates'][g] = A
        self.setSortingEnabled(True)
        self.expandAll()

    def updatecallback(self, start=True):
        if self._timer is not None:
            del self._timer
            
        if start:
            self._timer = QCodesTimer(fn=self.updatedata)
            self._timer.start()
        else:
            self._timer = None
        
        
    def updatedata(self):
        ''' Update data in viewer using station.snapshow '''
        dd = self._station.snapshot()
        gates = dd['instruments']['gates']
        pp = gates['parameters']
        # gatesroot = QtGui.QTreeWidgetItem(w, ["gates"])
        for g in pp:
            #ww = ['gates', g]
            value = pp[g]['value']
            x = self._itemsdict['gates'][g]
            logging.debug('update %s to %s' % (g, value))
            x.setText(1, str(value))

        for f in self.callbacklist:
            try:
                f()
            except Exception as e:
                logging.debug('update function failed')                  
                logging.debug(str(e))
                
            