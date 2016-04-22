# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
# pylint: disable=invalid-name

import logging
import numpy as np

#import qcodes

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
    def __init__(self, name, **kwargs):
        self._data = dict()
        #self.name = name

        for instr in ['gates', 'ivvi1x', 'ivvi2']:
            #setattr(self, '%s_get' % instr, partial(self._generic_get, instrument=instr) )
            #setattr(self, '%s_set' % instr, partial(self._generic_set, instrument=instr) )
            if not instr in self._data:
                self._data[instr] = dict()
            #setattr(self, '%s_get' % instr, lambda parameter: self._generic_get( instrument=instr, parameter=parameter) )
            #setattr(self, '%s_set' % instr, lambda parameter, value: self._generic_set( instrument=instr, parameter=parameter, value=value) )
            if 1:
                setattr(self, '%s_get' % instr, partial(self._generic_get, instr ) )
                setattr(self, '%s_set' % instr, partial( self._generic_set, instr) )
            else:
                setattr(self, '%s_get' % instr, self._dummy_get )
                setattr(self, '%s_set' % instr, self._dummy_set )
            
        self._data['ivvi1'] = dict()
        self._data['gates']=dict()
        self._data['keithley3']=dict()
        self._data['keithley3']['amplitude']=.5
        
        super().__init__(name=name)

    def _dummy_get(self, param):
        return 0
    def _dummy_set(self, param, value):
        pass
        
    def compute(self):
        ''' Compute output of the model '''

        logging.debug('compute')
        # current through keithley1, 2 and 3

        v = float(self._data['ivvi1']['c11'])
        c = qtt.logistic(v, 0., 1 / 40.)

        instrument = 'keithley3'
        if not instrument in self._data:
            self._data[instrument] = dict()
        val=c + np.random.rand() / 10.
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

    def gates_get2(self, param):
        logging.debug('gates_get: %s' % param)
        return self._data['gates']['param']
        #return 1
        
    def gates_set2(self, param, value):
        logging.debug('gates_set: %s value %s' % ( param, value) )
        print('gates_set: %s value %s' % ( param, value) )
        self._data['gates']['param']=value        
        #self._generic_set('ivvi1', param, value)
        pass
        
    def ivvi1_get(self, param):
        logging.debug('ivvi1_get: %s' % param)
        return self._generic_get('ivvi1', param)
        
    def ivvi1_set(self, param, value):
        logging.debug('ivvi1_set: %s value %s' % ( param, value) )
        print('ivvi1_set: before generic_set')
        self._generic_set('ivvi1', param, value)
        print('ivvi1_set: after generic_set')

    def meter_get(self, param):
        return self.keithley3_get(param)
        
    def keithley3_get(self, param):
        logging.debug('keithley3_get: %s' % param)
        self.compute()        
        return self._generic_get('keithley3', param)

    def keithley3_set(self, param, value):
        self.compute()        
        pass
        print('huh?')
        #return self._generic_get('keithley3', param)


    def write_old(self, instrument, parameter, value):
        if not instrument in self._data:
            self._data[instrument] = dict()

        self._data[instrument][parameter] = value
        return

    def ask_old(self, instrument, parameter):
        if not instrument in self._data:
            raise ModelError('could not read from instrument %s the parameter %s ' %
                             (instrument, parameter))
        try:
            # logging.warning('here')
            self.compute()
            # logging.warning('here2')
            value = self._data[instrument][parameter]
        except Exception as ex:
            print(ex)
            raise ModelError('could not read from instrument %s the parameter %s ' %
                             (instrument, parameter))

        return value


class VirtualIVVI(MockInstrument):

    ''' Virtual instrument representing an IVVI '''

    def __init__(self, name, model, gates=['c%d' % i for i in range(1, 17)], **kwargs):
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
                           label='Current (nA)',
                           get_cmd='amplitude?',
                           get_parser=float)

        self.add_function('readnext', call_cmd=partial(self.get, 'amplitude'))


#%%

class MyInstrument(Instrument):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)


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
                           label='Gate (mV)',  # (\u03bcV)',
                           get_cmd=partial(self._get, gate=gate),
                           set_cmd=partial(self._set, gate=gate),
                           #get_parser=float,
                           vals=Numbers(-800, 400), )
                           # sweep_step=0.1,
                           # sweep_delay=0.05)
        self.add_function(
            'get_{}'.format(gate), call_cmd=partial(self.get, param_name=gate))
        self.add_function('set_{}'.format(gate), call_cmd=partial(
            self._set_wrap, gate=gate), args=[Numbers()])

    def __repr__(self):
        s = 'virtual_gates: %s (%d gates)' % (self.name, len(self._gate_map))

        return s
        # func = lambda voltage: self._do_set_gate(voltage, gate)
        # setattr(self, '_do_set_%s' %gate, func)



#%%

import threading
import time


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

    def __init__(self, station, name='QuTech Parameter Viewer', **kwargs):
        super().__init__(**kwargs)
        w = self
        w.setGeometry(1700, 50, 300, 600)
        w.setColumnCount(3)
        header = QtGui.QTreeWidgetItem(["Parameter", "Value"])
        w.setHeaderItem(header)
                        # Another alternative is
                        # setHeaderLabels(["Tree","First",...])
        w.setWindowTitle(name)

        self._itemsdict = dict()
        self._itemsdict['gates'] = dict()
        self._timer = None
        self._station = station
        self.init()
        self.show()

    def init(self):
        ''' Initialize parameter viewer '''
        dd = self._station.snapshot()
        #x = 
        pp = dd['instruments']['gates']['parameters']
        gatesroot = QtGui.QTreeWidgetItem(self, ["gates"])
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
