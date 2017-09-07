# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
# pylint: disable=invalid-name

import os
import logging
import numpy as np
from functools import partial
import threading

import qcodes as qc
from qcodes import Instrument   # , Parameter, Loop, DataArray
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import ManualParameter


import qtt

logger = logging.getLogger('qtt')

import qtt.algorithms.functions
import qtt.simulation.dotsystem
from qtt.simulation.dotsystem import DotSystem, TripleDot, FourDot, GateTransform
from qtt.simulation.dotsystem import defaultDotValues
from qtt.simulation.dotsystem import DoubleDot


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


def dotConductance(self, index=0, verbose=0, T=6):
    ''' Calculate conductance in dot due to Coulomb peak tunneling '''

    cond = 0
    E0 = self.energies[0]
    for i, e in enumerate(self.energies):
        fac = np.exp(-e / T) / np.exp(-E0 / T)
        if verbose:
            print('energy: %f: factor %f' % (e, fac))
        v = self.stateoccs[i] - self.stateoccs[0]
        v[index]
        cond += np.abs(v[index]) * fac
    return cond


class FourdotModel(Instrument):

    ''' Dummy model for testing '''

    def __init__(self, name, gate_map, **kwargs):
        super().__init__(name, **kwargs)

        self._data = dict()
        # self.name = name
        self.lock = threading.Lock()  # lock to be able to use the simulation with threading

        # make parameters for all the gates...
        gateset = set(gate_map.values())
        gateset = [(i, a) for a in range(1, 17) for i in range(2)]
        for i, idx in gateset:
            g = 'ivvi%d_dac%d' % (i + 1, idx)
            logging.debug('add gate %s' % g)
            self.add_parameter(g,
                               label='Gate {} (mV)'.format(g),
                               get_cmd=partial(self._data_get, g),
                               set_cmd=partial(self._data_set, g),
                               get_parser=float,
                               )

        self.sdrandom = .001  # random noise in SD
        self.biasrandom = .01  # random noise in bias current

        self.gate_map = gate_map
        for instr in ['ivvi1', 'ivvi2']:  # , 'keithley1', 'keithley2']:
            # if not instr in self._data:
            #    self._data[instr] = dict()
            setattr(self, '%s_get' % instr, self._dummy_get)
            setattr(self, '%s_set' % instr, self._dummy_set)

        for instr in ['keithley1', 'keithley2', 'keithley3']:
            if not instr in self._data:
                self._data[instr] = dict()
            if 0:
                setattr(self, '%s_set' % instr, partial(self._generic_set, instr))
            g = instr + '_amplitude'
            self.add_parameter(g,
                               # label='Gate {} (mV)'.format(g),
                               get_cmd=partial(getattr(self, instr + '_get'), 'amplitude'),
                               get_parser=float,
                               )

        if 1:
            # initialize a 4-dot system
            self.ds = FourDot(use_tunneling=False)

            self.targetnames = ['det%d' % (i + 1) for i in range(4)]
            self.sourcenames = ['P%d' % (i + 1) for i in range(4)]

            self.sddist1 = [6, 4, 2, 1]
            self.sddist2 = [1, 2, 4, 6]
        else:
            # altenative: 3-dot (faster calculations...)

            self.ds = TripleDot(maxelectrons=2)

            self.targetnames = ['det%d' % (i + 1) for i in range(3)]
            self.sourcenames = ['P%d' % (i + 1) for i in range(3)]

            self.sddist1 = [6, 4, 2]
            self.sddist2 = [2, 4, 6]

        for ii in range(self.ds.ndots):
            setattr(self.ds, 'osC%d' % (ii + 1), 55)
        for ii in range(self.ds.ndots - 1):
            setattr(self.ds, 'isC%d' % (ii + 1), 3)

        Vmatrix = qtt.simulation.dotsystem.defaultVmatrix(n=self.ds.ndots)
        Vmatrix[0:self.ds.ndots, -1] = [-200] * self.ds.ndots
        self.gate_transform = GateTransform(Vmatrix, self.sourcenames, self.targetnames)

        # coulomb model for sd1
        self.sd1ds = qtt.simulation.dotsystem.OneDot()
        defaultDotValues(self.sd1ds)
        Vmatrix = np.matrix([[.23, 1, .23, 300.], [0, 0, 0, 1]])
        self.gate_transform_sd1 = GateTransform(Vmatrix, ['SD1a', 'SD1b', 'SD1c'], ['det1'])

        # super().__init__(name=name)

    def _dummy_get(self, param):
        return 0

    def _dummy_set(self, param, value):
        pass

    def _data_get(self, param):
        return self._data.get(param, 0)

    def _data_set(self, param, value):
        self._data[param] = value
        return

    def gate2ivvi(self, g):
        i, j = self.gate_map[g]
        return 'ivvi%d' % (i + 1), 'dac%d' % j

    def gate2ivvi_value(self, g):
        i, j = self.gate2ivvi(g)
        # value= self._data[i].get(j, 0)
        value = self._data.get(i + '_' + j, 0)
        return value

    def get_gate(self, g):
        return self.gate2ivvi_value(g)

    def _calculate_pinchoff(self, gates, offset=-200., random=0):
        c = 1
        for jj, g in enumerate(gates):
            v = self.gate2ivvi_value(g)
            # i, j = self.gate2ivvi(g)
            # v = float( self._data[i].get(j, 0) )
            c = c * qtt.algorithms.functions.logistic(v, offset + jj * 5, 1 / 40.)
        val = c
        if random:
            val = c + (np.random.rand() - .5) * random
        return val

    def computeSD(self, usediag=True, verbose=0):
        logger.debug('start SD computation')

        # main contribution
        val1 = self._calculate_pinchoff(['SD1a', 'SD1b', 'SD1c'], offset=-150, random=self.sdrandom)
        val2 = self._calculate_pinchoff(['SD2a', 'SD2b', 'SD2c'], offset=-150, random=self.sdrandom)

        val1x = self._calculate_pinchoff(['SD1a', 'SD1c'], offset=-50, random=0)

        # coulomb system for dot1
        ds = self.sd1ds
        gate_transform_sd1 = self.gate_transform_sd1
        gv = [float(self.get_gate(g)) for g in gate_transform_sd1.sourcenames]
        setDotSystem(ds, gate_transform_sd1, gv)
        ds.makeHsparse()
        ds.solveH(usediag=True)
        _ = ds.findcurrentoccupancy()
        cond1 = .75 * dotConductance(ds, index=0, T=3)

        cond1 = cond1 * np.prod((1 - val1x))
        if verbose >= 2:
            print('k1 %f, cond %f' % (k1, cond))

        # contribution of charge from bottom dots
        gv = [self.get_gate(g) for g in self.sourcenames]
        tv = self.gate_transform.transformGateScan(gv)
        ds = self.ds
        for k, val in tv.items():
            if verbose:
                print('computeSD: %d, %f' % (k, val))
            setattr(ds, k, val)
        ds.makeHsparse()
        ds.solveH(usediag=usediag)
        ret = ds.OCC

        sd1 = (1 / np.sum(self.sddist1)) * (ret * self.sddist1).sum()
        sd2 = (1 / np.sum(self.sddist1)) * (ret * self.sddist2).sum()

        return [val1 + sd1 + cond1, val2 + sd2]

    def compute(self, biasrandom=None):
        ''' Compute output of the model '''

        if biasrandom is None:
            biasrandom = self.biasrandom
        try:
            logger.debug('DummyModel: compute values')
            # current through keithley1, 2 and 3

            # v = float(self._data['ivvi1']['c11'])
            c = 1
            if 1:
                c *= self._calculate_pinchoff(['P1', 'P2', 'P3', 'P4'], offset=-200.)
                c *= self._calculate_pinchoff(['D1', 'D2', 'D3', 'L', 'R'], offset=-150.)
            instrument = 'keithley3'
            if not instrument in self._data:
                self._data[instrument] = dict()
            val = c + self.biasrandom * (np.random.rand() - .5)
            logging.debug('compute: value %f' % val)

            self._data[instrument]['amplitude'] = val

        except Exception as ex:
            print(ex)
            logging.warning(ex)
            # raise ex
            # msg = traceback.format_exc(ex)
            # logging.warning('compute failed! %s' % msg)
        return c

    def _generic_get(self, instrument, parameter):
        if not parameter in self._data[instrument]:
            self._data[instrument][parameter] = 0
        return self._data[instrument][parameter]

    def _generic_set(self, instrument, parameter, value):
        logging.debug('_generic_set: param %s, val %s' % (parameter, value))
        self._data[instrument][parameter] = float(value)

    def keithley1_get(self, param):
        self.lock.acquire()

        sd1, sd2 = self.computeSD()
        self._data['keithley1']['amplitude'] = sd1
        self._data['keithley2']['amplitude'] = sd2
        # print('keithley1_get: %f %f' % (sd1, sd2))
        val = self._generic_get('keithley1', param)
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
        val = self._generic_get('keithley3', param)
        self.lock.release()
        return val

   
