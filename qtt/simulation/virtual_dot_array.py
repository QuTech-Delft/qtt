""" Virtual version of a linear dot array

The system consists of:

- a linear array of dots
- a magic top gate that always works
- 2 barriers gates and 1 plunger gate for each dot
- a sensing dot that always works

There are virtual instruments for 

- DACs: several virtual IVVIs
- Virtual Keithleys (1 and 2 for the SDs, 4 for the ohmic)
- A virtual gates object
 """


#%% Load packages

import sys
import os
import logging
import warnings
import threading
import logging

import qcodes
import numpy as np

from qtt.instrument_drivers.virtual_instruments import VirtualMeter, VirtualIVVI
from qtt.instrument_drivers.gates import VirtualDAC as virtual_gates
from qtt.utilities.debug import dumpstring

from qcodes import Instrument
from functools import partial

import qtt
from qtt.simulation.dotsystem import DotSystem, GateTransform
from qtt.simulation.dotsystem import defaultDotValues

from qtt.simulation.dotsystem import OneDot
from qtt.simulation.classicaldotsystem import DoubleDot, TripleDot, MultiDot
from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer, SimulationAWG

logger=logging.getLogger(__name__)
qtt.check_version('0.24', qtt)


#%% Data for the model

class virtualdot_t:
    
    def __init__(self, ndots):
        self.ndots = ndots
        raise Exception('do not use this class any more')
        
    def ndots(self):
        """ Return number of dots in the sample """
        return self.ndots

def gate_settle(gate):
    """ Return gate settle times """

    return 0 # the virtual gates have no latency


def gate_boundaries(gate_map):
    """ Return gate boundaries 
    
    Args:
        gate_map (dict)
    Returns:
        gate_boundaries (dict)
    """
    gate_boundaries = {}
    for g in gate_map:
        if 'bias' in g:
                gate_boundaries[g]=(-900, 900)
        elif 'SD' in g: 
                gate_boundaries[g]=(-1000, 1000)
        elif 'O' in g: 
            # ohmics
            gate_boundaries[g]=(-4000, 4000)
        else:
                gate_boundaries[g]=(-1000, 800)
        
    return gate_boundaries

def generate_configuration(ndots):
    """ Generate configuration for a standard linear dot array sample
    
    Args:
        ndots (int): number of dots
    Returns:
        nr_ivvi (int)
        gate_map (dict)
        gates (list)
        bottomgates (list)
    """
    bottomgates=[]
    for ii in range(ndots+1): 
        if ii>0:
           bottomgates+=['P%d' % ii]
        bottomgates+=['B%d' % ii]

    sdgates=[]
    for ii in [1]:
        sdgates+=['SD%da' % ii, 'SD%db' % ii, 'SD%dc' % ii]
    gates = ['D0']+bottomgates+sdgates
    gates+=['bias_1', 'bias_2']
    gates+=['O1', 'O2','O3', 'O4', 'O5']
    
    nr_ivvi = int(np.ceil(len(gates) /14)       )
    gate_map = {}
    for ii, g in enumerate(sorted(gates)):
        i = int(np.floor(ii/16))
        d = ii%16
        gate_map[g]= (i, d+1)
        
    return nr_ivvi, gate_map, gates, bottomgates

if 0:
    nr_ivvi, gate_map, gates, bottomgates = generate_configuration(6)
    print(gate_map)
    print(gates)
    print(bottomgates)



#%%
class DotModel(Instrument):
    """ Simulation model for linear dot array
    
    The model is intended for testing the code and learning. It does _not simulate any meaningfull physics. 
    
    """

    def __init__(self, name, verbose=0, nr_dots=3, maxelectrons=2, sdplunger=None, **kwargs):
        """ The model is for a linear arrays of dots with a single sensing dot
        
            Args:
                name (str): name for the instrument
                verbose (int)
                nr_dots (int)
                maxelectrons (int)
                
        """
        
        super().__init__(name, **kwargs)
        logging.info('DotModel.__init__: start')

        nr_ivvi, gate_map, gates, bottomgates = generate_configuration(nr_dots)

        self.nr_ivvi = nr_ivvi
        self.gate_map = gate_map
        
        self._sdplunger = sdplunger
        
        # dictionary to hold the data of the model
        self._data = dict()
        self.lock = threading.Lock()

        self.sdnoise = .001 # noise for the sensing dot

        # make parameters for all the gates...
        gate_map = self.gate_map
        gateset = set(gate_map.values())

        self.gates = list(gate_map.keys())

        self.bottomgates = bottomgates 

        self.gate_pinchoff = -200
        
        if verbose>=1:
            print('DotModel: ndots %s, bottomgates %s' % (nr_dots, bottomgates))

        gateset = [(i, a) for a in range(1, 17) for i in range(nr_ivvi)]
        for i, idx in gateset:
            g = 'ivvi%d_dac%d' % (i + 1, idx)
            logging.debug('add gate %s' % g)
            self.add_parameter(g,
                               label='Gate {} (mV)'.format(g),
                               get_cmd=partial(self._data_get, g),
                               set_cmd=partial(self._data_set, g),
                               )

        # make entries for keithleys
        for instr in ['keithley1', 'keithley2', 'keithley3', 'keithley4']:
            if not instr in self._data:
                self._data[instr] = dict()
            g = instr + '_amplitude'
            self.add_parameter(g,
                               label='Amplitude (pA)'.format(g),
                               get_cmd=partial(getattr(self, instr + '_get'), 'amplitude'),
                               )

        # initialize the actual dot system
        if nr_dots==1:
            self.ds = OneDot(maxelectrons=maxelectrons)
            self.ds.alpha=np.eye(self.ds.ndots)
            self.sourcenames = bottomgates                
        elif nr_dots==2:
            self.ds = DoubleDot(maxelectrons=maxelectrons)
            self.ds.alpha=np.eye(self.ds.ndots)
            self.sourcenames = bottomgates
        elif nr_dots==3:
            self.ds = TripleDot(maxelectrons=maxelectrons)
            self.ds.alpha=np.eye(self.ds.ndots)
            self.sourcenames = bottomgates
        elif nr_dots==6 or nr_dots==4 or  nr_dots==5:
            self.ds = MultiDot(name='dotmodel', ndots=nr_dots, maxelectrons=maxelectrons)
            if verbose:
                print('ndots: %s maxelectrons %s' % (self.ds.ndots, maxelectrons))
                print('self.ds.coulomb_energy.shape %s' % ( self.ds.coulomb_energy.shape, ) )
            self.ds.alpha=np.eye(self.ds.ndots)
            self.sourcenames = bottomgates
        else:
            raise Exception('not implemented yet...')
        self.targetnames = ['det%d' % (i + 1) for i in range(self.ds.ndots)]

        Vmatrix = np.zeros((len(self.targetnames) + 1, len(self.sourcenames) + 1))

        Vmatrix[-1, -1] = 1
        if verbose>=1:
            print('ndots? %s' % self.ds.ndots)
            print('targetnames: %s' % self.targetnames)
        for ii in range(self.ds.ndots):
            ns = len(self.sourcenames)
            v = np.arange(ns) - (1 + 2 * ii)
            v = 1 / (1 + .08 * np.abs(v)**3)
            Vmatrix[ii, 0:ns] = v
            # compensate for the barriers
            Vmatrix[0:self.ds.ndots, -1] = (Vmatrix[ii,[2*ii, 2*ii+2 ] ].sum() ) * -self.gate_pinchoff
        self.gate_transform = GateTransform(Vmatrix, self.sourcenames, self.targetnames)

        self.sddist1 = self.ds.ndots / (1 + np.arange(self.ds.ndots))
        self.sddist2 = self.sddist1[::-1]
        
        if isinstance(self.ds, qtt.simulation.dotsystem.DoubleDot):
            for ii in range(self.ds.ndots):
                setattr(self.ds, 'osC%d' % (ii + 1), 55)
            for ii in range(self.ds.ndots - 1):
                setattr(self.ds, 'isC%d' % (ii + 1), 3)
        else:
            pass

    def get_idn(self):
        ''' Overrule because the default get_idn yields a warning '''
        IDN = {'vendor': 'QuTech', 'model': self.name,
               'serial': None, 'firmware': None}
        return IDN
    
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
        value = self._data.get(i + '_' + j, 0)
        return value

    def get_gate(self, g):
        return self.gate2ivvi_value(g)

    def _calculate_pinchoff(self, gates, offset=-200., random=0):
        """ Calculate current due to pinchoff of a couple of gates """
        c = 1
        for jj, g in enumerate(gates):
            v = self.gate2ivvi_value(g)
            c = c * qtt.algorithms.functions.logistic(v, offset + jj * 5, 1 / 40.)
        val = c
        if random:
            val = c + (np.random.rand() - .5) * random
        return val

    def computeSD(self, usediag=True, verbose=0):
        logging.debug('start SD computation')

        # contribution of charge from bottom dots
        gv = [self.get_gate(g) for g in self.sourcenames]
        tv = self.gate_transform.transformGateScan(gv)
        ds = self.ds

        if verbose:
                for k, val in tv.items():
                    print('computeSD: %s, %f' % (k, val))
                    setattr(ds, k, val)
        gatevalues = [float(tv[k]) for k in sorted(tv.keys())]
        ret = ds.calculate_ground_state(gatevalues)
        if 0:
            for k, val in tv.items():
                    setattr(ds, k, val)
            ds.makeHsparse()
            ds.solveH(usediag=usediag)
            ret = ds.OCC

        sd1 = (1 / np.sum(self.sddist1)) * (ret * self.sddist1).sum()
        sd2 = (1 / np.sum(self.sddist2)) * (ret * self.sddist2).sum()

        sd1 += self.sdnoise * (np.random.rand() - .5)
        sd2 += self.sdnoise * (np.random.rand() - .5)
        
        if self._sdplunger:
            val = self._calculate_pinchoff([self._sdplunger], offset=-100, random=.01)
            sd1 += val

        self.sd1 = sd1
        self.sd2 = sd2
            
        return sd1

    def compute(self, random=0.02):
        """ Compute output of the model """

        try:
            logging.debug('DummyModel: compute values')
            
            # current through 3
            val = self._calculate_pinchoff(self.bottomgates, offset=self.gate_pinchoff, random=.01)
            logging.debug('compute: value %f' % val)

            self._data['instrument_amplitude'] = val

        except Exception as ex:
            print(ex)
            logging.warning(ex)
            val = 0
        return val

    def keithley1_get(self, param):
        with self.lock:
            sd1 = self.computeSD()
            self._data['keithley1_amplitude'] = sd1
        return sd1

    def keithley2_get(self, param):
        return self.keithley1_get(param)

    def keithley4_get(self, param):
        with self.lock:
            k = 4e-3*self.get_gate('O1')
            self._data['keithley4_amplitude'] = k
        return k

    def keithley3_get(self, param):
        with self.lock:
            logging.debug('keithley3_get: %s' % param)
            val = self.compute()

            self._data['keithley3_amplitude'] = val
        return val

    def honeycomb(self, p1, p2, nx=50, ny=50, scanrange=300,  multiprocess=False):
        raise Exception('code not tested')
        test_dot = self.ds
        test2Dparams = np.zeros((test_dot.ngates, ny, nx))
        logging.info('honeycomb: %s %s' % (p1, p2))
        sweepx = self.get_gate(p1) + np.linspace(-scanrange, scanrange, nx)
        sweepy = self.get_gate(p2) + np.linspace(-scanrange, scanrange, ny)
        xv, yv = np.meshgrid(sweepx, sweepy)
        test2Dparams[0] = xv  # +yv
        test2Dparams[1] = yv

        # run the honeycomb simulation
        test_dot.simulate_honeycomb(test2Dparams, multiprocess=multiprocess, verbose=0)

        if self.sdnoise > 0:
            test_dot.honeycomb += self.sdnoise * (np.random.rand(*test_dot.honeycomb.shape) - .5)
        return test_dot.honeycomb


def close(verbose=1):
    """ Close all instruments """
    global station, model, _initialized

    if station is None:
        return

    for instr in station.components.keys():
        if verbose:
            print('close %s' % station.components[instr])
        try:
            station.components[instr].close()
        except:
            print('could not close instrument %s' % station.components[instr])

    _initialized = False
    
#%%
_initialized = False

# pointer to qcodes station
station = None

# pointer to model
model = None


def boundaries():
    global model
    if model is None:
        raise Exception('model has not been initialized yet')
    return gate_boundaries(model.gate_map)

#%%

def getStation():
    global station
    return station


def initialize(reinit=False, nr_dots=2, maxelectrons=2,
               verbose=2, start_manager=False):

    global station, _initialized, model

    logger.info('virtualDot: start')
    if verbose >= 2:
        print('initialize: create virtualdot')

    if _initialized:
        if reinit:
            close(verbose=verbose)
        else:
            return station
    logger.info('virtualDot: make DotModel')
    model = DotModel(name=qtt.measurements.scans.instrumentName('dotmodel'),
                     verbose=verbose>=3, nr_dots=nr_dots, 
                     maxelectrons=maxelectrons, sdplunger='SD1b')
    gate_map=model.gate_map
    if verbose >= 2:
        logger.info('initialize: DotModel created')
    ivvis = []
    for ii in range(model.nr_ivvi):
        ivvis.append(VirtualIVVI(name='ivvi%d' % (ii+1), model=model) )
    gates = virtual_gates(name='gates', gate_map=gate_map, instruments=ivvis)
    gates.set_boundaries(gate_boundaries(gate_map))

    logger.info('initialize: set values on gates')
    gates.set('D0', 101)
    for g in model.gates:
        gates.set(g, np.random.rand() - .5)

    # take into account the voltage divider on the ohmics
    for g in gates.parameters.keys():
        if g.startswith('O'):            
            gg=getattr(gates, g)
            gg.unit = 'uV'

    vawg = SimulationAWG(qtt.measurements.scans.instrumentName('vawg'))

    logger.info('initialize: create virtual keithley instruments')
    keithley1 = VirtualMeter('keithley1', model=model)
    keithley3 = VirtualMeter('keithley3', model=model)
    keithley4 = VirtualMeter('keithley4', model=model)

    digitizer = SimulationDigitizer(qtt.measurements.scans.instrumentName('sdigitizer'), model=model)
       
    logger.info('initialize: create station')
    station = qcodes.Station(gates, keithley1, keithley3, keithley4, *ivvis,
                         vawg, digitizer, update_snapshot= False)
    station.awg=station.vawg
    station.metadata['sample'] = 'virtual_dot'
    station.model = model

    station.gate_settle = gate_settle
    station.depletiongate_name = 'D0'
    station.bottomchannel_current = station.keithley3.amplitude

    station.jobmanager = None
    station.calib_master = None

    _initialized=True
    if verbose:
        print('initialized virtual dot system (%d dots)' % nr_dots)
    return station
  
#%%

from qtt.structures import onedot_t

def _getModel():
    return model

def bottomGates():
    return _getModel().bottomgates


def bottomBarrierGates():
    return _getModel().bottomgates[0::2]


def get_two_dots(full=1):
    """ return all posible simple two-dots """
    bg= bottomGates()
    two_dots = [dict({'gates':bg[0:3]+bg[2:5]})] # two dot case

    for td in two_dots:
        td['name'] = '-'.join(td['gates'])
    return two_dots


def get_one_dots(full=1, sdidx=[]):
    """ return all posible simple one-dots

    Each dot objects holds the gates, the name of the channel and the
    instrument measuring over the channel.

    """
    one_dots = []
    ki = 'keithley3.amplitude'
    
    bg= bottomGates()

    for ii in range(int((len(bg)-1)/2)) :    
        one_dots.append( onedot_t(gates= bg[2*ii:2*ii+3], transport_instrument= ki) )


    for x in sdidx:
        assert(x in [1,2,3])
        ki = 'keithley%d.amplitude' % x
        od = onedot_t(gates=['SD%d%s' % (x, l) for l in ['a', 'b', 'c']],transport_instrument=ki )
        one_dots.append(od)

            
    for od in one_dots:
        od['name'] = 'dot-%s' % ('-'.join(od['gates']))
    return one_dots

#%%


if __name__ == '__main__' and 1:
    import pdb    
    np.set_printoptions(precision=2, suppress=True)

    try:
        close()
    except:
        pass
            
    station = initialize(reinit=True, verbose=2)
    self=station.model
    model = station.model
    model.compute()
    model.computeSD()
    _=station.keithley1.amplitude()
    _=station.keithley4.amplitude()
    np.set_printoptions(suppress=True)
    print(model.gate_transform.Vmatrix)
