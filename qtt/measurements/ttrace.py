# -*- coding: utf-8 -*-
""" Code for creating and parsting t-traces

@author: eendebakpt
"""

#%% Load packages
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pyqtgraph as pg

import qtt
#import qtt.measurement.scans

from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import pulsar as ps

from pycqed.measurement.waveform_control.sequence import Sequence
from pycqed.measurement.waveform_control import element

#%% Virtual


def trace_read_virtual(ttraces, gates):
    """ Simulation of trace read """
    data_raw = []
    for ii, ttrace_element in enumerate(ttraces):
        tt, vv = ttrace_element.waveforms()
        kk = vv.keys()
        for k in kk:
            data_raw += [vv[k]]

    data_raw = np.array(data_raw).sum(axis=0)
    ww = np.linspace(0, 100, (data_raw.size))
    data_raw += .03 * np.sin(ww + gates.R.get())
    data_raw += .03 * np.cos(ww + gates.L.get())
    data_raw += .032 * (2 * np.random.rand(*data_raw.shape) - 1)**9
    return data_raw

#%%


sq_pulse = pulse.SquarePulse(channel='ch1', name='A square pulse')
sq_pulse_marker = pulse.SquarePulse(
    channel='ch1_marker1', name='A square pulse on MW pmod')
lin_pulse = pulse.LinearPulse(channel='ch1', name='Linear pulse')

def create_virtual_matrix_dict(cc_basis, physical_gates, c, verbose=1):
    """ Converts the virtual gate matrix into a virtual gate mapping
    Inputs:
        physical_gates (list): containing all the physical gates in the setup
        cc_basis (list): containing all the virtual gates in the setup
        c (array): virtual gate matrix
    Outputs: 
        virtual_matrix (dict): dictionary, mapping of the virtual gates"""
    virtual_matrix = OrderedDict()                                                                                            
    for ii,k in enumerate(cc_basis):
        if verbose:
            print('create_virtual_matrix_dict: adding %s ' % (k,))
        tmp=OrderedDict( zip(physical_gates, c[ii,:] ) )   
        tmp[ physical_gates[ii]] = 1
        virtual_matrix[k] = tmp
    return virtual_matrix

def create_virtual_matrix_dict_inv(cc_basis, physical_gates, c, verbose=1):
    """ Converts the virtual gate matrix into a virtual gate mapping needed for the ttraces
    Inputs:
        physical_gates (list): containing all the physical gates in the setup
        cc_basis (list): containing all the virtual gates in the setup
        c (array): virtual gate matrix
    Outputs: 
        virtual_matrix (dict): dictionary, mapping of the virtual gates needed for the ttraces """
    invc=np.linalg.inv(c)                                                                                    
    virtual_matrix = OrderedDict()                                                                                            
    for ii,k in enumerate(cc_basis):
        if verbose:
            print('create_virtual_matrix_dict: adding %s ' % (k,))
        tmp=OrderedDict( zip(physical_gates, invc[:,ii] ) )    #changed to test!!
        virtual_matrix[k] = tmp
    return virtual_matrix


def create_ttrace(ttrace, pulsars, name='ttrace', verbose=1, awg_map=None, markeridx=1):
    """ Create a Toivo trace

    Args:
        ttrace (dict)
        pulsars (list): list of Pulsar objects
        markeridx (int): index of Pular to use for marker 

    """

    fillperiod = ttrace['fillperiod']
    period = ttrace['period']
    alpha = ttrace['alpha']
    fpga_delay = ttrace['fpga_delay']
    awg_delay = ttrace['awg_delay']
    period0 = ttrace.get('period0', None)
    if period0 is None:
        period0 = fillperiod
    traces = ttrace['traces']
    ntraces = len(traces)

    ttraces = []
    # start with empty space
    for pi, pulsar in enumerate(pulsars):
        ttrace_element = element.Element(name, pulsar=pulsar)
        ttraces += [ttrace_element]

    for pi, ttrace_element in enumerate(ttraces):
        add_fill(ttrace_element, fillperiod=period0,
                 channels='all', tag='fillx', verbose=0)

    pulsar = pulsars[markeridx]
    ch = 1
    lp = lastpulse(ttrace_element)
    # ttrace_element.pulses[lp].effective_stop()
    endtime = lasttime(ttrace_element)

    lpm = lastpulse(ttrace_element)
    # add marker
    markerperiod = ttrace['markerperiod']

    pi, ci, mi = awg_map['fpga_mk']
    pulsar = pulsars[pi]
    ttrace_element = ttraces[pi]
    ttrace_element.add(pulse.cp(sq_pulse, amplitude=.1, length=markerperiod, channel='ch%d_marker%d' % (ci, mi), channels=[]),
                       name='marker%d' % ch, start=fpga_delay, refpulse=None, refpoint='end')  # refpulse=refpulse

    if verbose:
        print('create_ttrace: %d traces, %d pulsars' % (ntraces, len(ttraces)))

    print('time after first till: %e' % endtime)
    ttrace['tracedata'] = []
    for ii, tt in enumerate(traces):
        pass
        gg = [ga[0] for ga in tt]
        if verbose:
            print('trace %d: gates %s' % (ii, gg))
        start_time = endtime

        ttrace['tracedata'] += [{'start_time': start_time + alpha *
                                 period, 'end_time': start_time + (1 - alpha) * period}]

        for g, a in tt:
            if isinstance(g, int):
                ch = g
                ci = g
                pi = 0
            else:
                pi, ci = awg_map[g]
                ch = ci
            R = a
            print('  pi %d: channel %s: amplitude %.2f (%s)' % (pi, ch, R, g))
            #lp = lastpulse(filler_element)
            start_time = endtime
            ttrace_element = ttraces[pi]
            if 1:
                tag = 'trace%dch%d' % (ii, ch)
                print('tag %s, start_time %f' % (tag, start_time))
                ttrace_element.add(pulse.cp(lin_pulse, amplitude=.2, start_value=0, end_value=-R, length=alpha * period, channel='ch%d' % ch),
                                   name=tag + 'a', start=start_time + 0 * 1e-6, refpulse=None, refpoint='end')  # refpulse=refpulse
                ttrace_element.add(pulse.cp(lin_pulse, start_value=-R, end_value=R, length=(1 - 2 * alpha) * period, channel='ch%d' % ch),
                                   name=tag + 'b', refpulse=tag + 'a', refpoint='end')  # refpulse=refpulse
                ttrace_element.add(pulse.cp(lin_pulse, start_value=R, end_value=0, length=alpha * period, channel='ch%d' % ch),
                                   name=tag + 'c', refpoint='end', refpulse=tag + 'b')  # refpulse=refpulse

            # endtime
        lp = lastpulse(ttrace_element)
        add_fill(ttrace_element, refpulse=lp, tag='fill%d' %
                 ii, fillperiod=fillperiod, refpoint='end')
        lp = lastpulse(ttrace_element)
        endtime = lasttime(ttrace_element)  # endtime

    startx = lasttime(ttrace_element)
    for pi, ttrace_element in enumerate(ttraces):
        add_fill(ttrace_element, fillperiod=period0,
                 start=startx, tag='lastfill')

    if 'awg_mk' in awg_map:
        pi, ci, mi = awg_map['awg_mk']
        pulsar = pulsars[pi]
        ttrace_element = ttraces[pi]
        ttrace_element.add(pulse.cp(sq_pulse, amplitude=.1, length=markerperiod - awg_delay, channel='ch%d_marker%d' % (ci, mi), channels=[]),
                           name='awgmarker%dpost' % ci, start=0, refpulse=None, refpoint='end')  # refpulse=refpulse
        ttrace_element.add(pulse.cp(sq_pulse, amplitude=.1, length=awg_delay, channel='ch%d_marker%d' % (ci, mi), channels=[]),
                           name='awgmarker%dpre' % ci, start=lasttime(ttrace_element) - awg_delay, refpulse=None, refpoint='end')  # refpulse=refpulse

    return ttraces, ttrace

#%%


def set_awg_trace(vawg, awgs, clock=10e6):
    """  Set the awg in correct operation mode for the ttraces """
    vawg.AWG_clock = clock
    vawg.awg.AWG_clock=clock
    for a in awgs:
        a.clock_freq(clock)

def define_awg5014_channels(pulsar, marker1highs=.25):
    """ Helper function """
    if isinstance(marker1highs, (int, float)):
        marker1highs = [marker1highs] * 4
    for i in range(4):
        # Note that these are default parameters and should be kept so.
        # the channel offset is set in the AWG itself. For now the amplitude is
        # hardcoded. You can set it by hand but this will make the value in the
        # sequencer different.
        pulsar.define_channel(id='ch{}'.format(i + 1),
                              name='ch{}'.format(i + 1), type='analog',
                              # max safe IQ voltage
                              high=2.0, low=-2.0,
                              offset=0.0, delay=0, active=True)
        pulsar.define_channel(id='ch{}_marker1'.format(i + 1),
                              name='ch{}_marker1'.format(i + 1),
                              type='marker',
                              high=marker1highs[i], low=0, offset=0.,
                              delay=0, active=True)
        pulsar.define_channel(id='ch{}_marker2'.format(i + 1),
                              name='ch{}_marker2'.format(i + 1),
                              type='marker',
                              high=2.6, low=0, offset=0.,
                              delay=0, active=True)


def lastpulse(filler_element):
    """ Return last pulse from a sequence """
    keys = list(filler_element.pulses.keys())
    tt = [filler_element.pulses[k].effective_stop() for k in keys]
    idx = np.argmax(tt)
    return keys[idx]


def lasttime(filler_element):
    """ Return stop time of last pulse from a sequence """
    keys = list(filler_element.pulses.keys())
    tt = [filler_element.pulses[k].effective_stop() for k in keys]
    idx = np.argmax(tt)
    return tt[idx]


def add_fill(awg_element, tag, channels=None, refpulse=None, fillperiod=1e-7, start=0, refpoint='start', verbose=0):
    """ Add filling period to an element

    Args:
        awgelement (element):
        tag (str): name for the pulses to use
        ...

    """

    if channels is None:
        # just select the first channel
        channels = [list(awg_element.pulsar.channels.keys())[0]]
    if channels == 'all':
        channels = list(awg_element.pulsar.channels.keys())
    sq_pulse = pulse.SquarePulse(
        channel=channels[0], name='A dummy square pulse')
    for ii, ch in enumerate(channels):
        R = 0
        px = pulse.cp(sq_pulse, amplitude=R, length=fillperiod,
                      channel=ch, channels=[])
        name = str(tag) + '%s' % ch
        awg_element.add(px,
                        name=name, start=start, refpulse=refpulse, refpoint=refpoint)
        if verbose:
            print('add_fill: ch %s: name %s: amplitude %s' %
                  (ch, name, px.amplitude))
        if refpulse is None:
            refpulse = name


def show_element(elmnt, fig=100, keys=None, label_map=None):
    """ Show pycqed waveform element 
    
    Args:
        elmnt ()
    """
    tt, xx = elmnt.waveforms()
    if fig is not None:
        qtt.pgeometry.cfigure(fig)
        plt.clf()

        if keys is None:
            keys = sorted(xx.keys())
        for k in keys:
            v = xx[k]

            if label_map is None:
                label = k
            else:
                label = label_map[k]
            plt.plot(1e3 * tt, v, '.', label=label)
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal')

#%%


import pyqtgraph as pg
import qtpy.QtWidgets as QtWidgets
import qtpy.QtCore as QtCore
import qtt


class MultiTracePlot:

    def __init__(self, nplots, ncurves=1, title='Multi trace plot'):
        self.title = title
        self.verbose = 1

        plotwin = pg.GraphicsWindow(title=title)
        self.plotwin = plotwin

        win = QtWidgets.QWidget()
        win.show()
        win.setWindowTitle(self.title)
        win.resize(800, 600)
        self.win = win
        topLayout = QtWidgets.QHBoxLayout()
        win.start_button = QtWidgets.QPushButton('Start')
        win.stop_button = QtWidgets.QPushButton('Stop')

        for b in [win.start_button, win.stop_button]:
            b.setMaximumHeight(24)

        topLayout.addWidget(win.start_button)
        topLayout.addWidget(win.stop_button)

        vertLayout = QtWidgets.QVBoxLayout()

        vertLayout.addLayout(topLayout)
        vertLayout.addWidget(plotwin)

        win.setLayout(vertLayout)

        self.nx = int(np.ceil(np.sqrt(nplots)))
        self.ny = int(np.ceil((nplots / self.nx)))

        # Enable antialiasing for prettier plots
        # pg.setConfigOptions(antialias=True)

        self.plots = []
        self.curves = []
        pens = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)] * 2

        for ii in range(self.ny):
            for ix in range(self.nx):
                p = plotwin.addPlot()
                self.plots.append(p)
                cc = []
                for ii in range(ncurves):
                    c = p.plot(pen=pens[ii])
                    cc += [c]
                self.curves.append(cc)
            plotwin.nextRow()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._updatefunction)

        def connect_slot(target):
            """ Create a slot by dropping signal arguments """
            #@Slot()
            def signal_drop_arguments(*args, **kwargs):
                #print('call %s' % target)
                target()
            return signal_drop_arguments

        win.start_button.clicked.connect(connect_slot(self.startreadout))
        win.stop_button.clicked.connect(connect_slot(self.stopreadout))

    def _updatefunction(self):
        self.updatefunction()

    def updatefunction(self):
        qtt.pgeometry.tprint('updatefunction: dummy...', dt=10)
        pass

    def startreadout(self, callback=None, rate=1000, maxidx=None):
        if maxidx is not None:
            self.maxidx = maxidx
        if callback is not None:
            self.updatefunction = callback
        self.timer.start(1000 * (1. / rate))
        if self.verbose:
            print('MultiTracePlot: start readout: rate %.1f Hz' % rate)
        self.win.setWindowTitle(self.title + ': started')

    def stopreadout(self):
        if self.verbose:
            print('MultiTracePlot: stop readout')
        self.timer.stop()
        self.win.setWindowTitle(self.title + ': stopped')
        
#%%

class ttrace_t(dict):#TODO: check definitions of markerperiod and alpha
    """ Structure that contains information about ttraces
    
        
    Fields:
        period (float): the time of the trace for each dot
        markerperiod (float):  ?
        fillperiod (float): the time it takes to come to the start voltage of the relevant signal end to go back to the initial value afterwards
        period0:time before the trace sequence starts
        alpha (float):?
        fpga_delay (float): delay time between the actual signal and the readout of the FPGA 
        fpgafreq: readout frequency of the FPGA
        awgclock: clock frequency of the AWG
        traces: contains the extrema the traces have to have
        ....           
    ...
    
    """
           
#%% TODO: try to merge ttrace and ttraces 
def activate_ttraces(station, location, amplitudes, virt_map_for_traces, vgates,pgates,awgclock=10e6):   
    """Activates orthogonal 1D sweeps in each dimension shortly after each other according to the virtual gate map
    Inputs:
        station: containing at least the clock frequency and mapping of the AWG and the readout frequency of the FPGA
        location: location of the setup; 3dot,4dot,8dot or vdot
        amplitudes: amplitudes of the ttraces
        virt_map_for_traces: mapping of the virtual gate instrument according to the function create_virtual_matrix_dict_inv
        vgates: virtual gates corresponding to the chemical potentials
        pgates: plunger gates
        awgclock (default=10e6): clock frequency the awg needs to have for ttrace operation (typically different than that needed for scan1Dfast etc.
    Outputs:
        ttraces,ttrace: containing information about the ttraces put on the AWG"""
    #"""Define the pulsar object, the pulsar is the final object containing the traces which is put on the AWG"""
    pulsar_objects =[]    
    for ii,a in enumerate(station.awg._awgs):
        print('creating Pulsar %d' % ii)
        a.clock_freq.set(awgclock)
        p = ps.Pulsar()
        p.clock = awgclock
        setattr(station, 'pulsar%d' % ii, p)
        p.AWG=a        
        define_awg5014_channels(p, marker1highs=2.6)    
        pulsar_objects+=[p]
      
    #"""Define amplitudes and frequencies of Toivo traces according to the given virtual gate map"""  
    ttrace = ttrace_t({'markerperiod': 80e-6, 'fillperiod': 100e-6, 'period': 500e-6, 'alpha': .1})
    ttrace['period0']=250e-6
    ttrace['fpga_delay']=2e-6
    ttrace['traces'] = []; ttrace['traces_volt'] = []
    ttrace['fpgafreq']=station.fpga.sampling_frequency()
    ttrace['awgclock']=awgclock
    
    
    if location=='3dot':
        hw=station.hardware3dot#for now changed!!
        awg_to_plunger_plungers=dict( [('P1', hw.awg_to_P1()), ('P2', hw.awg_to_P2()), ('P3',hw.awg_to_P3() )])
    else:
        awg_to_plunger_plungers=dict( [('P1', 103), ('P2', 100), ('P3',102 ), ('P4', 104)])
    
    #"""Map them onto the traces itself"""
    for ii, v in enumerate(vgates):
        R= amplitudes[ii]
        print('gate %s: amplitude %.2f [mV]' % (v, R, ))   
        q=virt_map_for_traces[v] #replaced vg
        print(q)        
        w = [(k, R*q[k]/awg_to_plunger_plungers[k]) for k in pgates]
        wvolt = [(k, R*q[k]) for k in pgates]
        ttrace['traces'] += [w]
        ttrace['traces_volt'] += [wvolt]
    if location=='vdot':
        ttrace['traces'] = []
        w=[('P1', 60/awg_to_plunger_plungers['P1'])]
        ttrace['traces'] += [w]
        w=[('P3', 50/awg_to_plunger_plungers['P3'])]
        ttrace['traces'] += [w]
        w=[('P4', 50/awg_to_plunger_plungers['P4'])]
        ttrace['traces'] += [w]
                   
    #"""Create the ttrace waveforms"""     
    ttrace['awg_delay'] = 0e-4+2e-5 
    awg_map=station.awg.awg_map
    markeridx=awg_map['fpga_mk'][0]
    ttraces, ttrace = create_ttrace(ttrace, name='ttrace', pulsars=pulsar_objects, awg_map=awg_map, markeridx=markeridx)
    ttrace_element = ttraces[0]
    print('waveform: %d elements' % ttrace_element.waveforms()[0].size)
           
    #"""Put ttraces on the pulsars of the awg"""   
    for ii, t in enumerate(ttraces):
        seq = Sequence('8dot_sequence_awg%d' % ii)
        seq.append(name='toivotrace', wfname=t.name, trigger_wait=False,)    
        elts = [t]
        # program the Sequence
        pulsar = pulsar_objects[ii]
        pulsar.program_awg(seq, *elts)
    
      
    #"""Really run the awg"""
    awgs=station.awg._awgs
    set_awg_trace(station,awgs, awgclock)
    for awg in awgs:
        self = awg
        if 1:
            #TODO: needed?
            v = self.write('SOUR1:ROSC:SOUR INT')
            v = self.ask('SOUR1:ROSC:SOUR?')
            print('%s: %s' % (awg, v))    
        awg.run()
    print('ttraces running')
    return(ttraces,ttrace)
    

def plot_ttraces(ttraces): 
    """Plots the ttraces which are put on the AWG
    Inputs:
        ttraces: information of the ttraces put on the AWG"""
    for ii, ttrace_element in enumerate(ttraces):
        v = ttrace_element.waveforms()
        kk = v[1].keys()
        kkx = [k for k in kk if np.any(v[1][k])]  
        show_element(ttrace_element, fig=100 + ii, keys=kkx, label_map=None)
        plt.legend(numpoints=1)
       
      
def dummy_read(station, idx=[1,],Naverage=26):#TODO: what does idx exactly mean?
    """Reads the raw data
    Inputs: 
        station: station at leas containing the FPGA
        idx: indexes of channels used
        Naverage: averaging filter over the readout function
    Outputs: 
        data_raw: the raw readout data ="""
    ReadDevice = ['FPGA_ch%d' % c for c in idx ]
    qq=station.fpga.readFPGA(ReadDevice=ReadDevice, Naverage=Naverage)
    data_raw = np.array([qq[i] for i in idx])
    return data_raw    

def parse_data(data_raw, ttraces,ttrace, verbose=1): #TODO: definition of datax and tx, try tho put it in the ttrace class
    """Read the data, split them in the different dimension sweeps
    Inputs:
        data_raw: the raw readout data
        ttraces,ttrace: information of the ttraces put on the AWG in order to now how to split the data
    Outputs:
        tt: containing information of the timing of the function
        datax:
        tx: the actual signal which is can be used for further purposes """
    fpgafreq=ttrace['fpgafreq']
    
    ttrace_element= ttraces[0]
    tracedata=ttrace['tracedata']
    ttotal = ttrace_element.waveforms()[0].size / ttrace['awgclock']         
    qq=ttotal*fpgafreq

    datax = data_raw.copy()
    datax[:,0] = np.mean(datax[:,1:2], axis=1)
    
    ff=data_raw.shape[1]/qq                
    fpgafreqx=fpgafreq*ff
    tt=np.arange(0, datax.shape[1])/fpgafreqx
    tx=[]
    if tracedata is not None:
        for x in tracedata:
            sidx=int(x['start_time']*fpgafreqx)
            eidx=int(x['end_time']*fpgafreqx)
            if verbose>=2:
                print('sidx %s, eidx %s'  % (sidx, eidx))
            tx+=[ datax[:, sidx:eidx]]
    return tt, datax, tx

def show_data(tt,tx, data_raw, ttrace, tf=1e3, fig=10):#TODO: diminish the amount of input arguments
    """Plot the raw data and the parsed data of the resulting signal
    Inputs:
        tt: parsed data including timing
        tx: the actual signal 
        data_raw: raw readout data
        ttrace: data about the traces put on the AWG"""
    plt.figure(fig)
    plt.clf()
    for i in range(data_raw.shape[0]):
        plt.plot(tf*tt, data_raw[i], '.')
    if tf==1e3:
        plt.xlabel('Time [ms]')
    else:
        plt.xlabel('Time')
    for ii, x in enumerate(ttrace['tracedata']):
        s = x['start_time'] * tf # fpgafreq
        e = x['end_time'] * tf # fpgafreq
        qtt.pgeometry.plot2Dline([-1, 0, s], '--')
        qtt.pgeometry.plot2Dline([-1, 0, e], ':')
    plt.figure(400); plt.clf()
    nx=int(np.ceil(np.sqrt(len(tx))))
    ny=int(np.ceil(len(tx)/nx))
    for ii, q in enumerate(tx):
        plt.subplot(nx,ny,ii+1)
        plt.plot(q.T)