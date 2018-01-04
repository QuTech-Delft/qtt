# -*- coding: utf-8 -*-
""" Code for creating and parsting t-traces

@author: eendebakpt (houckm)
"""

#%% Load packages
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pyqtgraph as pg
import qtt
import warnings

import qtpy.QtWidgets as QtWidgets
import qtpy.QtCore as QtCore

import pycqed
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

def awg_info(awgs):
      """ Print information about awgs """
      for a in awgs:            
        print('awg %s'  % a.name)
        print('  clock %s MHz' % ( a.clock_freq()/1e6, ))
        print('  awg run_mode %s '  % (a.run_mode(),) )
        print('  awg trigger_mode %s '  % (a.trigger_mode(),     ) )
        print('  awg trigger sources %s'  % (a.trigger_source(),     ) )


sq_pulse = pulse.SquarePulse(channel='ch1', name='A square pulse')
sq_pulse_marker = pulse.SquarePulse(
    channel='ch1_marker1', name='A square pulse on MW pmod')
lin_pulse = pulse.LinearPulse(channel='ch1', name='Linear pulse')

def create_virtual_matrix_dict(virt_basis, physical_gates, c=None, verbose=1):
    """ Converts the virtual gate matrix into a virtual gate mapping
    Inputs:
        physical_gates (list): containing all the physical gates in the setup
        virt_basis (list): containing all the virtual gates in the setup
        c (array): virtual gate matrix
    Outputs: 
        virtual_matrix (dict): dictionary, mapping of the virtual gates"""
    virtual_matrix = OrderedDict()                                                                                            
    for ii, vname in enumerate(virt_basis):
        if verbose:
            print('create_virtual_matrix_dict: adding %s ' % (vname,))
        if c is None:
            v=np.zeros( len(physical_gates))
            v[ii]=1
        else:
            v=c[ii,:]
        tmp=OrderedDict( zip(physical_gates, v ) )           
        virtual_matrix[vname] = tmp
    return virtual_matrix

def create_virtual_matrix_dict_inv(cc_basis, physical_gates, c, verbose=1):
    """ Converts the virtual gate matrix into a virtual gate mapping needed for the ttraces
    Inputs:
        physical_gates (list): containing all the physical gates in the setup
        cc_basis (list): containing all the virtual gates in the setup
        c (array or None): inverse virtual gate matrix
    Outputs: 
        virtual_matrix (dict): dictionary, mapping of the virtual gates needed for the ttraces """

    if c is None:
        invc=None
    else:
        invc=np.linalg.inv(c)                                                                                    
    return create_virtual_matrix_dict(cc_basis, physical_gates, invc, verbose=1)


def show_ttrace_elements(ttrace_elements, fig=100):
    """ Show ttrace elements """
    for ii, ttrace_element in enumerate(ttrace_elements):
        v = ttrace_element.waveforms()
        
        c = list(v[0].keys())[0]
        print('waveform %d: channel %s: %d elements' % (ii, c, v[0][c].size) )
        kk = v[1].keys()
    
        kkx = [k for k in kk if np.any(v[1][k])]  # non-zero keys
    
        #pi = ii
        #label_map = [(t[1], k) for k, t in awg_map.items() if t[0] == pi]
    
        show_element(ttrace_element, fig=fig + ii, keys=kkx, label_map=None)
        plt.legend(numpoints=1)
        plt.title('ttrace element %s' % (ttrace_element.name,) )
    qtt.pgeometry.tilefigs(range(100,100+len(ttrace_elements)))

#%%


class ttrace_t(dict):
    """ Structure that contains information about ttraces
    
        
    Fields:
        period (float): the time of the trace for each dot
        markerperiod (float):  ?
        fillperiod (float): the time it takes to come to the start voltage of the relevant signal end to go back to the initial value afterwards
        period0:time before the trace sequence starts
        alpha (float): 
        fpga_delay (float): delay time between the actual signal and the readout of the FPGA 
        samplingfreq: readout frequency of the acquisition device
        awgclock: clock frequency of the AWG
        traces: contains the extrema the traces have to have
        ....           
    
    """
    
def create_ttrace(station, virtualgates, vgates, scanrange, sweepgates):
    """Define amplitudes and frequencies of Toivo traces according to the given virtual gate map"""  
    ttrace = ttrace_t({'markerperiod': 80e-6, 'fillperiod': 100e-6, 'period': 500e-6, 'alpha': .1})
    ttrace['period0']=250e-6
    ttrace['fpga_delay']=2e-6
    ttrace['traces'] = []; ttrace['traces_volt'] = []
    try:
        ttrace['samplingfreq']=station.fpga.sampling_frequency()
    except:
        try:
            ttrace['samplingfreq']=station.digitizer.sample_rate()
        except:
            warnings.warn('no fpga object available')    
    ttrace['awgclock']=station.awg.AWG_clock
    ttrace['awg_delay'] = 0e-4+2e-5 # ???

    
    map_inv=virtualgates.get_crosscap_map_inv()

    try:
        hw=station.hardware #for now hardcoded!!
        awg_to_plunger_plungers=dict( [ (g, getattr(hw, 'awg_to_%s' % g)() ) for g in sweepgates] )
    except Exception as ex:
        print(ex)
        warnings.warn('no hardware object available')    
        awg_to_plunger_plungers=dict( [ (g, 80)  for g in sweepgates] )
        
    pgates=sweepgates
    if isinstance(scanrange, (float, int)):
        scanrange=[scanrange]*len(vgates)
        
    #"""Map them onto the traces itself"""
    for ii, v in enumerate(vgates):
        R= scanrange[ii]
        print('gate %s: amplitude %.2f [mV]' % (v, R, ))   
        #q=virt_map_for_traces[v] #replaced vg
        #print(q)        
        w = [(k, R*map_inv[k][v]/awg_to_plunger_plungers[k]) for k in pgates]
        wvolt = [(k, R*map_inv[k][v]) for k in pgates]
        ttrace['traces'] += [w]
        ttrace['traces_volt'] += [wvolt]
    return ttrace

#%%

def read_trace_m4i(station, ttrace_elements, read_ch=[1], Naverage=60, verbose=0, fig=None):
    """ Read data from m4i device 
    
    TODO: merge with measuresegment function...
    """
    
    digitizer = station.digitizer
    if digitizer.sample_rate() == 0:
        raise Exception('error with digitizer')
    digitizer.sample_rate(10e6)
    #read_ch = [1]
    mV_range = 2000

    drate = digitizer.sample_rate()
    if drate == 0:
        raise Exception('sample rate of m4i is zero, please reset the digitizer')

    #ttotal = ttrace_elements[0].waveforms()[0].size / ttrace['awgclock']
    e = ttrace_elements[0]
    ttotal = e.ideal_length()

    # code for offsetting the data in software
    signal_delay = getattr(digitizer, 'signal_delay', None)
    if signal_delay is None:
        signal_delay = 0
    padding_offset = int(drate * signal_delay)

    period = ttotal

    paddingpix = 16
    padding = paddingpix / drate
    pretrigger_period = 16 / drate  # waveform['markerdelay'],  16 / samp_freq

    memsize = qtt.measurements.scans.select_digitizer_memsize(
        digitizer, period + 2 * padding, pretrigger_period + padding, verbose=verbose >= 1)
    post_trigger = digitizer.posttrigger_memory_size()

    digitizer.initialize_channels(read_ch, mV_range=mV_range, memsize=memsize)
    dataraw = digitizer.blockavg_hardware_trigger_acquisition(
        mV_range=mV_range, nr_averages=Naverage, post_trigger=post_trigger)

    # remove padding

    if isinstance(dataraw, tuple):
        dataraw = dataraw[0]
    data = np.transpose(np.reshape(dataraw, [-1, len(read_ch)]))
    data = data[:, padding_offset + paddingpix:(padding_offset + paddingpix + int(period * drate))]

    if verbose:
        print('measuresegment_m4i: processing data: data shape %s, memsize %s' % (data.shape, digitizer.data_memory_size()))

    if fig is not None:
        plt.figure(fig); plt.clf(); plt.plot(data.flatten(), '.b')
        plt.title('trace from m4i')
        
    return data


def ttrace2waveform(ttrace, pulsars, name='ttrace', verbose=1, awg_map=None, markeridx=1):
    """ Create a Toivo trace

    Args:
        ttrace (ttrace_t)
        pulsars (list): list of Pulsar objects
        markeridx (int): index of Pular to use for marker 

    Returns:
        ttraces (waveforms)
        ttrace
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
        ttrace_element = element.Element(name+'%d' % pi, pulsar=pulsar)
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

    try:
        pi, ci, mi = awg_map['fpga_mk']
    except:
        pi, ci, mi = awg_map['m4i_mk']
        
    pulsar = pulsars[pi]
    ttrace_element = ttraces[pi]
    ttrace_element.add(pulse.cp(sq_pulse, amplitude=.1, length=markerperiod, channel='ch%d_marker%d' % (ci, mi), channels=[]),
                       name='marker%d' % ch, start=fpga_delay, refpulse=None, refpoint='end')  # refpulse=refpulse

    if verbose:
        print('ttrace2waveform: %d traces, %d pulsars' % (ntraces, len(ttraces)))

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
            print('  awg %d: channel %s: amplitude %.2f (%s)' % (pi, ch, R, g))
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

    if verbose:
        lt=lasttime(ttraces[0])
        print('ttrace2waveform: last time on waveform 0: %.1f [ms]' % (1e3*lt))
    return ttraces, ttrace

#%%



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


def set_awg_trace(virtualawg, clock=10e6, verbose=0):
    """ Set the virtual awg in ttrace mode
    
    Args:
        virtualawg (virtual awg object)
        clock (float): clock speed to set    
    """
    
    virtualawg.AWG_clock=clock
    for a in virtualawg._awgs:
        a.clock_freq(clock)
        # needed?
        v = a.write('SOUR1:ROSC:SOUR INT')
        v = a.ask('SOUR1:ROSC:SOUR?')
        if verbose:
            print('%s: SOUR1:ROSC:SOUR? %s' % (a, v))

def init_ttrace(station, awgclock=10e6):    
    pulsar_objects =[]
    
    set_awg_trace(station.awg, awgclock)
    for ii,a in enumerate(station.awg._awgs):
        print('init_ttrace: creating Pulsar %d: awg name %s' % (ii, a.name))
        a.clock_freq.set(awgclock)

        p = ps.Pulsar(name=qtt.measurements.scans.instrumentName('Pulsar%d' % ii),
                      default_AWG=a.name)
        
        define_awg5014_channels(p, marker1highs=2.6)

        #p._clock_prequeried(False) # needed?
        _=p.clock(list(p.channels.keys())[0])
        p._clock_prequeried(True) # if not set the interface is _very_ slow
        
        setattr(station, 'pulsar%d' % ii, p)
        #p.AWG=a        


        
        pulsar_objects+=[p]
        #p.clock = awgclock

    return pulsar_objects

def run_ttrace(virtualawg, pulsar_objects, ttrace, ttrace_elements, sequence_name='ttrace'):
    """ Send the waveforms to the awg and run the awgs """
    
    #% Really run the awg
    awgs=virtualawg._awgs    
    awgclock = ttrace['awgclock']
    set_awg_trace(virtualawg, awgclock)
 
    for ii, t in enumerate(ttrace_elements):
        seq = Sequence(sequence_name + '_awg%d' % ii)
        seq.append(name='toivotrace', wfname=t.name, trigger_wait=False,)
    
        elts = [t]
    
        # program the Sequence
        pulsar = pulsar_objects[ii]
        ss = pulsar.program_awgs(seq, *elts)
    
        
    
    for awg in awgs:     
        awg.run()
        
def lastpulse(filler_element):
    """ Return last pulse from a sequence """
    keys = list(filler_element.pulses.keys())
    if len(keys)==0:
        return None
    tt = [filler_element.pulses[k].effective_stop() for k in keys]
    idx = np.argmax(tt)
    return keys[idx]


def lasttime(filler_element):
    """ Return stop time of last pulse from a sequence """
    keys = list(filler_element.pulses.keys())
    if len(keys)==0:
        return None
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
            print('add_fill: channel %s: name %s: amplitude %s, length %.6f [ms]' %
                  (ch, name, px.amplitude, 1e3*fillperiod))
        if refpulse is None:
            refpulse = name


def show_element(elmnt, fig=100, keys=None, label_map=None):
    """ Show pycqed waveform element 
    
    Args:
        elmnt (waveform_control.element.Element)
        fig (int or None): figure to plot to
        keys (None or list): channels to plot 
        label_map (None or dict)
    """
    ttc, xx = elmnt.waveforms()
    if fig is not None:
        qtt.pgeometry.cfigure(fig)
        plt.clf()

        if keys is None:
            keys = sorted(xx.keys())
        for k in keys:
            tt=ttc[k]
            v = xx[k]

            if label_map is None:
                label = k
            else:
                label = label_map[k]
            plt.plot(1e3 * tt, v, '.', label=label)
        plt.xlabel('Time [ms]')
        plt.ylabel('Signal')
    
#%%
import time

class ttrace_update:
    
    def __init__(self, station, read_function, channel, ttrace, ttrace_elements, multi_trace):
        self.station=station
        self.fps=qtt.pgeometry.fps_t()
        self.app=pg.mkQApp()
        self.read_function = read_function
        self.channel = channel
        self.ttrace =ttrace
        self.ttrace_elements =ttrace_elements
        self.multi_trace = multi_trace
        self.verbose=1
        
    def updatefunction(self):
        data_raw=self.read_function(self.station,  self.ttrace_elements, self.channel )
        tt, datax, tx = parse_data(data_raw, self.ttrace_elements, self.ttrace, verbose=self.verbose>=2)
        for ii, q in enumerate(tx):
            self.fps.showloop(dt=15)
            self.fps.addtime(time.time())
            ncurves = self.multi_trace.ncurves
            p=self.multi_trace.curves[ii]
            for jj in range(min(ncurves, len(q)) ):
                nn=len(q[jj,:])
                n0=int(nn/2)
                xrange=np.arange(-n0, -n0+nn)
                p[jj].setData(xrange, q[jj,:])
            self.app.processEvents() 
            

class MultiTracePlot:

    def __init__(self, nplots, ncurves=1, title='Multi trace plot', station=None):
        """ Plot window for multiple 1D traces """
        self.title = title
        self.verbose = 1
        self.station = station
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
        win.ppt_button = QtWidgets.QPushButton('PPT')
        for b in [win.start_button, win.stop_button, win.ppt_button]:
            b.setMaximumHeight(24)

        topLayout.addWidget(win.start_button)
        topLayout.addWidget(win.stop_button)
        topLayout.addWidget(win.ppt_button)

        vertLayout = QtWidgets.QVBoxLayout()

        vertLayout.addLayout(topLayout)
        vertLayout.addWidget(plotwin)

        win.setLayout(vertLayout)

        self.nx = int(np.ceil(np.sqrt(nplots)))
        self.ny = int(np.ceil((nplots / self.nx)))

        # Enable antialiasing for prettier plots
        # pg.setConfigOptions(antialias=True)

        self.plots = []
        self.ncurves=ncurves
        self.curves = []
        pens = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)] * 3
        
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
        win.ppt_button.clicked.connect(connect_slot(self.add_ppt))

    def add_ppt(self, notes=None):
        """ Copy current image window to PPT """
        
        if notes is None:
            notes = getattr(self, 'station', None)
        qtt.tools.addPPTslide(fig=self, title='T-traces', notes=notes)
        
    def add_verticals(self):
        vpen=pg.QtGui.QPen(pg.QtGui.QColor(100, 100, 155,60), 0, pg.QtCore.Qt.SolidLine)
        for p in self.plots:
            g=pg.InfiniteLine([0,0], angle=90, pen=vpen)    
            g.setZValue(-100)
            p.addItem(g)

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
       
      
def read_FPGA_line(station, idx=[1,],Naverage=26):
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
    samplingfreq=ttrace['samplingfreq']
    
    ttrace_element= ttraces[0]
    tracedata=ttrace['tracedata']
    
    ttotal  =ttrace_element.ideal_length() 
    
    #ttotal = ttrace_element.waveforms()[0].size / ttrace['awgclock']         
    qq=ttotal*samplingfreq

    datax = data_raw.copy()
    datax[:,0] = np.mean(datax[:,1:2], axis=1)
    
    ff=data_raw.shape[1]/qq                
    fpgafreqx=samplingfreq*ff
    tt=np.arange(0, datax.shape[1])/fpgafreqx
    tx=[]
    if tracedata is not None:
        for x in tracedata:
            sidx=int(x['start_time']*fpgafreqx)
            eidx=int(x['end_time']*fpgafreqx)
            if verbose>=2:
                print('sidx %s, eidx %s'  % (sidx, eidx))
            tx+=[ datax[:, sidx:eidx]]
            
    if verbose:
       # awg=station.awg._awgs[0]
        awgclock=ttrace['awgclock']
        tracedata=ttrace['tracedata']
        ttotal  =ttrace_element.ideal_length() 
        tsize = int(ttotal * awgclock)
        
        #tsize=ttraces[0].waveforms()[0].size
        #ttotal = ttraces[0].waveforms()[0].size / awgclock # is really slow!!
        samplingfreq=ttrace['samplingfreq']
        qq=ttotal*samplingfreq
        print('acquisition: freq %f [MHz]' % (samplingfreq / 1e6))
        print('trace length %.3f [ms], %d points' % (1e3*ttotal, tsize,))
        print('acquisition: expect %d, got %d'  % (qq, data_raw.shape[1]))

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
        plt.plot(tf*tt, data_raw[i], '.', label='raw data')
    if tf==1e3:
        plt.xlabel('Time [ms]')
    else:
        plt.xlabel('Time')
    for ii, x in enumerate(ttrace['tracedata']):
        s = x['start_time'] * tf # 
        e = x['end_time'] * tf # 
        if ii==0:
            qtt.pgeometry.plot2Dline([-1, 0, s], '--', label='start segment')
            qtt.pgeometry.plot2Dline([-1, 0, e], ':', label='end segment')
        else:
            qtt.pgeometry.plot2Dline([-1, 0, s], '--')
            qtt.pgeometry.plot2Dline([-1, 0, e], ':')
    plt.figure(fig+1); plt.clf()
    nx=int(np.ceil(np.sqrt(len(tx))))
    ny=int(np.ceil(len(tx)/nx))
    for ii, q in enumerate(tx):
        plt.subplot(nx,ny,ii+1)
        plt.plot(q.T)
