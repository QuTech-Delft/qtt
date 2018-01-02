# -*- coding: utf-8 -*-
"""
Contains code to do live plotting 

"""
#%%
import time
import datetime
import threading
import numpy as np
from scipy import ndimage
from colorama import Fore
import pyqtgraph as pg

import qtt
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers
from qtt.live_plotting import livePlot
from qtt.tools import connect_slot
import qtpy.QtWidgets as QtWidgets
from qtpy import QtCore
from qtt.measurements.scans import plotData, makeDataset_sweep, makeDataset_sweep_2D

#%%


class videomode_callback:

    def __init__(self, station, waveform, Naverage, minstrument, waittime=0,
                 diff_dir=None, resolution=None):
        """ Create callback object for videmode data

        Args:
            station
            waveform
            Naverage
            minstrument (tuple): instrumentname, channel
            waittime (float): ???
            diff_dir
        """
        self.station = station
        self.waveform = waveform
        self.Naverage = Naverage
        self.minstrument = minstrument[0]
        self.channel = minstrument[1]
        self.channels = self.channel
        if not isinstance(self.channels, list):
            self.channels = [self.channels]

        self.waittime = waittime

        # for 2D scans
        self.resolution = resolution
        self.diffsigma = 1
        self.diff = True
        self.diff_dir = diff_dir
        self.smoothing = False
        self.laplace = False

    def __call__(self, verbose=0):
        """ Callback function to read a single line of data from the device 

        Returns:
            data (list or array): either a list with length the number of channels or a numpy array with all data
        """

        minstrumenthandle = self.station.components[self.minstrument]
        data = qtt.measurements.scans.measuresegment(
            self.waveform, self.Naverage, minstrumenthandle, self.channels)

        dd = []
        for ii in range(len(data)):
            data_processed = np.array(data[ii])

            if self.diff_dir is not None:
                data_processed = qtt.diffImageSmooth(
                    data_processed, dy=self.diff_dir, sigma=self.diffsigma)

            if self.smoothing:
                data_processed = qtt.algorithms.generic.smoothImage(
                    data_processed)

            if self.laplace:
                data_processed = ndimage.filters.laplace(
                    data_processed, mode='nearest')
            dd.append(data_processed)
        return dd

#%%


class VideoMode:
    """ Controls the videomode tuning.

    Attributes:
        station (qcodes station): contains all the information about the set-up
        sweepparams (string, 1 x 2 list or dict): the parameter(s) to be swept
        sweepranges (int or 1 x 2 list): the range(s) to be swept over
        minstrument (int or tuple): the channel of the FPGA, or tuple (instrument, channel)
        Naverage (int): the number of times the FPGA averages
        resolution (1 x 2 list): for 2D the resolution
        nplots (int): number of plots to show. must be equal to the number of channels in the minstrument argument
    """
    # TODO: implement optional sweep directions, i.e. forward and backward
    # TODO: implement virtual gates functionality

    def __init__(self, station, sweepparams, sweepranges, minstrument, nplots=1, Naverage=10,
                 resolution=[90, 90], sample_rate='default', diff_dir=None, verbose=1,
                 dorun=True, show_controls=True, add_ppt=True):
        self.station = station
        self.verbose = verbose
        self.sweepparams = sweepparams
        self.sweepranges = sweepranges

        self.minstrumenthandle = minstrument[0]
        self.channels = minstrument[1]
        if isinstance(self.channels, int):
            self.channels = [self.channels]

        self.Naverage = Parameter('Naverage', get_cmd=self._get_Naverage,
                                  set_cmd=self._set_Naverage, vals=Numbers(1, 1023))
        self._Naverage_val = Naverage
        self.resolution = resolution
        self.sample_rate = sample_rate
        self.diff_dir = diff_dir
        self.datalock = threading.Lock()

        # parse instrument
        if 'fpga' in station.components:
            self.sampling_frequency = station.fpga.sampling_frequency
        elif 'digitizer' in station.components:
            if sample_rate == 'default':
                self.sampling_frequency = station.digitizer.sample_rate
            else:
                station.digitizer.sample_rate(sample_rate)
                self.sampling_frequency = station.digitizer.sample_rate
        else:
            try:
                minstrumenthandle = qtt.measurements.scans.get_instrument(
                    minstrument, station)
                self.sampling_frequency = minstrumenthandle.sample_rate
            except:
                raise Exception('no fpga or digitizer found')

        self.idx = 0
        self.fps = qtt.pgeometry.fps_t(nn=6)

        # Setup the GUI
        if nplots is None:
            nplots = len(self.channels)
        self.nplots = nplots
        self.window_title = "%s: nplots %d" % (self.__class__.__name__, self.nplots )

        win = QtWidgets.QWidget()
        win.setWindowTitle(self.window_title)
        self.mainwin = win

        vertLayout = QtWidgets.QVBoxLayout()
        self.topLayout = None
        if show_controls:
            topLayout = QtWidgets.QHBoxLayout()
            win.start_button = QtWidgets.QPushButton('Start')
            win.stop_button = QtWidgets.QPushButton('Stop')
            topLayout.addWidget(win.start_button)
            topLayout.addWidget(win.stop_button)
            if add_ppt:
                win.ppt_button = QtWidgets.QPushButton('Copy to PPT')
                win.ppt_button.clicked.connect(connect_slot(self.addPPT))
                topLayout.addWidget(win.ppt_button)

            win.averaging_box = QtWidgets.QCheckBox('Averaging')

            for b in [win.start_button, win.stop_button]:
                b.setMaximumHeight(24)

            topLayout.addWidget(win.averaging_box)
            vertLayout.addLayout(topLayout)
            self.topLayout = topLayout

        win.setLayout(vertLayout)
        # self.mainwin.add

        self.plotLayout = QtWidgets.QHBoxLayout()
        vertLayout.addLayout(self.plotLayout)

        self.lp = []
        for ii in range(nplots):

            lp = livePlot(None, self.station.gates,
                          self.sweepparams, self.sweepranges, show_controls=False,
                          plot_title=str(self.minstrumenthandle)+' '  + str(self.channels[ii]) )
            self.lp.append(lp)
            self.plotLayout.addWidget(self.lp[ii].win)

        if show_controls:
            self.mainwin.averaging_box.clicked.connect(
                connect_slot(self.enable_averaging_slot))

        self.mainwin.start_button.clicked.connect(connect_slot(self.run))
        self.mainwin.stop_button.clicked.connect(connect_slot(self.stop))
        self.box = self._create_naverage_box(Naverage=Naverage)
        self.topLayout.addWidget(self.box)

        self.setGeometry = self.mainwin.setGeometry
        self.mainwin.resize(800, 600)
        self.mainwin.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updatebg)

        if dorun:
            self.run()

    def enable_averaging_slot(self, *args, **kwargs):
        """ Update the averaging mode of the widget """
        self._averaging_enabled = self.mainwin.averaging_box.checkState()
        for l in self.lp:
            l.enable_averaging(self._averaging_enabled)

    def addPPT(self):
        """ Copy image of videomode window to PPT """
        self.stopreadout() # prevent multi-threading issues        
        time.sleep(0.2)
        qtt.tools.addPPTslide(fig=self, title='VideoMode', notes=self.station)
        self.startreadout()

    def updatebg(self):
        """ Update function for the widget 

        Calls the datafunction() and update() function for all subplots
        """
        if self.idx % 10 == 0:
            logging.debug('%s: updatebg %d' %
                          (self.__class__.__name__, self.idx))
        self.idx = self.idx + 1
        self.fps.addtime(time.time())
        if self.datafunction is not None:
            try:
                dd = self.datafunction()
                self.datafunction_result = dd
                if self.nplots == 1:
                    self.lp[0].update(data=dd[0])
                else:
                    for ii, d in enumerate(dd):
                        self.lp[ii].update(data=d, processevents=False)
                    pg.mkQApp().processEvents()
            except Exception as e:
                logging.exception(e)
                print('%s: Exception in updatebg, stopping readout' %
                      self.__class__.__name__)
                self.stopreadout()
        else:
            self.stopreadout()
            dd = None

        self.mainwin.setWindowTitle(self.window_title + ' %.1f [fps]' % self.fps.framerate())

        if self.fps.framerate() < 10:
            time.sleep(0.1)
        time.sleep(0.00001)

    def startreadout(self, callback=None, rate=30, maxidx=None):
        """
        Args:
            rate (float): sample rate in ms

        """
        if maxidx is not None:
            self.maxidx = maxidx
        if callback is not None:
            self.datafunction = callback
        self.timer.start(1000 * (1. / rate))
        if self.verbose:
            print('%s: start readout' % (self.__class__.__name__,))

    def stopreadout(self):
        if self.verbose:
            print('%s: stop readout' % (self.__class__.__name__,))
        self.timer.stop()
        self.mainwin.setWindowTitle(self.window_title + ' stopped')

    def _create_naverage_box(self, Naverage=1):
        box = QtWidgets.QSpinBox()
        box.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        # do not emit signals when still editing
        box.setKeyboardTracking(False)
        box.setMinimum(1)
        box.setMaximum(1023)
        box.setPrefix('Naverage: ')
        box.setValue(Naverage)
        box.setMaximumWidth(120)
        box.valueChanged.connect(self.Naverage.set)
        return box

    def close(self):
        self.mainwin.close()

    def get_dataset(self, run=False):
        """ Return latest recorded dataset """
        with self.datalock:
            if run:
                #data = [l.datafunction() for l in self.lp]
                data = self.datafunction_result # [l.datafunction_result for l in self.lp]
                data = np.array(data)
            else:
                data = self.datafunction_result # [l.datafunction_result for l in self.lp]
                data = np.array(data)
            self.alldata = self.makeDataset(data, Naverage=None)
            return self.alldata

    def run(self, startreadout=True):
        """ Programs the AWG, starts the read-out and the plotting. """

        if self.verbose:
            print(Fore.BLUE + '%s: run ' %
                  (self.__class__.__name__,) + Fore.RESET)

        if type(self.sweepranges) is int:
            # 1D scan
            if type(self.sweepparams) is str:
                waveform, _ = self.station.awg.sweep_gate(
                    self.sweepparams, self.sweepranges, period=1e-3)
            elif type(self.sweepparams) is dict:
                waveform, _ = self.station.awg.sweep_gate_virt(
                    self.sweepparams, self.sweeprange, period=1e-3)
            else:
                raise Exception('arguments not supported')
            self.datafunction = videomode_callback(
                self.station, waveform, self.Naverage.get(), minstrument=(self.minstrumenthandle, self.channels))
        elif isinstance(self.sweepranges, list):
            # 2D scan
            if isinstance(self.sweepparams, list):
                waveform, _ = self.station.awg.sweep_2D(self.sampling_frequency.get(
                ), self.sweepparams, self.sweepranges, self.resolution)
            elif isinstance(self.sweepparams, dict):
                waveform, _ = self.station.awg.sweep_2D_virt(self.sampling_frequency.get(), self.sweepparams[
                                                             'gates_horz'], self.sweepparams['gates_vert'], self.sweepranges, self.resolution)
            else:
                raise Exception('arguments not supported')
            self.datafunction = videomode_callback(self.station, waveform, self.Naverage.get(), minstrument=(
                self.minstrumenthandle, self.channels), resolution=self.resolution, diff_dir=self.diff_dir)

        self._waveform = waveform
        #self.lp.datafunction = self.datafunction
        # self.box.setValue(self.Naverage.get())


        if startreadout:
            if self.verbose:
                print(Fore.BLUE + '%s: run: startreadout' %
                      (self.__class__.__name__,) + Fore.RESET)
            self.startreadout()

    def stop(self):
        """ Stops the plotting, AWG(s) and if available RF. """
        self.stopreadout()
        if self.station is not None:
            if hasattr(self.station, 'awg'):
                self.station.awg.stop()
            if hasattr(self.station, 'RF'):
                self.station.RF.off()

    def single(self):
        """Do a single scan with a lot averaging.

        Note: this does not yet support the usage of linear combinations of 
        gates (a.k.a. virtual gates).
        """
        raise Exception('not implemented')

    def makeDataset(self, data, Naverage=None):
        if data.ndim == 2:
            if (data.shape[0]>1):
                raise Exception('not yet implemented')
            data=data[0]
            alldata, _ = makeDataset_sweep(data, self.sweepparams, self.sweepranges,
                                           gates=self.station.gates, loc_record={'label': 'videomode_1d_single'})
        elif data.ndim == 3:
            if (data.shape[0]>1):
                raise Exception('not yet implemented')
            data=data[0]
            alldata, _ = makeDataset_sweep_2D(data, self.station.gates, self.sweepparams, self.sweepranges, loc_record={
                                              'label': 'videomode_2d_single'})
        else:
            raise Exception('makeDataset: data.ndim %d' % data.ndim)
        alldata.metadata = {'scantime': str(datetime.datetime.now(
        )), 'station': self.station.snapshot(), 'allgatevalues': self.station.gates.allvalues()}
        alldata.metadata['Naverage'] = Naverage
        if hasattr(self.datafunction, 'diff_dir'):
            alldata.metadata['diff_dir'] = self.datafunction.diff_dir
        return alldata

    def _get_Naverage(self):
        return self._Naverage_val

    def _set_Naverage(self, value):
        self._Naverage_val = value
        self.datafunction.Naverage = value
        self.box.setValue(value)


#%%
import qcodes
import logging


class SimulationDigitizer(qcodes.Instrument):

    def __init__(self, name, model=None, **kwargs):
        super().__init__(name, **kwargs)
        self.current_sweep = None
        self.model = model
        self.add_parameter('sample_rate', set_cmd=None, initial_value=1e6)
        self.debug = {}
        self.verbose = 0

    def measuresegment(self, waveform, channels=[0]):
        import time
        if self.verbose:
            print('%s: measuresegment: channels %s' % (self.name, channels))
            print(waveform)
        self._waveform = waveform

        sd1, sd2 = self.myhoneycomb()
        time.sleep(.05)
        return [sd1, sd2][0:len(channels)]

    def myhoneycomb(self, multiprocess=False, verbose=0):
        """
        Args:
            model (object):
            Vmatrix (array): transformation from ordered scan gates to the sourcenames 
            erange, drange (float):
            nx, ny (integer):
        """
        test_dot = self.model.ds
        waveform = self._waveform
        model = self.model
        sweepgates = waveform['sweepgates']
        ndim = len(sweepgates)

        nn = waveform['resolution']
        if isinstance(nn, float):
            nn = [nn] * ndim
        nnr = nn[::-1]  # funny reverse ordering

        if verbose >= 2:
            print('myhoneycomb: start resolution %s' % (nn,))

        if ndim != len(nn):
            raise Exception(
                'number of sweep gates %d does not match resolution' % ndim)

        ng = len(model.gate_transform.sourcenames)
        test2Dparams = np.zeros((test_dot.ngates, *nnr))
        gate2Dparams = np.zeros((ng, *nnr))
        logging.info('honeycomb: %s' % (nn,))

        rr = waveform['sweepranges']

        v = model.gate_transform.sourcenames
        ii = [v.index(s) for s in sweepgates]
        Vmatrix = np.eye(len(v))  # , 3) )
        idx = np.array((range(len(v))))
        for i, j in enumerate(ii):
            idx[i], idx[j] = idx[j], idx[i]
        Vmatrix = Vmatrix[:, idx].copy()

        sweeps = []
        for ii in range(ndim):
            sweeps.append(np.linspace(-rr[ii], rr[ii], nn[ii]))
        meshgrid = np.meshgrid(*sweeps)
        mm = tuple([xv.flatten() for xv in meshgrid])
        w = np.vstack((*mm, np.zeros((ng - ndim, mm[0].size))))
        ww = np.linalg.inv(Vmatrix).dot(w)

        for ii, p in enumerate(model.gate_transform.sourcenames):
            val = model.get_gate(p)
            gate2Dparams[ii] = val

        for ii, p in enumerate(model.gate_transform.sourcenames):
            gate2Dparams[ii] += ww[ii].reshape(nnr)

        qq = model.gate_transform.transformGateScan(
            gate2Dparams.reshape((gate2Dparams.shape[0], -1)))
        # for debugging
        self.debug['gate2Dparams'] = gate2Dparams
        self.debug['qq'] = qq

        for ii in range(test_dot.ndots):
            test2Dparams[ii] = qq['det%d' % (ii + 1)].reshape(nnr)

        if ndim == 1:
            test2Dparams = test2Dparams.reshape(
                (test2Dparams.shape[0], test2Dparams.shape[1], 1))
        # run the honeycomb simulation
        test_dot.simulate_honeycomb(
            test2Dparams, multiprocess=multiprocess, verbose=0)

        sd1 = ((test_dot.hcgs) * (model.sddist1.reshape((1, 1, -1)))).sum(axis=-1)
        sd2 = ((test_dot.hcgs) * (model.sddist2.reshape((1, 1, -1)))).sum(axis=-1)
        sd1 *= (1 / np.sum(model.sddist1))
        sd2 *= (1 / np.sum(model.sddist2))

        if model.sdnoise > 0:
            sd1 += model.sdnoise * \
                (np.random.rand(*test_dot.honeycomb.shape) - .5)
            sd2 += model.sdnoise * \
                (np.random.rand(*test_dot.honeycomb.shape) - .5)
        if ndim == 1:
            sd1 = sd1.reshape((-1,))
            sd2 = sd2.reshape((-1,))
        #plt.figure(1000); plt.clf(); plt.plot(sd1, '.b'); plt.plot(sd2,'.r')
        return sd1, sd2


class simulation_awg(qcodes.Instrument):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.add_parameter('sampling_frequency',
                           set_cmd=None, initial_value=1e6)

    def sweep_2D(self, samp_freq, sweepgates, sweepranges, resolution):
        self.current_sweep = {'waveform': 'simulation_awg', 'sweepgates': sweepgates, 'sweepranges': sweepranges,
                              'type': 'sweep_2D', 'samp_freq': samp_freq, 'resolution': resolution}
        waveform = self.current_sweep
        return waveform, None

    def sweep_gate(self, gate, sweeprange, period, width=.95, wave_name=None, delete=True):
        self.current_sweep = {'waveform': 'simulation_awg', 'gate': gate, 'sweeprange': sweeprange,
                              'type': 'sweep_gate', 'period': period, 'width': width, 'wave_name': wave_name}

        waveform = self.current_sweep
        waveform['resolution'] = [int(period * self.sampling_frequency())]
        waveform['sweepgates'] = [waveform['gate']]
        waveform['sweepranges'] = [waveform['sweeprange']]

        sweep_info = None
        return waveform, sweep_info

    def stop(self):
        pass

#%% Testing


if __name__ == '__main__':
    import pdb
    from imp import reload
    import matplotlib.pyplot as plt
    reload(qtt.live_plotting)
    from qtt.live_plotting import *

    pv = qtt.createParameterWidget([gates])

    reload(qtt.measurements.scans)
    verbose = 1
    multiprocess = False

    digitizer = SimulationDigitizer(
        qtt.measurements.scans.instrumentName('sdigitizer'), model=station.model)
    station.components[digitizer.name] = digitizer

    station.awg = simulation_awg(qtt.measurements.scans.instrumentName('vawg'))
    station.components[station.awg.name] = station.awg

    if 1:
        sweepparams = ['B0', 'B3']
        sweepranges = [160, 80]
        resolution = [80, 48]
        minstrument = (digitizer.name, [0, 1])
    else:
        sweepparams = 'B0'
        sweepranges = 160
        resolution = [60]
        minstrument = (digitizer.name, [0])
    station.model.sdnoise = .1
    vm = VideoMode(station, sweepparams, sweepranges, minstrument, Naverage=25,
                   resolution=resolution, sample_rate='default', diff_dir=None,
                   verbose=1, nplots=None, dorun=True)

    self = vm
    vm.setGeometry(4310, 100, 800, 800)

    # NEXT: use sddist1 in calculations, output simular to sd1 model
    # NEXT: 1d scans
    # NEXT: two output windows
    # NEXT: faster simulation
