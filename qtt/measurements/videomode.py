# -*- coding: utf-8 -*-
"""
Contains code to do live plotting 

"""
# %%
import time
import datetime
import threading
import numpy as np
import warnings
import logging
from scipy import ndimage
from colorama import Fore
import pyqtgraph

import qtt
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers
from qtt.live_plotting import livePlot
from qtt.tools import connect_slot
import qtpy.QtWidgets as QtWidgets
import qtpy.QtCore
from qtt.measurements.scans import makeDataset_sweep, makeDataset_sweep_2D


# %%


class videomode_callback:

    def __init__(self, station, waveform, Naverage, minstrument,
                 diff_dir=None, resolution=None):
        """ Create callback object for videmode data

        Args:
            station (QCoDeS station)
            waveform
            Naverage (int): number of average to take
            minstrument (tuple): instrumentname, channel
            diff_dir (list or (int, str)): differentiation modes for the data
        """
        self.station = station
        self.waveform = waveform
        self.Naverage = Naverage
        self.minstrument = minstrument[0]
        self.channel = minstrument[1]
        self.channels = self.channel
        if not isinstance(self.channels, list):
            self.channels = [self.channels]

        self.unique_channels = list(np.unique(self.channels))

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

        minstrumenthandle = qtt.measurements.scans.get_instrument(self.minstrument)

        data = qtt.measurements.scans.measuresegment(
            self.waveform, self.Naverage, minstrumenthandle, self.unique_channels)

        if np.all(data == 0):
            #self.stopreadout()
            raise Exception('data returned contained only zeros, aborting')

        dd = []
        for ii, channel in enumerate(self.channels):
            uchannelidx = self.unique_channels.index(channel)
            data_processed = np.array(data[uchannelidx])

            if self.diff_dir is not None:
                if isinstance(self.diff_dir, (list, tuple)):
                    diff_dir = self.diff_dir[ii]
                else:
                    diff_dir = self.diff_dir
                data_processed = qtt.tools.diffImageSmooth(
                    data_processed, dy=diff_dir, sigma=self.diffsigma)

            if self.smoothing:
                data_processed = qtt.algorithms.generic.smoothImage(
                    data_processed)

            if self.laplace:
                data_processed = ndimage.filters.laplace(
                    data_processed, mode='nearest')
            dd.append(data_processed)
        return dd


# %%


class VideoMode:
    """ Controls the videomode tuning.

    Attributes:
        station (qcodes station): contains all the information about the set-up
        sweepparams (string, 1 x 2 list or dict): the parameter(s) to be swept
        sweepranges (int or 1 x 2 list): the range(s) to be swept over
        minstrument (int or tuple): the channel of the FPGA, or tuple (instrument, channel)
        Naverage (int): the number of times the FPGA averages
        resolution (1 x 2 list): for 2D the resolution
        nplots (int or None): number of plots to show. must be equal to the number of channels in the minstrument argument
        sample_rate (float): sample rate for acquisition device
        crosshair (bool): enable crosshair
    """

    # TODO: implement optional sweep directions, i.e. forward and backward
    def __init__(self, station, sweepparams, sweepranges, minstrument, nplots=None, Naverage=10,
                 resolution=[90, 90], sample_rate='default', diff_dir=None, verbose=1,
                 dorun=True, show_controls=True, add_ppt=True, crosshair=False, averaging=True, name='VideoMode',
                 mouse_click_callback=None):
        """ Tool for fast acquisition of charge stability diagram
        
        The class assumes that the station has a station.digitizer and a station.awg.
        
        Args:
            station (QCoDeS station): reference for the instruments
            sweepparams (list or dict or str): parameters to sweep
            mouse_click_callback (None or function): if None then update scan position with callback
            
        """
        self.name = name
        self.station = station
        self.verbose = verbose
        self.sweepparams = sweepparams
        self.sweepranges = sweepranges
        self.virtual_awg = station.virtual_awg

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
        self.datafunction_result = None
        self.update_sleep = 1e-5

        # parse instrument
        if 'fpga' in station.components:
            self.sampling_frequency = station.fpga.sampling_frequency
        elif 'digitizer' in station.components:
            if sample_rate == 'default':
                self.sampling_frequency = station.digitizer.sample_rate
            else:
                station.digitizer.sample_rate(sample_rate)
                self.sampling_frequency = station.digitizer.sample_rate
        elif 'ZIUHFLI' in station.components:
            self.sampling_frequency = station.ZIUHFLI.scope_samplingrate
        else:
            try:
                minstrumenthandle = qtt.measurements.scans.get_instrument(
                    minstrument, station)
                self.sampling_frequency = minstrumenthandle.sample_rate
            except:
                raise Exception('no fpga or digitizer found')

        # for addPPT and debugging
        self.scanparams = {'sweepparams': sweepparams, 'sweepranges': sweepranges, 'minstrument': minstrument,
                           'resolution': self.resolution, 'sampling_frequency': self.sampling_frequency()}
        self.idx = 0
        self.fps = qtt.pgeometry.fps_t(nn=6)

        # Setup the GUI
        if nplots is None:
            nplots = len(self.channels)
        self.nplots = nplots
        self.window_title = "%s: nplots %d" % (
            self.name, self.nplots)

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

        self.plotLayout = QtWidgets.QHBoxLayout()
        vertLayout.addLayout(self.plotLayout)

        self.lp = []
        for ii in range(nplots):
            lp = livePlot(None, self.station.gates,
                          self.sweepparams, self.sweepranges, show_controls=False,
                          plot_title=str(self.minstrumenthandle) + ' ' + str(self.channels[ii]))
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

        self.timer = qtpy.QtCore.QTimer()
        self.timer.timeout.connect(self.updatebg)

        self.crosshair(show=crosshair)

        self.enable_averaging_slot(averaging=averaging)

        if mouse_click_callback is None:
            mouse_click_callback = self._update_position
        for p in self.lp:
            p.sigMouseClicked.connect(mouse_click_callback)

        if dorun:
            self.run()

    def __repr__(self):
        s = '%s: %s at 0x%x' % (self.__class__.__name__, self.name, id(self))
        return s

    def _update_position(self, position, verbose=1):
        """ Update position of gates with selected point

        Args:
            position (array): point with new coordinates
            verbose (int): verbosity level
        """
        station = self.station

        if verbose:
            print('# %s: update position: %s' % (self.__class__.__name__, position,))

        if self.scan_dimension() == 1:
            delta = position[0]
            if isinstance(self.scanparams['sweepparams'], str):
                param = getattr(station.gates, self.scanparams['sweepparams'])
                param.set(delta)
                if verbose > 2:
                    print('  set %s to %s' % (param, delta))
                return
        try:
            for ii, p in enumerate(self.scanparams['sweepparams']):
                delta = position[ii]
                if verbose:
                    print('param %s: delta %.3f' % (p, delta))

                if isinstance(p, str):
                    if p == 'gates_horz' or p == 'gates_vert':
                        d = self.scanparams['sweepparams'][p]
                        for g, f in d.items():
                            if verbose >= 2:
                                print('  %: increment %s with %s' % (p, g, f * delta))
                            param = getattr(station.gates, g)
                            param.increment(f * delta)
                    else:
                        g = p
                        if verbose > 2:
                            print('  set %s to %s' % (g, delta))
                        param = getattr(station.gates, g)
                        param.set(delta)
                else:
                    raise Exception('not supported')
        except Exception as ex:
            logging.exception(ex)

    def enable_averaging_slot(self, averaging=None, *args, **kwargs):
        """ Update the averaging mode of the widget """
        if averaging is None:
            self._averaging_enabled = self.mainwin.averaging_box.checkState()
        else:
            self._averaging_enabled = averaging
            self.mainwin.averaging_box.setChecked(self._averaging_enabled)

        for l in self.lp:
            l.enable_averaging(self._averaging_enabled)

    def addPPT(self):
        """ Copy image of videomode window to PPT """
        isrunning = self.is_running()
        if isrunning:
            self.stopreadout()  # prevent multi-threading issues
            time.sleep(0.2)
        qtt.tools.addPPTslide(fig=self, title='VideoMode %s' % self.name,
                              notes=self.station,
                              extranotes='date: %s' % (qtt.data.dateString(),) + '\n' + 'scanjob: ' + str(
                                  self.scanparams))
        if isrunning:
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
                    pyqtgraph.mkQApp().processEvents()
            except Exception as e:
                logging.exception(e)
                print('%s: Exception in updatebg, stopping readout' %
                      self.__class__.__name__)
                self.stopreadout()
        else:
            self.stopreadout()
            dd = None

        self.mainwin.setWindowTitle(
            self.window_title + ' %.1f [fps]' % self.fps.framerate())

        if self.fps.framerate() < 10:
            time.sleep(0.1)
        time.sleep(self.update_sleep)

    def is_running(self):
        return self.timer.isActive()

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
        """ Return latest recorded dataset

        Returns:
            alldata (dataset or list of datasets)

        """
        with self.datalock:
            if run:
                warnings.warn('not supported')
                # [l.datafunction_result for l in self.lp]
                data = self.datafunction_result
                data = np.array(data)
            else:
                data = self.datafunction_result
                if data is not None:
                    data = np.array(data)
            if data is not None:
                self.alldata = self.makeDataset(data, Naverage=None)
            return self.alldata

    def scan_dimension(self):
        if isinstance(self.sweepranges, (float, int)):
            return 1
        elif isinstance(self.sweepranges, list):
            if len(self.sweepranges) == 2:
                return 2
            elif len(self.sweepranges) == 1:
                return 1
            else:
                raise Exception('scan dimension not supported')
        else:
            return -1

    def run(self, start_readout=True):
        """ Programs the AWG, starts the read-out and the plotting. """
        if self.verbose:
            print(Fore.BLUE + '%s: run ' % (self.__class__.__name__,) + Fore.RESET)
        scan_dimentions = self.scan_dimension()
        virtual_awg = getattr(self.station, 'virtual_awg', None)
        awg = getattr(self.station, 'awg', None)

        if scan_dimentions == 1:
            self.__run_1d_scan(awg, virtual_awg)
        elif scan_dimentions == 2:
            self.__run_2d_scan(awg, virtual_awg)
        else:
            raise Exception('type of scan not supported')

        if start_readout:
            if self.verbose:
                print(Fore.BLUE + '%s: run: startreadout' %
                      (self.__class__.__name__,) + Fore.RESET)
            self.startreadout()

    def __run_1d_scan(self, awg, virtual_awg, period=1e-3):
        if virtual_awg:
            if not isinstance(self.sweepparams, (str, dict)):
                raise Exception('arguments not supported')
            sweep_range = self.sweepranges * 2
            gates = self.sweepparams if isinstance(self.sweepparams, dict) else {self.sweepparams: 1}
            waveform = virtual_awg.sweep_gates(gates, sweep_range, period)
            virtual_awg.enable_outputs(list(gates.keys()))
            virtual_awg.run()
        else:
            if type(self.sweepparams) is str:
                waveform, _ = awg.sweep_gate(self.sweepparams, self.sweepranges, period=period)
            elif type(self.sweepparams) is dict:
                waveform, _ = awg.sweep_gate_virt(self.sweepparams, self.sweepranges, period=period)
            else:
                raise Exception('arguments not supported')
        self.datafunction = videomode_callback(self.station, waveform, self.Naverage.get(),
                                               minstrument=(self.minstrumenthandle, self.channels))

    def __run_2d_scan(self, awg, virtual_awg, period=1e-6):
        if virtual_awg:
            sweep_ranges = [i * 2 for i in self.sweepranges]
            if isinstance(self.sweepparams, dict):
                gates = self.sweepparams
            elif isinstance(self.sweepparams, list):
                gates = self.sweepparams

            waveform = virtual_awg.sweep_gates_2d(gates, sweep_ranges, period, self.resolution)
            keys = [list(item.keys())[0] for item in gates]
            virtual_awg.enable_outputs(keys)
            virtual_awg.run()
        else:
            if isinstance(self.sweepparams, list):
                waveform, _ = awg.sweep_2D(self.sampling_frequency.get(), self.sweepparams,
                                           self.sweepranges, self.resolution)
            elif isinstance(self.sweepparams, dict):
                waveform, _ = awg.sweep_2D_virt(self.sampling_frequency.get(), self.sweepparams[
                    'gates_horz'], self.sweepparams['gates_vert'], self.sweepranges, self.resolution)
            else:
                raise Exception('arguments not supported')
            if self.verbose:
                print(Fore.BLUE + '%s: 2d scan, define callback ' %
                      (self.__class__.__name__,) + Fore.RESET)
        self.datafunction = videomode_callback(self.station, waveform, self.Naverage.get(),
                                               minstrument=(self.minstrumenthandle, self.channels),
                                               resolution=self.resolution, diff_dir=self.diff_dir)

    def stop(self):
        """ Stops the plotting, AWG(s) and if available RF. """
        self.stopreadout()
        if self.station is not None:
            if hasattr(self.station, 'awg'):
                self.station.awg.stop()
            if hasattr(self.station, 'RF'):
                self.station.RF.off()
            if hasattr(self.station, 'virtual_awg'):
                self.station.virtual_awg.stop()

    def single(self):
        """Do a single scan with a lot averaging.

        Note: this does not yet support the usage of linear combinations of 
        gates (a.k.a. virtual gates).
        """
        raise Exception('not implemented')

    def crosshair(self, *args, **kwargs):
        for l in self.lp:
            l.crosshair(*args, **kwargs)

    def makeDataset(self, data, Naverage=None):
        """ Helper function """
        metadata = {'scantime': str(datetime.datetime.now(
        )), 'station': self.station.snapshot(), 'allgatevalues': self.station.gates.allvalues()}
        metadata['Naverage'] = Naverage
        if hasattr(self.datafunction, 'diff_dir'):
            metadata['diff_dir'] = self.datafunction.diff_dir

        if data.ndim == 2:
            if (data.shape[0] > 1):
                raise Exception('not yet implemented')
            data = data[0]
            alldata, _ = makeDataset_sweep(data, self.sweepparams, self.sweepranges,
                                           gates=self.station.gates, loc_record={'label': 'videomode_1d_single'})
            alldata.metadata = metadata
        elif data.ndim == 3:
            if (data.shape[0] > 1):
                warnings.warn('getting dataset for multiple dimensions not yet tested')

            import copy
            alldata = [None] * len(data)
            for jj in range(len(data)):
                datax = data[jj]
                alldatax, _ = makeDataset_sweep_2D(datax, self.station.gates, self.sweepparams, self.sweepranges,
                                                   loc_record={
                                                       'label': 'videomode_2d_single'})
                alldatax.metadata = copy.copy(metadata)
                alldata[jj] = alldatax
        else:
            raise Exception('makeDataset: data.ndim %d' % data.ndim)
        return alldata

    def _get_Naverage(self):
        return self._Naverage_val

    def _set_Naverage(self, value):
        self._Naverage_val = value
        self.datafunction.Naverage = value
        self.box.setValue(value)

    @staticmethod
    def all_instances():
        """ Return all VideoMode instances """
        lst = qtt.pgeometry.list_objects(VideoMode)
        return lst

    @staticmethod
    def stop_all_instances():
        """ Stop readout on all all VideoMode instances """
        lst = qtt.pgeometry.list_objects(VideoMode)
        for v in lst:
            v.stopreadout()


# %% Testing


if __name__ == '__main__':
    from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer
    from qtt.instrument_drivers.simulation_instruments import SimulationAWG
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

    station.awg = SimulationAWG(qtt.measurements.scans.instrumentName('vawg'))
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
    vm.setGeometry(1310, 100, 800, 800)

    # %% Test MultiTracePlot
    app = pyqtgraph.mkQApp()
    waveform, ix = station.awg.sweep_gate('P1', 50, 1e-3)
    nplots = 3
    ncurves = 2


    def read_trace_dummy():
        data = qtt.measurements.scans.measuresegment(
            waveform, Naverage=1, minstrhandle=station.sdigitizer.name, read_ch=[1, 2])
        dd = [data] * nplots
        xd = np.linspace(-waveform['sweeprange'] / 2,
                         waveform['sweeprange'] / 2, data[0].size)
        xdata = [xd] * nplots
        return xdata, dd


    xd, yd = read_trace_dummy()

    # %%
    import qtt.measurements.ttrace

    reload(qtt.measurements.ttrace)
    from qtt.measurements.ttrace import MultiTracePlot

    mt = MultiTracePlot(nplots=nplots, ncurves=ncurves)
    mt.win.setGeometry(1400, 40, 500, 500)
    mt.add_verticals()


    def callback():
        xdata, ydata = read_trace_dummy()
        mt.plot_curves(xdata, ydata)
        app.processEvents()


    mt.startreadout(callback=callback)
    mt.updatefunction()

    mt.get_dataset()
