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
import copy
import numbers

from scipy import ndimage
import pyqtgraph

import qcodes
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers
import qtt
from qtt.gui.live_plotting import livePlot
from qtt.utilities.tools import connect_slot
import qtpy.QtWidgets as QtWidgets
import qtpy.QtCore
from qtt.measurements.scans import makeDataset_sweep, makeDataset_sweep_2D
from qtt.measurements.acquisition.interfaces import AcquisitionScopeInterface

# %%
from abc import ABC, abstractmethod


class VideoModeProcessor(ABC):
    """ Base class for VideoMode functionality """

    @abstractmethod
    def initialize(self, videomode):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def measure(self, videomode):
        return None

    @abstractmethod
    def scan_dimension(self):
        """ Return the dimension of the data to be shown (1 or 2) """

    @abstractmethod
    def process(self, measurement_data, videomode):
        return measurement_data

    @classmethod
    def extend_videomode_name(self, name):
        """ String to append to VideoMode name """
        return name

    @classmethod
    def ppt_notes(self):
        """ Generate notes to be added in a powerpoint slide """
        return ''

    @classmethod
    def default_processing(self, measurement_data, videomode):
        diff_sigma = videomode.diffsigma

        dd = []
        for ii, measurement_data_channel in enumerate(measurement_data):
            data_processed = np.array(measurement_data_channel)

            if videomode.diff_dir is not None:
                if isinstance(videomode.diff_dir, (list, tuple)):
                    diff_dir = videomode.diff_dir[ii]
                else:
                    diff_dir = videomode.diff_dir
                data_processed = qtt.utilities.tools.diffImageSmooth(
                    data_processed, dy=diff_dir, sigma=diff_sigma)

            if videomode.smoothing:
                data_processed = qtt.algorithms.generic.smoothImage(
                    data_processed)

            if videomode.laplace:
                data_processed = ndimage.filters.laplace(
                    data_processed, mode='nearest')
            dd.append(data_processed)
        return dd

    @classmethod
    def plot_title(self, index):
        """ Return title for plot window """
        plot_title = f'plot {index}'
        return plot_title

    def create_dataset(self, processed_data, metadata):
        alldata = [None] * len(processed_data)
        for jj, data_block in enumerate(processed_data):
            if self.scan_dimension() == 1:
                xdata = np.arange(0., data_block.size)
                alldatax = qtt.data.makeDataSet1Dplain(
                    xname='x', x=xdata, yname='signal', y=data_block, loc_record={'label': 'videomode_1d_single'})
            else:
                xdata = np.arange(0., data_block.shape[1])
                ydata = np.arange(0., data_block.shape[0])
                alldatax = qtt.data.makeDataSet2Dplain(
                    xname='x', x=xdata,
                    yname='y', y=ydata,
                    zname='signal', z=data_block,
                    loc_record={'label': 'videomode_1d_single'})
            alldatax.metadata = copy.copy(metadata)
            alldata[jj] = alldatax
        return alldata

    def acquisition_device_type(self):
        """ Return type of acquisition device 
        
        Returns:
            Device type as a string. Can be
        """
        measurement_instrument_handle = self.minstrumenthandle

        if isinstance(measurement_instrument_handle, AcquisitionScopeInterface):
            return 'AcquisitionScopeInterface'
        elif measurement_instrument_handle.name in ['digitizer', 'm4i']:
            return 'm4i'
        elif measurement_instrument_handle.name in ['ZIUHFLI', 'ziuhfli']:
            return 'ziuhfli'
        else:
            return 'other'


class VideomodeSawtoothMeasurement(VideoModeProcessor):

    def __init__(self, station, verbose=1):
        self.station = station
        self.verbose = verbose

        self.scan_parameters = {}
        self.period_1d = qcodes.ManualParameter('period_1d', initial_value=1e-3)

    def plot_title(self, index):
        plot_title = str(self.minstrumenthandle) + ' ' + str(self.channels[index])
        return plot_title

    def create_dataset(self, processed_data, metadata):
        alldata = [None] * len(processed_data)
        if processed_data.ndim == 2:
            for jj, data_block in enumerate(processed_data):
                dataset, _ = makeDataset_sweep(data_block, self.sweepparams, self.sweepranges[0],
                                               gates=self.station.gates, loc_record={'label': 'videomode_1d'})
                dataset.metadata = copy.copy(metadata)
                alldata[jj] = dataset
        elif processed_data.ndim == 3:
            for jj, data_block in enumerate(processed_data):
                dataset, _ = makeDataset_sweep_2D(data_block, self.station.gates, self.sweepparams,
                                                  self.sweepranges,
                                                  loc_record={'label': 'videomode_2d'})
                dataset.metadata = copy.copy(metadata)
                alldata[jj] = dataset
        else:
            raise Exception('makeDataset: data.ndim %d' % processed_data.ndim)

        return alldata

    def set_properties(self, waveform, measurement_instrument, resolution=None):
        """ Create callback object for videmode data

        Args:
            station (QCoDeS station)
            waveform
            measurement_instrument (tuple): instrumentname, channel
            diff_dir (list or (int, str)): differentiation modes for the data
        """
        self.waveform = waveform
        self.minstrument = measurement_instrument[0]
        self.channels = measurement_instrument[1]
        self.measuresegment_arguments = {'mV_range': 5000}

        if not isinstance(self.channels, list):
            self.channels = [self.channels]

        self.unique_channels = list(np.unique(self.channels))

        self.resolution = resolution

    def update_position(self, position, verbose=1):
        if verbose:
            print('# %s: update position: %s' % (self.__class__.__name__, position,))

        station = self.station
        if self.scan_dimension() == 1:
            delta = position[0]
            if isinstance(self.scan_parameters['sweepparams'], str):
                param = getattr(station.gates, self.scan_parameters['sweepparams'])
                param.set(delta)
                if verbose > 2:
                    print('  set %s to %s' % (param, delta))
                return
        try:
            for ii, parameter in enumerate(self.scan_parameters['sweepparams']):
                delta = position[ii]
                if verbose:
                    print('param %s: delta %.3f' % (parameter, delta))

                if isinstance(parameter, str):
                    if parameter == 'gates_horz' or parameter == 'gates_vert':
                        d = self.scan_parameters['sweepparams'][parameter]
                        for gate, factor in d.items():
                            if verbose >= 2:
                                print('  %s: increment %s with %s' % (parameter, gate, factor * delta))
                            param = getattr(station.gates, gate)
                            param.increment(factor * delta)
                    else:
                        if verbose > 2:
                            print('  set %s to %s' % (parameter, delta))
                        param = getattr(station.gates, parameter)
                        param.set(delta)
                else:
                    raise Exception('_update_position with parameter type %s not supported' % (type(parameter),))
        except Exception as ex:
            logging.exception(ex)

    def ppt_notes(self):
        return str(self.scan_parameters)

    def initialize(self, videomode):
        scan_dimensions = self.scan_dimension()
        virtual_awg = getattr(self.station, 'virtual_awg', None)
        awg = getattr(self.station, 'awg', None)

        if isinstance(self.minstrumenthandle, AcquisitionScopeInterface):
            if scan_dimensions == 1:
                self.minstrumenthandle.period = self.period_1d()
            elif scan_dimensions == 2:
                self.minstrumenthandle.number_of_samples = np.prod(self.resolution)
            scan_channels = self.scanparams['minstrument'][1]
            channel_attributes = self.scanparams['minstrument'][2]
            self.minstrumenthandle.enabled_channels = tuple(scan_channels)
            for channel, attribute in zip(scan_channels, channel_attributes):
                self.minstrumenthandle.set_input_signal(channel, attribute)
            self.minstrumenthandle.start_acquisition()

        if scan_dimensions == 1:
            self._run_1d_scan(awg, virtual_awg, period=self.period_1d())
        elif scan_dimensions == 2:
            self._run_2d_scan(awg, virtual_awg)
        else:
            raise Exception(f'type of scan not supported scan_dimensions is {scan_dimensions}')

    def stop(self):
        if self.station is not None:
            if hasattr(self.station, 'awg'):
                self.station.awg.stop()
            if hasattr(self.station, 'RF'):
                self.station.RF.off()
            if hasattr(self.station, 'virtual_awg'):
                self.station.virtual_awg.stop()
                if isinstance(self.sweepparams[0], dict):
                    keys = [list(item.keys())[0] for item in self.sweepparams]
                    self.station.virtual_awg.disable_outputs(keys)
                elif isinstance(self.sweepparams, str):
                    self.station.virtual_awg.disable_outputs([self.sweepparams])
        if isinstance(self.minstrumenthandle, AcquisitionScopeInterface):
            self.minstrumenthandle.stop_acquisition()

    def measure(self, videomode):

        if self.acquisition_device_type() == 'm4i':
            if self.scan_dimension() == 1:
                if self.sampling_frequency() * self.period_1d() > 56:
                    trigger_re_arm_compensation = True
                else:
                    trigger_re_arm_compensation = False
            else:
                trigger_re_arm_compensation = True

            trigger_re_arm_compensation = getattr(self, 'trigger_re_arm_compensation', trigger_re_arm_compensation)
            device_parameters = {'trigger_re_arm_compensation': trigger_re_arm_compensation}
        else:
            device_parameters = {}
        data = qtt.measurements.scans.measuresegment(
            self.waveform, videomode.Naverage(), self.minstrumenthandle, self.unique_channels,
            **self.measuresegment_arguments, device_parameters=device_parameters)
        if np.all(data == 0):
            raise Exception('data returned contained only zeros, aborting')

        return data

    def process(self, measurement_data, videomode):
        measurement_data_selected = []
        for ii, channel in enumerate(self.channels):
            unique_channel_idx = self.unique_channels.index(channel)
            data_processed = np.array(measurement_data[unique_channel_idx])
            measurement_data_selected.append(data_processed)

        dd = self.default_processing(measurement_data_selected, videomode)
        return dd

    def parse_instrument(self, measurement_instrument_handle, sample_rate):
        measurement_instrument_handle = qtt.measurements.scans.get_instrument(measurement_instrument_handle,
                                                                              self.station)
        self.minstrumenthandle = measurement_instrument_handle

        device_type = self.acquisition_device_type()

        if device_type == 'AcquisitionScopeInterface':
            if sample_rate != 'default':
                measurement_instrument_handle.sample_rate = sample_rate
            self.sampling_frequency = lambda: measurement_instrument_handle.sample_rate
        elif device_type == 'm4i':
            if sample_rate == 'default':
                self.sampling_frequency = measurement_instrument_handle.sample_rate
            else:
                measurement_instrument_handle.sample_rate(sample_rate)
                self.sampling_frequency = station.digitizer.sample_rate
        elif device_type == 'ziuhfli':
            self.sampling_frequency = measurement_instrument_handle.scope_samplingrate
        else:
            try:
                measurement_instrument_handle = qtt.measurements.scans.get_instrument(
                    measurement_instrument_handle, self.station)
                self.sampling_frequency = measurement_instrument_handle.sample_rate
            except Exception as ex:
                raise Exception(
                    f'no fpga or digitizer found minstrumenthandle is {measurement_instrument_handle}') from ex
        return self.sampling_frequency

    def _run_1d_scan(self, awg, virtual_awg, period=1e-3):
        if virtual_awg:
            if not isinstance(self.sweepparams, (str, dict)):
                raise Exception('argument sweepparams of type %s not supported' % type(self.sweepparams))
            sweep_range = self.sweepranges[0]
            gates = self.sweepparams if isinstance(self.sweepparams, dict) else {self.sweepparams: 1}
            waveform = virtual_awg.sweep_gates(gates, sweep_range, period)
            virtual_awg.enable_outputs(list(gates.keys()))
            virtual_awg.run()
        else:
            if isinstance(self.sweepparams, str):
                waveform, _ = awg.sweep_gate(self.sweepparams, self.sweepranges[0], period=period)
            elif isinstance(self.sweepparams, dict):
                waveform, _ = awg.sweep_gate_virt(self.sweepparams, self.sweepranges[0], period=period)
            else:
                raise Exception('argument sweepparams of type %s not supported' % type(self.sweepparams))
        self.set_properties(waveform, measurement_instrument=(self.minstrumenthandle, self.scan_parameters['channels']))

    def _run_2d_scan(self, awg, virtual_awg, period=None):
        if virtual_awg:
            sweep_ranges = [i for i in self.sweepranges]
            if isinstance(self.sweepparams[0], dict):
                gates = self.sweepparams
            else:
                # convert list of strings to dict format
                gates = self.sweepparams
                gates = [dict([(gate, 1)]) for gate in gates]

            if period is None:
                sampling_frequency = qtt.measurements.scans.get_sampling_frequency(self.minstrumenthandle)
                base_period = 1. / sampling_frequency
                total_period = base_period * np.prod(self.resolution)
            else:
                total_period = period
                warnings.warn('code untested')
            waveform = virtual_awg.sweep_gates_2d(gates, sweep_ranges, total_period, self.resolution)
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
                raise Exception('argument sweepparams of type %s not supported' % type(self.sweepparams))
            if self.verbose:
                print('%s: 2d scan, define callback ' % (self.__class__.__name__,))

        self.set_properties(waveform, measurement_instrument=(self.minstrumenthandle, self.scan_parameters['channels']),
                            resolution=self.resolution)

    def set_scan_parameters(self, scan_parameters):
        self.scan_parameters = scan_parameters
        self.sweepranges = scan_parameters['sweepranges']

        if isinstance(self.sweepranges, numbers.Number):
            self.sweepranges = [self.sweepranges]
        self.sweepparams = scan_parameters['sweepparams']
        self.resolution = scan_parameters['resolution']
        self.channels = scan_parameters['channels']

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

    def extend_videomode_name(self, name):
        """ String to append to VideoMode name """
        if name is not None:
            if isinstance(self.sweepparams, (str, list)):
                name += ': %s' % str(self.sweepparams)
        return name


def add_sawtooth_videomode_processor(self, sweepparams, sweepranges, resolution, sample_rate, minstrument):
    """ Add all required variables to the VideoMode for the VideomodeSawtoothMeasurement """
    self.resolution = resolution
    self.sample_rate = sample_rate
    self.minstrumenthandle = minstrument[0]
    channels = minstrument[1]
    if isinstance(channels, int):
        channels = [channels]
    self.channels = channels

    self.videomode_processor = VideomodeSawtoothMeasurement(self.station)

    sampling_frequency = self.videomode_processor.parse_instrument(self.minstrumenthandle, sample_rate)

    self.videomode_processor.set_scan_parameters(
        {'sweepparams': sweepparams, 'sweepranges': sweepranges, 'minstrument': minstrument,
         'resolution': self.resolution, 'sampling_frequency': sampling_frequency, 'channels': channels})


# %%


class DummyVideoModeProcessor(VideoModeProcessor):

    def __init__(self, station, verbose=1):
        """ Dummy implementation of the VideoModeProcessor """
        self.station = station

    def ppt_notes(self):
        return f'dummy notes for {self}'

    def initialize(self, videomode):
        pass

    def stop(self):
        pass

    def measure(self, videomode):
        data = [np.random.rand(100, )]

        return data

    def process(self, measurement_data, videomode):
        dd = self.default_processing(measurement_data, videomode)
        return dd

    def scan_dimension(self):
        return 1


# %%


class VideoMode:
    """ Controls the videomode tool.

    The VideoMode tools allows for fast plotting of measurement results. The basic operation of the VideoMode consists
    of the following stages:

        i) Initialize. For example start a periodic waveform on the AWG
        ii) Start the readout. This starts a loop with the following steps:
            - Measure data
            - Post-process data
            - Plot data
            The loop continues running in the background untill the user aborts the loop.
        iii) Stop the readout. This stops the measure-process-plot loop
        iv) Stop. This stops all activity (e.g. both the readout loop and and activity on the AWG)


    Attributes:
        station (qcodes station): contains all the information about the set-up
        videomode_processor (VideoModeProcessor): class performing the measurements and post-processing
        Naverage (Parameter): the number of times the raw measurement data should be averaged
    """

    videomode_class_index = 0

    def __init__(self, station, sweepparams=None, sweepranges=None, minstrument=None, nplots=None, Naverage=10,
                 resolution=(96, 96), sample_rate='default', diff_dir=None, verbose=1,
                 dorun=True, show_controls=True, add_ppt=True, crosshair=False, averaging=True, name=None,
                 mouse_click_callback=None, videomode_processor=None):
        """ Tool for fast acquisition of charge stability diagram

        The acquisition and post-processing of data is performed by a VideoModeProcessor.

        The class assumes that the station has a station.digitizer and a station.awg.

        Args:
            station (QCoDeS station): reference for the instruments
            sweepparams (list or dict or str): parameters to sweep
            sweepranges (list): list of sweep ranges
            minstrument (object): specification of the measurement instrument
            crosshair (bool): Enable crosshair in plots
            videomode_processor (None or VideoModeProcessor): Processor to use for measurement and post-processing
            mouse_click_callback (None or function): if None then update scan position with callback

        """
        if VideoMode.videomode_class_index == 0:
            # create instance of QApplication
            _ = pyqtgraph.mkQApp()
        VideoMode.videomode_class_index = VideoMode.videomode_class_index + 1
        self.videomode_index = VideoMode.videomode_class_index

        self.station = station
        self.verbose = verbose
        self.sweepparams = sweepparams
        if isinstance(sweepranges, numbers.Number):
            sweepranges = [sweepranges]
        self.sweepranges = sweepranges

        self.virtual_awg = getattr(station, ' virtual_awg', None)

        self.Naverage = Parameter('Naverage', get_cmd=self._get_Naverage,
                                  set_cmd=self._set_Naverage, vals=Numbers(1, 1023))

        # set core VideoMode properties
        self.diff_dir = diff_dir
        self.smoothing = 0
        self.datalock = threading.Lock()
        self.datafunction_result = None
        self.update_sleep = 1e-5
        self.diffsigma = 1
        self.diff = True
        self.laplace = False
        self.idx = 0
        self.fps = qtt.pgeometry.fps_t(nn=6)

        if videomode_processor is None:
            add_sawtooth_videomode_processor(self, sweepparams, sweepranges, resolution, sample_rate, minstrument)
            # Setup the GUI
            if nplots is None:
                nplots = len(self.channels)
        else:
            self.videomode_processor = videomode_processor

        self.set_videomode_name(name)

        self.nplots = nplots
        self.window_title = "%s: nplots %d" % (
            self.name, self.nplots)

        self._create_gui(nplots, show_controls, add_ppt, crosshair, Naverage, averaging)

        self.Naverage(Naverage)

        if mouse_click_callback is None:
            mouse_click_callback = self._update_position
        for p in self.lp:
            p.sigMouseClicked.connect(mouse_click_callback)

        if dorun:
            self.run()

    def set_videomode_name(self, name):
        """ Set the name for this instance of the tool """
        if name is None:
            name = 'VideoMode %d' % self.videomode_index
            name = self.videomode_processor.extend_videomode_name(name)
        self.name = name

    def _create_gui(self, number_of_plots, show_controls, add_ppt, crosshair, Naverage, averaging):
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
        plot_dimension = self.videomode_processor.scan_dimension()
        for ii in range(number_of_plots):
            lp = livePlot(self.videomode_processor, self.station.gates,
                          self.sweepparams, self.sweepranges, show_controls=False,
                          plot_title=self.videomode_processor.plot_title(ii), plot_dimension=plot_dimension)
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

    def __repr__(self):
        s = '%s: %s at 0x%x' % (self.__class__.__name__, self.name, id(self))
        return s

    def _update_position(self, position, verbose=1):
        """ Update position of gates with selected point

        Args:
            position (array): point with new coordinates
            verbose (int): verbosity level
        """
        if isinstance(self.videomode_processor, VideomodeSawtoothMeasurement):
            self.videomode_processor.update_position(position, verbose=verbose)

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

        setting_notes = 'moving averaging enabled: %d\n' % self._averaging_enabled
        setting_notes += 'number of averages %d\n' % self.Naverage()

        qtt.utilities.tools.addPPTslide(fig=self, title='VideoMode %s' % self.name,
                                        notes=self.station,
                                        extranotes='date: %s' % (
                                            qtt.data.dateString(),) + '\n' + 'videomode_processor: ' + str(
                                            self.videomode_processor.ppt_notes()) + '\n\n' + setting_notes)
        if isrunning:
            self.startreadout()

    def updatebg(self):
        """ Update function for the tool

        Calls the videomode_processor.measure() and videomode_processor.process() and updates the GUI
        """
        if self.idx % 10 == 0:
            logging.debug('%s: updatebg %d' %
                          (self.__class__.__name__, self.idx))
        self.idx = self.idx + 1
        self.fps.addtime(time.time())
        try:
            measured_data = self.videomode_processor.measure(self)
            datafunction_result = self.videomode_processor.process(measured_data, self)
            self.datafunction_result = datafunction_result
            for ii, d in enumerate(datafunction_result):
                self.lp[ii].update(data=d, processevents=False)
            pyqtgraph.mkQApp().processEvents()
        except Exception as ex:
            logging.exception(ex)
            print('%s: Exception in updatebg, stopping readout' %
                  self.__class__.__name__)
            self.stopreadout()

        self.mainwin.setWindowTitle(
            self.window_title + ' %.1f [fps]' % self.fps.framerate())

        if self.fps.framerate() < 10:
            time.sleep(0.1)
        time.sleep(self.update_sleep)

    def is_running(self):
        """ Return True if the readout loop is running """
        return self.timer.isActive()

    def startreadout(self, callback=None, rate=30, maxidx=None):
        """ Start the readout loop

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
        """ Stop the readout loop """
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
        """ Stop the videomode and close the GUI"""
        self.stop()
        self.mainwin.close()

    def get_dataset(self):
        """ Return latest recorded dataset

        Returns:
            alldata (dataset or list of datasets)

        """
        with self.datalock:
            data = self.datafunction_result
            if data is not None:
                data = np.array(data)
                self.alldata = self._makeDataset(data, Naverage=None)
            else:
                self.alldata = None
            return self.alldata

    def initialize(self):
        """ Initialize the videomode tool for the readout loop """
        self.videomode_processor.initialize(self)

    def run(self, start_readout=True):
        """ Initialize the tool and start the readout loop """
        if self.verbose:
            print('%s: run ' % (self.__class__.__name__,))

        self.initialize()

        if start_readout:
            if self.verbose:
                print('%s: run: startreadout' % (self.__class__.__name__,))
            self.startreadout()

    def stop(self):
        """ Stops the readout loop and the input signals """
        self.stopreadout()
        self.stop_videomode()

    def stop_videomode(self):
        if self.verbose:
            print('%s: stop_videomode ' % (self.__class__.__name__,))

        self.videomode_processor.stop()

    def single(self):
        """ Do a single scan with a lot averaging.

        Note: this does not yet support the usage of linear combinations of gates (a.k.a. virtual gates).
        """
        raise Exception('not implemented')

    def crosshair(self, *args, **kwargs):
        """ Enable or disable a crosshair in the plotting windows """
        for l in self.lp:
            l.crosshair(*args, **kwargs)

    def _makeDataset(self, data, Naverage=None):
        """ Create datasets from the processed data """
        metadata = {'scantime': str(datetime.datetime.now(
        )), 'station': self.station.snapshot(), 'allgatevalues': self.station.gates.allvalues()}
        metadata['Naverage'] = Naverage

        alldata = self.videomode_processor.create_dataset(data, metadata)
        return alldata

    def _get_Naverage(self):
        return self._Naverage_val

    def _set_Naverage(self, value):
        self._Naverage_val = value
        self.box.setValue(value)

    @staticmethod
    def all_instances(verbose=1):
        """ Return all VideoMode instances """
        lst = qtt.pgeometry.list_objects(VideoMode, verbose=verbose)
        return lst

    @staticmethod
    def get_instance(idx):
        """ Return instance by index """
        lst = VideoMode.all_instances(verbose=0)
        for l in lst:
            if l.videomode_index == idx:
                return l
        return None

    @staticmethod
    def stop_all_instances():
        """ Stop readout on all all VideoMode instances """
        lst = qtt.pgeometry.list_objects(VideoMode)
        for v in lst:
            v.stopreadout()


# %% Testing


if __name__ == '__main__':
    import qtt.simulation.virtual_dot_array
    import qtt.instrument_drivers.simulation_instruments
    from importlib import reload

    reload(qtt.instrument_drivers.simulation_instruments)
    reload(qtt.measurements.scans)
    from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer
    from qtt.instrument_drivers.simulation_instruments import SimulationAWG

    station = qtt.simulation.virtual_dot_array.initialize()
    gates = station.gates

    pv = qtt.createParameterWidget([gates])  # type: ignore

    verbose = 1
    multiprocess = False

    digitizer = SimulationDigitizer(
        qtt.measurements.scans.instrumentName('sdigitizer'), model=station.model)
    station.add_component(digitizer)

    station.awg = SimulationAWG(qtt.measurements.scans.instrumentName('vawg'))
    station.add_component(station.awg)

    dummy_processor = DummyVideoModeProcessor(station)
    vm = VideoMode(station, Naverage=25, diff_dir=None, verbose=1,
                   nplots=1, dorun=False, videomode_processor=dummy_processor)
    vm.updatebg()

    self = vm

    if 0:
        sweepparams = [{'P1': 1}, {'P2': 1}]
        minstrument = (digitizer.name, [0])
        vm = VideoMode(station, sweepparams, sweepranges=[120] * 2,
                       minstrument=minstrument, resolution=[12] * 2, Naverage=2)
        self = vm

    if 1:
        sweepparams = ['P1', 'P2']
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
                   verbose=1, nplots=None, dorun=False)
    self = vm
    self.initialize()
    measured_data = self.videomode_processor.measure(vm)
    dd = self.videomode_processor.process(measured_data, vm)

    vm.updatebg()
    vm.run()
    vm.get_dataset()

    vm.setGeometry(1310, 100, 800, 800)
    self = vm

    vm.stopreadout()
    vm.updatebg()
    vm.stop()
