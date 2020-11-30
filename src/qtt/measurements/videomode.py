# -*- coding: utf-8 -*-
"""
Contains code for the VideoMode tools

"""
# %%
import time
import datetime
import threading
import numpy as np
import logging
import numbers

import pyqtgraph

from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers
import qtt
from qtt.gui.live_plotting import livePlot
from qtt.utilities.tools import connect_slot
import qtpy.QtWidgets as QtWidgets
import qtpy.QtCore
from qtt.measurements.videomode_processor import VideomodeSawtoothMeasurement

# %%


def add_sawtooth_videomode_processor(videomode, sweepparams, sweepranges, resolution, sample_rate, minstrument):
    """ Add all required variables to the VideoMode for the VideomodeSawtoothMeasurement """
    videomode.resolution = resolution
    videomode.sample_rate = sample_rate
    videomode.minstrumenthandle = minstrument[0]
    channels = minstrument[1]
    if isinstance(channels, int):
        channels = [channels]
    videomode.channels = channels

    videomode.videomode_processor = VideomodeSawtoothMeasurement(videomode.station)

    sampling_frequency = videomode.videomode_processor.parse_instrument(videomode.minstrumenthandle, sample_rate)

    videomode.videomode_processor.set_scan_parameters(
        {'sweepparams': sweepparams, 'sweepranges': sweepranges, 'minstrument': minstrument,
         'resolution': videomode.resolution, 'sampling_frequency': sampling_frequency, 'channels': channels})


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
        VideoMode.videomode_class_index += 1
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
        self.name = ''
        self.maxidx = None
        self.datafunction = None
        self.alldata = None
        self._averaging_enabled = None
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
        self.timer.start(int(1000 * (1. / rate)))
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

        if self.verbose >= 2:
            print(f'{self.__class__}: close')
        for liveplot in self.lp:
            liveplot.stopreadout()
            liveplot.deleteLater()
        if self.verbose >= 2:
            print(f'{self.__class__}: call mainwin.close')

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
        metadata = {'scantime': str(datetime.datetime.now()), 'station': self.station.snapshot(),
                    'allgatevalues': self.station.gates.allvalues(), 'Naverage': Naverage}

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

    @staticmethod
    def destruct():
        """ Stop all VideoMode instances and cleanup """
        lst = qtt.pgeometry.list_objects(VideoMode)
        for v in lst:
            v.stopreadout()
            v.stop()
            v.close()
        pyqtgraph.cleanup()
        VideoMode.videomode_class_index = 0


# %% Testing


if __name__ == '__main__':
    import qtt.simulation.virtual_dot_array
    import qtt.instrument_drivers.simulation_instruments
    from importlib import reload

    reload(qtt.instrument_drivers.simulation_instruments)
    reload(qtt.measurements.scans)
    from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer
    from qtt.instrument_drivers.simulation_instruments import SimulationAWG
    from qtt.measurements.videomode_processor import DummyVideoModeProcessor

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
