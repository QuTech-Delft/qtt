# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:21 2015

@author: tud205521
"""

#%%
import time
import logging
import qtpy.QtWidgets as QtWidgets
import qtpy.QtCore as QtCore
import scipy.ndimage as ndimage

import qcodes

import qtt
import qtt.legacy
from qtt import pmatlab
import numpy as np
import pyqtgraph as pg
from functools import partial

import qtt.algorithms.generic
import qtt

#%% Communication

try:
    import redis
except:
    pass


class rda_t:

    def __init__(self):
        """ Class for simple real-time data access

        Every object has a `get` and `set` method to access simple parameters 
        globally (e.g. across different python sessions).

        """

        # we use redis as backend now
        self.r = redis.Redis(host='127.0.0.1', port=6379)  # , password='test')

        try:
            self.set('dummy_rda_t', -1)
            v = self.get_float('dummy_rda_t')
            if (not v == -1):
                raise Exception('set not equal to get')
        except Exception as e:
            print(e)
            print('rda_t: check whether redis is installed and the server is running')
            raise Exception('rda_t: communication failure')

    def get_float(self, key, default_value=None):
        """ Get value by key and convert to a float """
        v = self.get(key, default_value)
        if v is None:
            return v
        return float(v)

    def get_int(self, key, default_value=None):
        """ Get value by key and convert to an int

        Returns:
            value (int)
        """
        v = self.get(key, default_value)
        if v is None:
            return v
        return int(float(v))

    def get(self, key, default_value=None):
        """ Get a value

        Args:
            key (str): value to be retrieved
            default_value (Anything): value to return if the key is not present
        Returns:
            value (str)
        """
        value = self.r.get(key)
        if value is None:
            return default_value
        else:
            return value

    def set(self, key, value):
        """ Set a value

        Args:
            key (str): key 
            value (str): the value to be set
        """

        self.r.set(key, value)
        pass


class MeasurementControl(QtWidgets.QMainWindow):

    def __init__(self, name='Measurement Control', **kwargs):
        """ Simple control for real-time data parameters """
        super().__init__(**kwargs)
        w = self
        w.setWindowTitle(name)
        vbox = QtWidgets.QVBoxLayout()
        self.verbose = 0
        self.name = name

        self.rda = rda_t()

        self.abortbutton = QtWidgets.QPushButton()
        self.abortbutton.setText('Abort measurement')
        self.abortbutton.setStyleSheet("background-color: rgb(255,150,100);")
        self.abortbutton.clicked.connect(self.abort_measurements)
        vbox.addWidget(self.abortbutton)

        widget = QtWidgets.QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        w.resize(300, 300)
        #w.setGeometry(1700, 50, 300, 600)
        # self.update_values()
        self.show()

    def install_qcodes_hook(self):
        """ xxx """
        # patch the qcodes abort function
        def myabort():
            return int(self.rda.get('abort_measurements', 0))

        qcodes.loops.abort_measurements = myabort

    def enable_measurements(self):
        """ xxx """
        self.rda.set('abort_measurements', 0)

    def abort_measurements(self):
        """ xxxx """
        if self.verbose:
            print('%s: setting abort_measurements to 1' % self.name)

        self.rda.set('abort_measurements', 1)


if __name__ == '__main__':
    mc = MeasurementControl()
    mc.verbose = 1
    mc.setGeometry(1700, 50, 300, 400)


class RdaControl(QtWidgets.QMainWindow):

    def __init__(self, name='LivePlot Control', boxes=['xrange', 'yrange', 'nx', 'ny'], **kwargs):
        """ Simple control for real-time data parameters """
        super().__init__(**kwargs)
        w = self
        w.setGeometry(1700, 50, 300, 600)
        w.setWindowTitle(name)
        vbox = QtWidgets.QVBoxLayout()
        self.verbose = 0

        self.rda = rda_t()
        self.boxes = boxes
        self.widgets = {}
        for ii, b in enumerate(self.boxes):
            self.widgets[b] = {}
            hbox = QtWidgets.QHBoxLayout()
            self.widgets[b]['hbox'] = hbox
            tbox = QtWidgets.QLabel(b)
            self.widgets[b]['tbox'] = tbox
            dbox = QtWidgets.QDoubleSpinBox()
            dbox.setKeyboardTracking(False)  # do not emit signals when still editing

            self.widgets[b]['dbox'] = dbox
            val = self.rda.get_float(b, 100)

            dbox.setMinimum(-10000)
            dbox.setMaximum(10000)
            dbox.setSingleStep(10)
            dbox.setValue(val)
            dbox.setValue(100)
            dbox.valueChanged.connect(partial(self.valueChanged, b))
            hbox.addWidget(tbox)
            hbox.addWidget(dbox)
            vbox.addLayout(hbox)

        widget = QtWidgets.QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        self.update_values()
        self.show()

    def update_values(self):
        for ii, b in enumerate(self.boxes):
            val = self.rda.get_float(b)
            if val is None:
                # default...
                val = 100
            dbox = self.widgets[b]['dbox']
            # oldstate = dbox.blockSignals(True)
            dbox.setValue(val)
            # dbox.blockSignals(oldstate)

    def valueChanged(self, name, value):
        if self.verbose:
            print('valueChanged: %s %s' % (name, value))
        self.rda.set(name, value)
        # self.label.setStyleSheet("QLabel { background-color : #baccba; margin: 2px; padding: 2px; }");

#%%


class LivePlotControl(QtWidgets.QMainWindow):

    def __init__(self, name='LivePlot Control', boxes=['xrange', 'yrange', 'nx', 'ny'], **kwargs):
        """ Simple control widget """
        super().__init__(**kwargs)
        w = self
        w.setGeometry(1700, 50, 300, 600)
        w.setWindowTitle(name)
        vbox = QtWidgets.QVBoxLayout()
        self.verbose = 0

        self.rda = rda_t()
        self.boxes = boxes
        self.widgets = {}
        for ii, b in enumerate(self.boxes):
            self.widgets[b] = {}
            hbox = QtWidgets.QHBoxLayout()
            self.widgets[b]['hbox'] = hbox
            tbox = QtWidgets.QLabel(b)
            self.widgets[b]['tbox'] = tbox
            dbox = QtWidgets.QDoubleSpinBox()
            dbox.setKeyboardTracking(False)  # do not emit signals when still editing

            self.widgets[b]['dbox'] = dbox
            val = self.rda.get_float(b, 100)

            dbox.setMinimum(-10000)
            dbox.setMaximum(10000)
            dbox.setSingleStep(10)
            dbox.setValue(val)
            dbox.setValue(100)
            dbox.valueChanged.connect(partial(self.valueChanged, b))
            hbox.addWidget(tbox)
            hbox.addWidget(dbox)
            vbox.addLayout(hbox)

        # w.setLayout(vbox)
        widget = QtWidgets.QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        for b in self.boxes:
            if self.rda.get(b) is None:
                # make defaults...
                self.rda.set(b, 100)
        self.update_values()

    def update_values(self):
        for ii, b in enumerate(self.boxes):
            val = self.rda.get_float(b)
            if val is None:
                # default...
                val = 100
            dbox = self.widgets[b]['dbox']
            #oldstate = dbox.blockSignals(True)
            dbox.setValue(val)
            # dbox.blockSignals(oldstate)

    def valueChanged(self, name, value):
        if self.verbose:
            print('valueChanged: %s %s' % (name, value))
        self.rda.set(name, value)

#%% Liveplot object


class livePlot:
    """ Class to enable live plotting of data.

    Attributes:
        datafunction: the function to call for data acquisition
        sweepInstrument: the instrument to which sweepparams belong
        sweepparams: the parameter(s) being swept
        sweepranges: the range over which sweepparams are being swept
    """

    def __init__(self, datafunction, sweepInstrument=None, sweepparams=None, sweepranges=None, verbose=1):
        plotwin = pg.GraphicsWindow(title="Live view")

        # TODO: automatic scaling?
        # TODO: implement FPGA callback in qcodes
        # TODO: implement 2 function plot (for 2 sensing dots)

        win = QtWidgets.QWidget()
        win.resize(800, 600)
        win.setWindowTitle('livePlot')

        topLayout = QtWidgets.QHBoxLayout()
        win.start_button = QtWidgets.QPushButton('Start')
        win.stop_button = QtWidgets.QPushButton('Stop')

        for b in [win.start_button, win.stop_button]:
            b.setMaximumHeight(24)

            #self.reloadbutton.setText('Reload data')
        topLayout.addWidget(win.start_button)
        topLayout.addWidget(win.stop_button)

        vertLayout = QtWidgets.QVBoxLayout()

        vertLayout.addItem(topLayout)
        vertLayout.addWidget(plotwin)

        win.setLayout(vertLayout)

        self.win = win
        self.verbose = verbose
        self.idx = 0
        self.maxidx = 1e9
        self.data = None
        self.sweepInstrument = sweepInstrument
        self.sweepparams = sweepparams
        self.sweepranges = sweepranges
        self.fps = pmatlab.fps_t(nn=6)
        self.datafunction = datafunction

        if self.sweepparams is None:
            p1 = plotwin.addPlot(title="Videomode")
            p1.setLabel('left', 'param1')
            p1.setLabel('left', 'param2')
            data = np.array(self.datafunction())
            if data.ndim == 1:
                dd = np.zeros((0,))
                plot = p1.plot(dd, pen='b')
                self.plot = plot
            else:
                self.plot = pg.ImageItem()
                p1.addItem(self.plot)
        elif len(self.sweepparams) == 1:
            p1 = plotwin.addPlot(title="1d scan")
            p1.setLabel('left', 'Value')
            p1.setLabel('bottom', self.sweepparams[0], units='mV')
            dd = np.zeros((0,))
            plot = p1.plot(dd, pen='b')
            self.plot = plot
        elif len(self.sweepparams) == 2:
            p1 = plotwin.addPlot(title='2d scan')
            p1.setLabel('bottom', self.sweepparams[0], units='mV')
            p1.setLabel('left', self.sweepparams[1], units='mV')
            self.plot = pg.ImageItem()
            p1.addItem(self.plot)
        else:
            raise Exception(
                'The number of sweep parameters should be either None, 1 or 2.')

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updatebg)
        self.win.show()

        #from pgeometry import Slot
        def connect_slot(target):
            """ Create a slot by dropping signal arguments """
            #@Slot()
            def signal_drop_arguments(*args, **kwargs):
                #print('call %s' % target)
                target()
            return signal_drop_arguments

        win.start_button.clicked.connect(connect_slot(self.startreadout))
        win.stop_button.clicked.connect(connect_slot(self.stopreadout))
        # win.start_button.clicked.connect(self.startreadout)
        # win.stop_button.clicked.connect(self.stopreadout)

    def resetdata(self):
        self.idx = 0
        self.data = None

    def update(self, data=None, processevents=True):
        self.win.setWindowTitle('live view, fps: %.2f' % self.fps.framerate())
        if self.verbose >= 2:
            print('livePlot: update')
        if data is not None:
            self.data = np.array(data)
            if self.data.ndim == 1:
                if None in (self.sweepInstrument, self.sweepparams, self.sweepranges):
                    self.plot.setData(self.data)
                else:
                    sweep_param = getattr(self.sweepInstrument, self.sweepparams[0])
                    paramval = sweep_param.get_latest()
                    sweepvalues = np.linspace(paramval - self.sweepranges[0] / 2, self.sweepranges[
                        0] / 2 + paramval, len(data))
                    self.plot.setData(sweepvalues, self.data)
            elif self.data.ndim == 2:
                self.plot.setImage(self.data.T)
                if None not in (self.sweepInstrument, self.sweepparams, self.sweepranges):
                    param_horz = getattr(self.sweepInstrument, self.sweepparams[0])
                    value_x = param_horz.get_latest()
                    value_y = self.sweepInstrument.get(self.sweepparams[1])
                    param_vert = getattr(self.sweepInstrument, self.sweepparams[1])
                    value_y = param_vert.get_latest()
                    self.horz_low = value_x - self.sweepranges[0] / 2
                    self.horz_range = self.sweepranges[0]
                    self.vert_low = value_y - self.sweepranges[1] / 2
                    self.vert_range = self.sweepranges[1]
                    self.rect = QtCore.QRect(self.horz_low, self.vert_low, self.horz_range, self.vert_range)
                    self.plot.setRect(self.rect)
        else:
            pass

        self.idx = self.idx + 1
        if self.idx > self.maxidx:
            self.idx = 0
            self.timer.stop()
        if processevents:
            QtWidgets.QApplication.processEvents()

    def updatebg(self):
        if self.idx % 10 == 0:
            logging.debug('livePlot: updatebg %d' % self.idx)
        self.idx = self.idx + 1
        self.fps.addtime(time.time())
        if self.datafunction is not None:
            try:
                # print(self.datafunction)
                dd = self.datafunction()
                self.update(data=dd)
            except Exception as e:
                logging.exception(e)
                print('livePlot: Exception in updatebg')
                print(e)
                self.stopreadout()
        else:
            self.stopreadout()
            dd = None
        time.sleep(0.00001)

    def startreadout(self, callback=None, rate=1000, maxidx=None):
        #$print('-- startreadout: callback: %s'  % callback)
        if maxidx is not None:
            self.maxidx = maxidx
        if callback is not None:
            self.datafunction = callback
        self.timer.start(1000 * (1. / rate))
        if self.verbose:
            print('live_plotting: start readout: rate %.1f Hz' % rate)

    def stopreadout(self):
        if self.verbose:
            print('live_plotting: stop readout')
        self.timer.stop()

if __name__ == '__main__':
    lp = livePlot(datafunction=MockCallback_2d(), sweepInstrument=None, sweepparams=['L', 'R'], sweepranges=[50, 50])
    lp.win.setGeometry(1500, 10, 400, 400)

#%% Some default callbacks


class MockCallback_2d:

    def __init__(self, npoints=6400):
        self.npoints = npoints

    def __call__(self):
        data = np.random.rand(self.npoints)
        data_reshaped = data.reshape(80, 80)
        return data_reshaped


class fpgaCallback_1d:

    def __init__(self, station, waveform, Naverage=4, fpga_ch=1):
        self.station = station
        self.waveform = waveform
        self.Naverage = Naverage
        self.fpga_ch = fpga_ch

    def __call__(self, verbose=0):
        ''' Callback function to read a single line of data from the FPGA '''
        ReadDevice = ['FPGA_ch%d' % self.fpga_ch]
        totalpoints, DataRead_ch1, DataRead_ch2 = self.station.fpga.readFPGA(
            Naverage=self.Naverage, ReadDevice=ReadDevice)

        if 'FPGA_ch1' in ReadDevice:
            data = DataRead_ch1
        elif 'FPGA_ch2' in ReadDevice:
            data = DataRead_ch2
        else:
            raise Exception('FPGA channel not well specified')

        data_processed = self.station.awg.sweep_process(
            data, self.waveform, self.Naverage)

        return data_processed


class fpgaCallback_2d:

    def __init__(self, station, waveform, Naverage, fpga_ch, resolution, diff_dir=None, waittime=0):
        self.station = station
        self.waveform = waveform
        self.Naverage = Naverage
        self.fpga_ch = fpga_ch
        self.resolution = resolution
        self.waittime = waittime
        self.diffsigma = 1
        self.diff = True
        self.diff_dir = diff_dir
        self.smoothing = False
        self.laplace = False

    def __call__(self):
        ''' Callback function to read a 2d scan of data from the FPGA '''
        ReadDevice = ['FPGA_ch%d' % self.fpga_ch]
        totalpoints, DataRead_ch1, DataRead_ch2 = self.station.fpga.readFPGA(
            ReadDevice=ReadDevice, Naverage=self.Naverage, waittime=self.waittime)

        if 'FPGA_ch1' in ReadDevice:
            data = DataRead_ch1
        elif 'FPGA_ch2' in ReadDevice:
            data = DataRead_ch2
        else:
            raise Exception('FPGA channel not well specified')

        im_diff = qtt.instrument_drivers.virtual_awg.sweep_2D_process(data, self.waveform, self.diff_dir)

        if self.smoothing:
            im_diff = qtt.algorithms.generic.smoothImage(im_diff)

        if self.laplace:
            im_diff = ndimage.filters.laplace(im_diff, mode='nearest')

        return np.array(im_diff)
