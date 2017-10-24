# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:55:21 2015

@author: tud205521
"""

#%% Load packages
import time
import logging
import numpy as np
from functools import partial
import qtpy.QtWidgets as QtWidgets
import qtpy.QtCore as QtCore
import scipy.ndimage as ndimage
import pyqtgraph as pg
import pyqtgraph
import pyqtgraph.multiprocess as mp

import qcodes

import qtt
from qtt import pgeometry as pmatlab

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

    def __init__(self, name='Measurement Control', rda_variable='qtt_abort_running_measurement', **kwargs):
        """ Simple control for real-time data parameters """
        super().__init__(**kwargs)
        w = self
        w.setWindowTitle(name)
        vbox = QtWidgets.QVBoxLayout()
        self.verbose = 0
        self.name = name
        self.rda_variable = rda_variable
        self.rda = rda_t()

        self.text = QtWidgets.QLabel()
        self.updateStatus()
        vbox.addWidget(self.text)

        self.abortbutton = QtWidgets.QPushButton()
        self.abortbutton.setText('Abort measurement')
        self.abortbutton.setStyleSheet("background-color: rgb(255,150,100);")
        self.abortbutton.clicked.connect(self.abort_measurements)
        vbox.addWidget(self.abortbutton)

        self.enable_button = QtWidgets.QPushButton()
        self.enable_button.setText('Enable measurements')
        self.enable_button.setStyleSheet("background-color: rgb(255,150,100);")
        self.enable_button.clicked.connect(self.enable_measurements)
        vbox.addWidget(self.enable_button)

        widget = QtWidgets.QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        w.resize(300, 300)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateStatus)  # this also works
        self.timer.start(1000)
        self.show()

    def updateStatus(self):
        if self.verbose >= 2:
            print('updateStatus...')
        value = int(self.rda.get(self.rda_variable, 0))
        self.text.setText('%s: %d' % (self.rda_variable, value))

    def enable_measurements(self):
        """ Enable measurements """
        if self.verbose:
            print('%s: setting %s to 0' % (self.name, self.rda_variable))
        self.rda.set(self.rda_variable, 0)
        self.updateStatus()

    def abort_measurements(self):
        """ Abort the current measurement """
        if self.verbose:
            print('%s: setting %s to 1' % (self.name, self.rda_variable))

        self.rda.set(self.rda_variable, 1)
        self.updateStatus()


if __name__ == '__main__' and 0:
    app = pg.mkQApp()

    mc = MeasurementControl()
    mc.verbose = 1
    mc.setGeometry(1700, 50, 300, 400)


def start_measurement_control(doexec=False):
    """ Start measurement control GUI

    Args:
        doexec(bool): if True run the event loop
    """
    #import warnings
    #from pyqtgraph.multiprocess.remoteproxy import RemoteExceptionWarning
    #warnings.simplefilter('ignore', RemoteExceptionWarning)
    proc = mp.QtProcess()
    lp = proc._import('qtt.live_plotting')
    mc = lp.MeasurementControl()

    qtt._dummy_mc = mc

    app = pyqtgraph.mkQApp()
    if doexec:
        app.exec()


#%%

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
            # do not emit signals when still editing
            dbox.setKeyboardTracking(False)

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
        # self.label.setStyleSheet("QLabel { background-color : #baccba;
        # margin: 2px; padding: 2px; }");

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
            # do not emit signals when still editing
            dbox.setKeyboardTracking(False)

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
        verbose (int): output level of logging information
        alpha (float): parameter (value between 0 and 1) which determines the weight given in averaging to the latest
                        measurement result (alpha) and the previous measurement result (1-alpha), default value 0.3
    """

    def __init__(self, datafunction=None, sweepInstrument=None, sweepparams=None, sweepranges=None, alpha=.3, verbose=1):
        """Return a new livePlot object."""      
        
        plotwin = pg.GraphicsWindow(title="Live view")

        win = QtWidgets.QWidget()
        win.resize(800, 600)
        win.setWindowTitle('livePlot')

        topLayout = QtWidgets.QHBoxLayout()
        win.start_button = QtWidgets.QPushButton('Start')
        win.stop_button = QtWidgets.QPushButton('Stop')
        win.averaging_box = QtWidgets.QCheckBox('Averaging')

        for b in [win.start_button, win.stop_button]:
            b.setMaximumHeight(24)

            #self.reloadbutton.setText('Reload data')
        topLayout.addWidget(win.start_button)
        topLayout.addWidget(win.stop_button)
        topLayout.addWidget(win.averaging_box)

        vertLayout = QtWidgets.QVBoxLayout()

        vertLayout.addLayout(topLayout)
        vertLayout.addWidget(plotwin)

        win.setLayout(vertLayout)

        self.win = win
        self.plotwin = plotwin
        self.verbose = verbose
        self.idx = 0
        self.maxidx = 1e9
        self.data = None
        self.sweepInstrument = sweepInstrument
        self.sweepparams = sweepparams
        self.sweepranges = sweepranges
        self.fps = pmatlab.fps_t(nn=6)
        self.datafunction = datafunction
        self._averaging_enabled = False
        
        self.datafunction_result = None
        self.alpha=alpha
        
        # TODO: allow arguments like ['P1']
        if self.sweepparams is None:
            p1 = plotwin.addPlot(title="Videomode")
            p1.setLabel('left', 'param1')
            p1.setLabel('left', 'param2')
            if self.datafunction is None:
                raise Exception(
                    'Either specify a datafunction or sweepparams.')
            else:
                data = np.array(self.datafunction())
                if data.ndim == 1:
                    dd = np.zeros((0,))
                    plot = p1.plot(dd, pen='b')
                    self.plot = plot
                else:
                    self.plot = pg.ImageItem()
                    p1.addItem(self.plot)
        elif isinstance(self.sweepparams, str):
            p1 = plotwin.addPlot(title="1d scan")
            p1.setLabel('left', 'Value')
            p1.setLabel('bottom', self.sweepparams, units='mV')
            dd = np.zeros((0,))
            plot = p1.plot(dd, pen='b')
            self.plot = plot
        elif type(self.sweepparams) is list:
            p1 = plotwin.addPlot(title='2d scan')
            if self.sweepparams[0] is dict:
                [xlabel, ylabel] = ['sweepparam_v', 'stepparam_v']
            else:
                [xlabel, ylabel] = self.sweepparams
            p1.setLabel('bottom', xlabel, units='mV')
            p1.setLabel('left', ylabel, units='mV')
            self.plot = pg.ImageItem()
            p1.addItem(self.plot)
        else:
            raise Exception(
                'The number of sweep parameters should be either None, 1 or 2.')

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updatebg)
        self.win.show()

        def connect_slot(target):
            """ Create a slot by dropping signal arguments """
            #@Slot()
            def signal_drop_arguments(*args, **kwargs):
                #print('call %s' % target)
                target()
            return signal_drop_arguments

        win.start_button.clicked.connect(connect_slot(self.startreadout))
        win.stop_button.clicked.connect(connect_slot(self.stopreadout))
        win.averaging_box.clicked.connect(connect_slot(self.enable_averaging))

        self.datafunction_result = None

    def close(self):
        if self.verbose:
            print('LivePlot.close()')
        self.stopreadout()
        self.win.close()

    def resetdata(self):
        self.idx = 0
        self.data = None

    def update(self, data=None, processevents=True):
        self.win.setWindowTitle('live view, fps: %.2f' % self.fps.framerate())
        if self.verbose >= 2:
            print('livePlot: update: idx %d ' % self.idx )
        if data is not None:
            self.data = np.array(data)
            if self.data.ndim == 1:
                if None in (self.sweepInstrument, self.sweepparams, self.sweepranges):
                    self.plot.setData(self.data)
                else:
                    sweep_param = getattr(
                        self.sweepInstrument, self.sweepparams)
                    paramval = sweep_param.get_latest()
                    sweepvalues = np.linspace(
                        paramval - self.sweepranges / 2, self.sweepranges / 2 + paramval, len(data))
                    self.plot.setData(sweepvalues, self.data)
            elif self.data.ndim == 2:
                self.plot.setImage(self.data.T)
                if None not in (self.sweepInstrument, self.sweepparams, self.sweepranges):
                    param_horz = getattr(
                        self.sweepInstrument, self.sweepparams[0])
                    value_x = param_horz.get_latest()
                    value_y = self.sweepInstrument.get(self.sweepparams[1])
                    param_vert = getattr(
                        self.sweepInstrument, self.sweepparams[1])
                    value_y = param_vert.get_latest()
                    self.horz_low = value_x - self.sweepranges[0] / 2
                    self.horz_range = self.sweepranges[0]
                    self.vert_low = value_y - self.sweepranges[1] / 2
                    self.vert_range = self.sweepranges[1]
                    self.rect = QtCore.QRect(
                        self.horz_low, self.vert_low, self.horz_range, self.vert_range)
                    self.plot.setRect(self.rect)
            else:
                raise Exception('ndim %d not supported' % self.data.ndim)

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

                if self.datafunction_result is None:
                    ddprev = dd
                else:
                    ddprev = self.datafunction_result
                
                # depending on value of self.averaging_enabled either do or don't do the averaging
                if self._averaging_enabled:
                    newdd = self.alpha*dd + (1-self.alpha)*ddprev
                else:
                    newdd = dd
            
                self.datafunction_result = newdd
                
                self.update(data=newdd)
            except Exception as e:
                logging.exception(e)
                print('livePlot: Exception in updatebg, stopping readout')
                self.stopreadout()
        else:
            self.stopreadout()
            dd = None

        if self.fps.framerate() < 10:
            #print('slow rate...?')
            time.sleep(0.1)
        time.sleep(0.00001)
        
    def enable_averaging(self, *args, **kwargs):
        
        self._averaging_enabled = self.win.averaging_box.checkState()
        if self.verbose>=1:
            if self._averaging_enabled == 2:
                print('enable_averaging called, alpha = '+str(self.alpha))
            elif self._averaging_enabled == 0:
                print('enable_averaging called, averaging turned off')
            else:
                print('enable_averaging called, undefined')
    
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
            print('live_plotting: start readout: rate %.1f Hz' % rate)

    def stopreadout(self):
        if self.verbose:
            print('live_plotting: stop readout')
        self.timer.stop()
        self.win.setWindowTitle('Live view stopped')

#%% Some default callbacks


class MockCallback_2d(qcodes.Instrument):

    def __init__(self, name, nx=80, **kwargs):
        super().__init__(name, **kwargs)

        self.nx = nx

        self.add_parameter(
            'p', parameter_class=qcodes.ManualParameter, initial_value=-20)
        self.add_parameter(
            'q', parameter_class=qcodes.ManualParameter, initial_value=30)

    def __call__(self):
        import qtt.deprecated.linetools as lt

        data = np.random.rand(self.nx * self.nx)
        data_reshaped = data.reshape(self.nx, self.nx)

        lt.semiLine(data_reshaped, [self.nx / 2, self.nx / 2],
                    np.deg2rad(self.p()), w=2, l=self.nx / 3, H=2)
        lt.semiLine(data_reshaped, [self.nx / 2, self.nx / 2],
                    np.deg2rad(self.q()), w=2, l=self.nx / 4, H=3)

        return data_reshaped


def test_mock2d():
    m = MockCallback_2d(qtt.measurements.scans.instrumentName('dummy2d') )
    d = m()


#%% Example
if __name__ == '__main__':
    lp = livePlot(datafunction=MockCallback_2d(qtt.measurements.scans.instrumentName('mock')), sweepInstrument=None,
                  sweepparams=['L', 'R'], sweepranges=[50, 50])
    lp.win.setGeometry(1500, 10, 400, 400)
    lp.startreadout()
    pv = qtt.createParameterWidget([lp.datafunction])
