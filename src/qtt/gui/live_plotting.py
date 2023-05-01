# %% Load packages
import logging
import time
from functools import partial
from typing import Any, Optional

import numpy as np
import pyqtgraph
import pyqtgraph as pg
import pyqtgraph.multiprocess as mp
import qcodes
import qtpy.QtCore as QtCore
import qtpy.QtWidgets as QtWidgets
from qtpy.QtCore import Signal

import qtt
import qtt.algorithms.generic
from qtt import pgeometry

# %% Static variables

liveplotwindow = None  # global handle for live plotting


def getLivePlotWindow():
    """ Return handle to live plotting window

    Returns:
        None or object: handle to live plotting window
    """
    global liveplotwindow
    if liveplotwindow is not None:
        return liveplotwindow
    return None

# %% Communication


try:
    import redis
except BaseException:
    pass


class rda_t:

    def __init__(self, host: str = '127.0.0.1', port: int = 6379, password: Optional[str] = None):
        """ Class for simple real-time data access

        Every object has a `get` and `set` method to access simple parameters
        globally (e.g. across different python sessions).

        Args:
            host: Hostname for connection
            port: Port for connection
            password: Password for connection

        """

        # we use redis as backend now
        self.r = redis.Redis(host=host, port=port, password=password)
        try:
            self.set('dummy_rda_t', -3141592)
            v = self.get_float('dummy_rda_t')
            if not v == -3141592:
                raise Exception('set not equal to get')
        except redis.exceptions.ConnectionError as e:
            print('rda_t: check whether redis is installed and the server is running')
            raise e

    def get_float(self, key: str, default_value: Optional[float] = None) -> Optional[float]:
        """ Get value by key and convert to a float

        Args:
            key: Key to retrieve
            default_value: Value to return when the key is not present

        Returns:
            Value retrieved
        """
        v = self.get(key, default_value)
        if v is None:
            return v
        return float(v)

    def get_int(self, key: str, default_value: Optional[int] = None) -> Optional[int]:
        """ Get value by key and convert to an int

        Args:
            key: Key to retrieve
            default_value: Value to return when the key is not present

        Returns:
            Value retrieved
        """
        v = self.get(key, default_value)
        if v is None:
            return v
        return int(float(v))

    def get(self, key: str, default_value: Optional[Any] = None) -> Optional[Any]:
        """ Get a value

        Args:
            key: value to be retrieved
            default_value: value to return if the key is not present
        Returns:
            Value retrieved
        """
        value = self.r.get(key)
        if value is None:
            return default_value
        else:
            return value

    def set(self, key: str, value: Any):
        """ Set a value

        Args:
            key: key
            value: the value to be set
        """
        self.r.set(key, value)


class MeasurementControl(QtWidgets.QMainWindow):
    """ Simple control for running measurements """

    def __init__(self, name='Measurement Control',
                 rda_variable='qtt_abort_running_measurement', text_vars=[],
                 **kwargs):
        """ Simple control for running measurements

        Args:
            name (str): used as window title
            rda_variable (str):
            text_vars (list):

        """
        super().__init__(**kwargs)
        w = self
        w.setWindowTitle(name)
        vbox = QtWidgets.QVBoxLayout()
        self.verbose = 0
        self.name = name
        self.rda_variable = rda_variable
        self.rda = rda_t()
        self.text_vars = text_vars
        if len(text_vars) > 0:
            self.vLabels = {}
            self.vEdits = {}
            self.vButtons = {}
            self.vActions = []
            for tv in text_vars:
                self.vLabels[tv] = QtWidgets.QLabel()
                self.vLabels[tv].setText('%s:' % tv)
                self.vEdits[tv] = (QtWidgets.QTextEdit())
                try:
                    self.vEdits[tv].setText(
                        self.rda.get(tv, b'').decode('utf-8'))
                except Exception as Ex:
                    print('could not retrieve value %s: %s' % (tv, str(Ex)))
                self.vButtons[tv] = QtWidgets.QPushButton()
                self.vButtons[tv].setText('Send')
                self.vButtons[tv].setStyleSheet(
                    "background-color: rgb(255,150,100);")
                self.vButtons[tv].clicked.connect(partial(self.sendVal, tv))
                vbox.addWidget(self.vLabels[tv])
                vbox.addWidget(self.vEdits[tv])
                vbox.addWidget(self.vButtons[tv])
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

        menuBar = self.menuBar()

        menuDict = {
            '&Edit': {'&Get all Values': self.getAllValues},
            '&Help': {'&Info': self.showHelpBox}
        }
        for (k, menu) in menuDict.items():
            mb = menuBar.addMenu(k)
            for (kk, action) in menu.items():

                act = QtWidgets.QAction(kk, self)
                mb.addAction(act)
                act.triggered.connect(action)
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

    def sendVal(self, tv):
        """ send text value """
        print('sending value %s' % tv)
        self.rda.set(tv, self.vEdits[tv].toPlainText())

    def getVal(self, tv):
        """ get text value """
        value = self.rda.get(tv, b'').decode('utf-8')
        self.vEdits[tv].setText(value)

    def showHelpBox(self):
        """ Show help dialog """
        self.infotext = "This widget is used for live control of your measurement via inter-process communication.\
                        <br/><br/>To add additional variables (str) to the control use the text_vars argmument. \
                        To access values, use the <code>qtt.redisvalue</code> method."
        QtWidgets.QMessageBox.information(self, 'qtt measurement control info', self.infotext)

    def getAllValues(self):
        """ get all string values """
        for tv in self.text_vars:
            self.getVal(tv)


def start_measurement_control(doexec=False):
    """ Start measurement control GUI

    Args:
        doexec(bool): if True run the event loop
    """
    _ = pyqtgraph.mkQApp()
    # import warnings
    # from pyqtgraph.multiprocess.remoteproxy import RemoteExceptionWarning
    # warnings.simplefilter('ignore', RemoteExceptionWarning)
    proc = mp.QtProcess()
    lp = proc._import('qtt.gui.live_plotting')
    mc = lp.MeasurementControl()
    app = pyqtgraph.mkQApp()
    if doexec:
        app.exec()


try:
    from qcodes_loop.plots.pyqtgraph import QtPlot

    import qtt.gui
    import qtt.gui.parameterviewer
    from qtt.utilities.tools import monitorSizes

    def setupMeasurementWindows(station=None, create_parameter_widget=True,
                                ilist=None, qtplot_remote=True):
        """
        Create liveplot window and parameter widget (optional)

        Args:
            station (QCoDeS station): station with instruments
            create_parameter_widget (bool): if True create ParameterWidget
            ilist (None or list): list of instruments to add to ParameterWidget
            qtplot_remote (bool): If True, then use remote plotting
        Returns:
            dict: created gui objects
        """
        windows = {}

        ms = monitorSizes()
        vv = ms[-1]

        if create_parameter_widget and any([station, ilist]):
            if ilist is None:
                ilist = [station.gates]
            w = qtt.createParameterWidget(ilist)
            w.setGeometry(vv[0] + vv[2] - 400 - 300, vv[1], 300, 600)
            windows['parameterviewer'] = w

        plotQ = QtPlot(window_title='Live plot', interval=.5, remote=qtplot_remote)
        plotQ.setGeometry(vv[0] + vv[2] - 600, vv[1] + vv[3] - 400, 600, 400)
        plotQ.update()

        qtt.gui.live_plotting.liveplotwindow = plotQ

        windows['plotwindow'] = plotQ

        app = QtWidgets.QApplication.instance()
        app.processEvents()

        return windows

except Exception as ex:
    logging.exception(ex)
    print('failed to add setupMeasurementWindows!')
    pass


# %%

class RdaControl(QtWidgets.QMainWindow):

    def __init__(self, name='LivePlot Control', boxes=[
                 'xrange', 'yrange', 'nx', 'ny'], **kwargs):
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
        for _, b in enumerate(self.boxes):
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
        for _, b in enumerate(self.boxes):
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


# legacy name
LivePlotControl = RdaControl

# %% Liveplot object


class livePlot(QtCore.QObject):
    """ Class to enable live plotting of data.

    Attributes:
        datafunction: the function to call for data acquisition
        sweepInstrument: the instrument to which sweepparams belong
        sweepparams: the parameter(s) being swept
        sweepranges (list): the ranges over which sweepparams are being swept
        verbose (int): output level of logging information
        show_controls (bool): show gui elements for control of the live plotting
        alpha (float): parameter (value between 0 and 1) which determines the weight given in averaging to the latest
                        measurement result (alpha) and the previous measurement result (1-alpha), default value 0.3
    """

    sigMouseClicked = Signal(object)

    def __init__(
            self,
            datafunction=None,
            sweepInstrument=None,
            sweepparams=None,
            sweepranges=None,
            alpha=.3,
            verbose=1,
            show_controls=True,
            window_title='live view',
            plot_dimension=None,
            plot_title=None,
            is1dscan=None, **kwargs):
        """Return a new livePlot object."""
        super().__init__(**kwargs)

        self.window_title = window_title
        win = QtWidgets.QWidget()
        win.resize(800, 600)
        win.setWindowTitle(self.window_title)
        vertLayout = QtWidgets.QVBoxLayout()

        self._averaging_enabled = 2
        if show_controls:
            topLayout = QtWidgets.QHBoxLayout()
            win.start_button = QtWidgets.QPushButton('Start')
            win.stop_button = QtWidgets.QPushButton('Stop')
            win.averaging_box = QtWidgets.QCheckBox('Averaging')
            win.averaging_box.setChecked(self._averaging_enabled)
            for b in [win.start_button, win.stop_button]:
                b.setMaximumHeight(24)
            topLayout.addWidget(win.start_button)
            topLayout.addWidget(win.stop_button)
            topLayout.addWidget(win.averaging_box)
            vertLayout.addLayout(topLayout)
        plotwin = pg.GraphicsLayoutWidget(title="Live view")
        vertLayout.addWidget(plotwin)
        win.setLayout(vertLayout)
        self.setGeometry = win.setGeometry
        self.win = win
        self.plotwin = plotwin
        self.verbose = verbose
        self.idx = 0
        self.maxidx = 1e9
        self.data = None
        self.data_avg = None
        self.sweepInstrument = sweepInstrument
        self.sweepparams = sweepparams
        self.sweepranges = sweepranges
        self.fps = pgeometry.fps_t(nn=6)
        self.datafunction = datafunction
        self.datafunction_result = None
        self.plot_dimension = plot_dimension
        self.alpha = alpha
        if is1dscan is None:
            is1dscan = (
                isinstance(
                    self.sweepparams, str) or (
                    isinstance(
                        self.sweepparams, (list, dict)) and len(
                        self.sweepparams) == 1))
            if isinstance(self.sweepparams, dict):
                if 'gates_horz' not in self.sweepparams:
                    is1dscan = True
        if verbose:
            print('live_plotting: is1dscan %s' % is1dscan)
        if self.sweepparams is None:
            p1 = plotwin.addPlot(title=plot_title)
            p1.setLabel('left', 'param2')
            p1.setLabel('bottom', 'param1')
            if plot_dimension == 1:
                dd = np.zeros((0,))
                plot = p1.plot(dd, pen='b')
                self.plot = plot
            else:
                self.plot = pg.ImageItem()
                p1.addItem(self.plot)
            self._crosshair = []
        elif is1dscan:
            p1 = plotwin.addPlot(title=plot_title)
            p1.setLabel('left', 'Value')
            p1.setLabel('bottom', self.sweepparams, units='mV')
            dd = np.zeros((0,))
            plot = p1.plot(dd, pen='b')
            self.plot = plot
            vpen = pg.QtGui.QPen(pg.QtGui.QColor(
                130, 130, 175, 60), 0, pg.QtCore.Qt.SolidLine)
            gv = pg.InfiniteLine([0, 0], angle=90, pen=vpen)
            gv.setZValue(0)
            p1.addItem(gv)
            self._crosshair = [gv]
            self.crosshair(show=False)
        elif isinstance(self.sweepparams, (list, dict)):
            # 2D scan
            p1 = plotwin.addPlot(title=plot_title)
            if type(self.sweepparams) is dict:
                [xlabel, ylabel] = ['sweepparam_v', 'stepparam_v']
            else:
                [xlabel, ylabel] = self.sweepparams
            p1.setLabel('bottom', xlabel, units='mV')
            p1.setLabel('left', ylabel, units='mV')
            self.plot = pg.ImageItem()
            p1.addItem(self.plot)
            vpen = pg.QtGui.QPen(pg.QtGui.QColor(
                0, 130, 235, 60), 0, pg.QtCore.Qt.SolidLine)
            gh = pg.InfiniteLine([0, 0], angle=90, pen=vpen)
            gv = pg.InfiniteLine([0, 0], angle=0, pen=vpen)
            gh.setZValue(0)
            gv.setZValue(0)
            p1.addItem(gh)
            p1.addItem(gv)
            self._crosshair = [gh, gv]
            self.crosshair(show=False)
        else:
            raise Exception(
                'The number of sweep parameters should be either None, 1 or 2.')
        self.plothandle = p1
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updatebg)
        self.win.show()

        def connect_slot(target):
            """ Create a slot by dropping signal arguments """
            def signal_drop_arguments(*args, **kwargs):
                target()
            return signal_drop_arguments

        if show_controls:
            win.start_button.clicked.connect(connect_slot(self.startreadout))
            win.stop_button.clicked.connect(connect_slot(self.stopreadout))
            win.averaging_box.clicked.connect(
                connect_slot(self.enable_averaging_slot))

        self.datafunction_result = None

        self.plotwin.scene().sigMouseClicked.connect(self._onClick)

    def __del__(self):
        self.stopreadout()
        pyqtgraph.mkQApp().processEvents()
        self.close()
        parent = super()
        if hasattr(parent, '__del__'):
            parent.__del__()

    def _onClick(self, event):
        image_pt = self.plot.mapFromScene(event.scenePos())

        tr = self.plot.transform()
        pt = tr.map(image_pt.x(), image_pt.y())
        if self.verbose >= 2:
            print('pt %s' % (pt,))
        self.sigMouseClicked.emit(pt)

    def close(self):
        if self.verbose:
            print('LivePlot.close()')
        self.stopreadout()
        pyqtgraph.mkQApp().processEvents()
        self.win.close()

    def crosshair(self, show=None, pos=None):
        """ Enable or disable crosshair

        Args:
            show (None, True or False)
            pos (None or position)
        """
        for x in self._crosshair:
            if show is not None:
                if show:
                    x.show()
                else:
                    x.hide()
            if pos is not None:
                x.setPos(pos)

    def update(self, data=None, processevents=True):
        self.win.setWindowTitle('%s, fps: %.2f' %
                                (self.window_title, self.fps.framerate()))
        if self.verbose >= 2:
            print('livePlot: update: idx %d ' % self.idx)
        if data is not None:
            self.data = np.array(data)
            if self.data_avg is None:
                self.data_avg = self.data
            # depending on value of self.averaging_enabled either do or
            # don't do the averaging
            if self._averaging_enabled:
                self.data_avg = self.alpha * self.data + \
                    (1 - self.alpha) * self.data_avg
            else:
                self.data_avg = self.data
            if self.data.ndim == 1:
                if None in (self.sweepInstrument, self.sweepparams,
                            self.sweepranges):
                    sweepvalues = np.arange(0, self.data_avg.size)
                    self.plot.setData(sweepvalues, self.data_avg)
                else:
                    if type(self.sweepparams) is dict:
                        paramval = 0
                    else:
                        sweep_param = getattr(
                            self.sweepInstrument, self.sweepparams)
                        paramval = sweep_param.get_latest()
                    sweepvalues = np.linspace(
                        paramval - self.sweepranges[0] / 2,
                        self.sweepranges[0] / 2 + paramval,
                        len(data))
                    self.plot.setData(sweepvalues, self.data_avg)
                    self._sweepvalues = [sweepvalues]
                    self.crosshair(show=None, pos=[paramval, 0])
            elif self.data.ndim == 2:
                self.plot.setImage(self.data_avg.T)
                if None not in (self.sweepInstrument,
                                self.sweepparams, self.sweepranges):
                    if isinstance(self.sweepparams, dict):
                        value_x = 0
                        value_y = 0
                    else:
                        if isinstance(self.sweepparams[0], dict):
                            value_x = 0
                            value_y = 0
                        else:
                            value_x = self.sweepInstrument.get(
                                self.sweepparams[0])
                            value_y = self.sweepInstrument.get(
                                self.sweepparams[1])
                    self.horz_low = value_x - self.sweepranges[0] / 2
                    self.horz_range = self.sweepranges[0]
                    self.vert_low = value_y - self.sweepranges[1] / 2
                    self.vert_range = self.sweepranges[1]
                    self.rect = QtCore.QRect(
                        int(self.horz_low), int(self.vert_low), int(self.horz_range), int(self.vert_range))
                    self.plot.setRect(self.rect)
                    self.crosshair(show=None, pos=[value_x, value_y])
                    self._sweepvalues = [
                        np.linspace(
                            self.horz_low,
                            self.horz_low
                            + self.horz_range,
                            self.data.shape[1]),
                        np.linspace(
                            self.vert_low,
                            self.vert_low
                            + self.vert_range,
                            self.data.shape[0])]
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
        """ Update function for the widget

        Calls the datafunction() and update() function
        """
        if self.idx % 10 == 0:
            logging.debug('livePlot: updatebg %d' % self.idx)
        self.idx = self.idx + 1
        self.fps.addtime(time.time())
        if self.datafunction is not None:
            try:
                dd = self.datafunction()
                self.datafunction_result = dd
                self.update(data=dd)
            except Exception as e:
                logging.exception(e)
                print('livePlot: Exception in updatebg, stopping readout')
                self.stopreadout()
        else:
            self.stopreadout()
            dd = None
        if self.fps.framerate() < 10:
            time.sleep(0.1)
        time.sleep(0.00001)

    def enable_averaging(self, value):
        """ Enabling rolling average """
        self._averaging_enabled = value
        if self.verbose >= 1:
            if self._averaging_enabled == 2:
                if self.verbose:
                    print('enable_averaging called, alpha = ' + str(self.alpha))
            elif self._averaging_enabled == 0:
                if self.verbose:
                    print('enable_averaging called, averaging turned off')
            else:
                if self.verbose:
                    print('enable_averaging called, undefined: value %s' % (self._averaging_enabled,))

    def enable_averaging_slot(self, *args, **kwargs):
        """ Update the averaging mode of the widget """
        self._averaging_enabled = self.win.averaging_box.checkState()
        self.enable_averaging(self._averaging_enabled)

    def startreadout(self, callback=None, rate=30, maxidx=None):
        """
        Args:
            callback (None or method): Method to call on update
            rate (float): sample rate in ms
            maxidx (None or int): Stop reading if the index is larger than the maxidx
        """
        if maxidx is not None:
            self.maxidx = maxidx
        if callback is not None:
            self.datafunction = callback
        self.timer.start(int(1000 * (1. / rate)))
        if self.verbose:
            print('live_plotting: start readout: rate %.1f Hz' % rate)

    def stopreadout(self):
        """ Stop the readout loop """
        if self.verbose:
            print('live_plotting: stop readout')
        self.timer.stop()
        self.win.setWindowTitle('Live view stopped')

# %% Some default callbacks


class MockCallback_2d(qcodes.Instrument):

    def __init__(self, name, nx=80, **kwargs):
        super().__init__(name, **kwargs)
        self.nx = nx
        self.add_parameter('p', parameter_class=qcodes.ManualParameter, initial_value=-20)
        self.add_parameter('q', parameter_class=qcodes.ManualParameter, initial_value=30)

    def __call__(self):
        import qtt.utilities.imagetools as lt
        data = np.random.rand(self.nx * self.nx)
        data_reshaped = data.reshape(self.nx, self.nx)
        lt.semiLine(data_reshaped, [self.nx / 2, self.nx / 2],
                    np.deg2rad(self.p()), w=2, l=self.nx / 3, H=2)
        lt.semiLine(data_reshaped, [self.nx / 2, self.nx / 2],
                    np.deg2rad(self.q()), w=2, l=self.nx / 4, H=3)
        return data_reshaped
