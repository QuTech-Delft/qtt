"""
Contains code for viewing Parameters in a gui window
"""
import time
import threading
import logging
import numpy as np
import sys
from typing import Sequence, Any, Optional
import multiprocessing as mp
from functools import partial

from qtpy import QtWidgets
from qtpy.QtCore import Signal, Slot, QSize
import pyqtgraph
from qtt import pgeometry


# %%


class QCodesTimer(threading.Thread):

    def __init__(self, callback_function, dt=2, **kwargs):
        """ Simple timer to perform periodic execution of function """
        super().__init__(**kwargs)
        self.callback_function = callback_function
        self.dt = dt
        self._run = None

    def run(self):
        self._run = True
        while self._run:
            logging.debug('QCodesTimer: start sleep')
            time.sleep(self.dt)
            logging.debug('QCodesTimer: execute callback function')
            self.callback_function()

    def stop(self):
        self._run = False


def isfloat(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


class ParameterViewer(QtWidgets.QTreeWidget):
    """ Simple class to show qcodes parameters """

    update_field_signal = Signal(str, str, str, object, bool)
    _create_gui_signal = Signal()

    def __init__(self, instruments: list, name: str = 'QuTech Parameter Viewer',
                 fields: Sequence[str] = ('Value', 'unit'), **kwargs):
        """ Simple class to show qcodes parameters

        Args:
            instruments (list): list of Qcodes Instruments to show
            name: String used in the window title
            fields: Names of Parameter fields to show
        """
        super().__init__(**kwargs)
        self.verbose = 1
        self._fields = fields
        self._update_counter = 0
        self._timer: Optional[QCodesTimer] = None
        self.update_field_signal.connect(self._set_field)  # type: ignore
        self._create_gui_signal.connect(self._create_gui)  # type: ignore
        self._itemsdict: dict = dict()
        self._default_sizehints = [200, 140]

        window = self
        window.setGeometry(1700, 50, 300, 600)
        window.setColumnCount(2 + len(self._fields))
        self._tree_header = QtWidgets.QTreeWidgetItem(['Parameter'] + list(self._fields))
        window.setHeaderItem(self._tree_header)
        self.set_window_name(name)

        self.callbacklist: Sequence = []

        self.initialize_viewer(instruments)

        self.updatecallback()
        self.show()

    def set_window_name(self, name):
        self.name = name
        self.setWindowTitle(name)

    def initialize_viewer(self, instruments):
        self._clear_gui()
        self._init(instruments)
        self._create_gui_signal.emit()
        self.updatecallback()

    def _init(self, instruments):
        """ Initialize parameter viewer """
        instrumentnames = [i.name for i in instruments]
        self._instruments = instruments
        self._instrumentnames = instrumentnames
        for instrument_name in instrumentnames:
            self._itemsdict[instrument_name] = dict()

    def close(self):
        self.stop()
        super(ParameterViewer, self).close()

    def _clear_gui(self):

        instrument_names = list(self._itemsdict.keys())
        for _, instrument_name in enumerate(instrument_names):
            lst = self._itemsdict[instrument_name]
            gatesroot = self._itemsdict[instrument_name]['_treewidgetitem']

            params = [param for param in lst if not param.startswith('_')]
            for parameter_name in params:
                widget = self._itemsdict[instrument_name][parameter_name]['widget']
                gatesroot.removeChild(widget)

            self.takeTopLevelItem(0)

        self._itemsdict = {}

    @Slot()
    def _create_gui(self):
        """ Initialize parameter viewer GUI

        This function creates all the GUI elements.
        """
        for ii, instrument_name in enumerate(self._instrumentnames):
            instr = self._instruments[ii]
            parameters = instr.parameters
            parameter_names = sorted(instr.parameters.keys())

            parameter_names = [p for p in parameter_names if hasattr(
                instr.parameters[p], 'get')]
            gatesroot = QtWidgets.QTreeWidgetItem(self, [instrument_name])
            self._itemsdict[instrument_name]['_treewidgetitem'] = gatesroot

            for parameter_name in parameter_names:
                # hack to make this semi thread-safe
                si = min(sys.getswitchinterval(), 0.1)
                # hack to make this semi thread-safe
                sys.setswitchinterval(100)
                sys.setswitchinterval(si)  # hack to make this semi thread-safe
                box = QtWidgets.QDoubleSpinBox()
                # do not emit signals when still editing
                box.setKeyboardTracking(False)

                initial_values = [parameter_name] + [''] * len(self._fields)
                widget = QtWidgets.QTreeWidgetItem(gatesroot, initial_values)

                self._itemsdict[instrument_name][parameter_name] = {'widget': widget, 'double_box': None}

                px = parameters[parameter_name].get()
                if hasattr(parameters[parameter_name], 'set') and isfloat(px):
                    self.setItemWidget(widget, 1, box)
                    self._itemsdict[instrument_name][parameter_name]['double_box'] = box

                box.valueChanged.connect(partial(self._valueChanged, instrument_name, parameter_name))

        self.set_column_sizehints(self._default_sizehints)
        self.set_parameter_properties(step_size=5, minimum_value=-1e20, maximum_value=1e20)
        self.setSortingEnabled(True)
        self.expandAll()

    def set_column_sizehints(self, size_hints):
        for ii, instrument_name in enumerate(self._instrumentnames):
            lst = self._itemsdict[instrument_name]

            params = [param for param in lst if not param.startswith('_')]
            for parameter_name in params:
                widget = self._itemsdict[instrument_name][parameter_name]['widget']
                double_box = self._itemsdict[instrument_name][parameter_name]['double_box']
                for column, hint in enumerate(size_hints):
                    if double_box is None:
                        widget.setSizeHint(column, QSize(hint, 20))
                    else:
                        widget.setSizeHint(column, QSize(hint, -1))
        for column in range(len(size_hints)):
            self.resizeColumnToContents(column)

    def set_parameter_properties(self, minimum_value=None, maximum_value=None, step_size=None):
        """ Set properties of the parameter viewer widget elements """
        for _, instrument_name in enumerate(self._instrumentnames):
            lst = self._itemsdict[instrument_name]

            params = [param for param in lst if not param.startswith('_')]
            for parameter_name in params:
                box = lst[parameter_name]['double_box']

                if box is not None:
                    if step_size is not None:
                        box.setSingleStep(step_size)
                    if minimum_value is not None:
                        box.setMinimum(minimum_value)
                    if maximum_value is not None:
                        box.setMaximum(maximum_value)

    def is_running(self):
        if self._timer is None:
            return False
        if self._timer.is_alive():
            return True
        else:
            return False

    def setParamSingleStep(self, instr: str, param: str, value: Any):
        """ Set the default step size for a parameter in the viewer

        Args:
            instr (str): instrument
            param (str): parameter of the instrument
            value (float): step size
        """
        box = self._itemsdict[instr][param]['double_box']
        if box is not None:
            box.setSingleStep(value)

    def setSingleStep(self, value: float, instrument_name: Optional[str] = None):
        """ Set the default step size for all parameters in the viewer

        Args:
            value (float): step size
        """
        if instrument_name is None:
            names = self._instrumentnames
        else:
            names = [instrument_name]
        for instrument_name in names:
            lst = self._itemsdict[instrument_name]
            for parameter_name in lst:
                if parameter_name == '_treewidgetitem':
                    continue
                box = lst[parameter_name]['double_box']

                if box is not None:
                    box.setSingleStep(value)

    def _valueChanged(self, instrument_name: str, parameter_name: str, value: Any, *args, **kwargs):
        """ Callback used to update values in an instrument """
        instrument = self._instruments[self._instrumentnames.index(instrument_name)]
        logging.info('set %s.%s to %s' % (instrument_name, parameter_name, value))
        instrument.set(parameter_name, value)

    def updatecallback(self, start: bool = True, dt: float = 3):
        """ Update the data and restarts timer """
        if self._timer is not None:
            self._timer.stop()
            del self._timer

        self.updatedata()

        if start:
            self._timer = QCodesTimer(callback_function=self.updatedata, dt=dt)
            self._timer.start()
        else:
            self._timer = None

    def stop(self):
        """ Stop readout of the parameters in the widget """
        self.setWindowTitle(self.name + ': stopped')
        self._timer.stop()

    @Slot(str, str, str, object, bool)
    def _set_field(self, instrument_name, parameter_name, field, value, force_update):
        """ Helper function

        Update field of parameter viewer with a string value
        """
        if self.verbose >= 2:
            print('_set_field: %s %s: %s' % (instrument_name, parameter_name, str(value)))
        tree_widget = self._itemsdict[instrument_name][parameter_name]['widget']
        double_box = self._itemsdict[instrument_name][parameter_name]['double_box']

        field_index = self._fields.index(field)

        double_value = False
        if field_index == 0 and double_box is not None:
            double_value = True
        if not double_value:
            tree_widget.setText(field_index + 1, str(value))
        else:
            # update a float value
            try:
                update_value = np.abs(tree_widget.value() - value) > 1e-9
            except Exception as ex:
                logging.debug(ex)
                update_value = True
            if update_value or force_update:
                if not double_box.hasFocus():  # do not update when editing
                    logging.debug('update %s to %s' % (parameter_name, value))
                    try:
                        oldstate = double_box.blockSignals(True)
                        double_box.setValue(value)
                        double_box.blockSignals(oldstate)
                    except Exception as ex:
                        logging.debug(ex)

    def updatedata(self, force_update=False):
        """ Update data in viewer using station.snapshow """

        self._update_counter = self._update_counter + 1
        logging.debug('ParameterViewer: update values')
        for instrument_name in self._instrumentnames:
            instr = self._instruments[self._instrumentnames.index(instrument_name)]
            parameters = {}

            try:
                parameters = instr.parameters
            except AttributeError as ex:
                # instrument was removed
                print('instrument was removed, stopping ParameterViewer')
                # logging.exception(ex)
                self._timer.stop()

            parameter_names = sorted(parameters.keys())

            si = sys.getswitchinterval()

            for parameter_name in parameter_names:
                # hack to make this semi thread-safe

                for field_name in self._fields:
                    if field_name == 'Value':
                        sys.setswitchinterval(100)
                        value = parameters[parameter_name].get_latest()
                        sys.setswitchinterval(si)
                        self.update_field_signal.emit(instrument_name, parameter_name, field_name, value, force_update)
                    else:
                        if self._update_counter % 20 == 1 or 1:
                            sys.setswitchinterval(100)
                            value = getattr(parameters[parameter_name], field_name, '')
                            sys.setswitchinterval(si)
                            self.update_field_signal.emit(instrument_name, parameter_name,
                                                          field_name, value, force_update)

        for callback_function in self.callbacklist:
            try:
                callback_function()
            except Exception as ex:
                logging.debug('update function failed')
                logging.exception(str(ex))


def createParameterWidgetRemote(instruments, doexec=True):
    """ Create a parameter widget in a remote process.

    Note: this can only be used if all the Instruments are remote instruments.
    """
    p = mp.Process(target=createParameterWidget, args=(instruments,))
    p.start()
    return p


def createParameterWidget(instruments, doexec=False, remote=False):
    """ Create a parameter widget

    Args:
        instruments (list)
        doexec (bool): if True execute as a standalone Qt app
        remote (bool): if True, then start in a remote process.
                       Note: this can only be used if all the Instruments are remote instruments.
    """
    if remote:
        p = mp.Process(target=createParameterWidget,
                       args=(instruments, doexec))
        p.start()
        return p

    app = pyqtgraph.mkQApp()

    ms = pgeometry.monitorSizes()[-1]
    p = ParameterViewer(instruments=instruments)
    p.setGeometry(ms[0] + ms[2] - 320, 30, 300, 600)
    p.show()
    p.updatecallback()

    logging.info('created update widget...')

    if doexec:
        app.exec()
    return p
