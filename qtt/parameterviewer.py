"""
Contains code for viewing Parameters in a separate gui
"""
import time
import threading
import logging
import numpy as np
import sys

import multiprocessing as mp

from qtpy.QtCore import Qt
from qtpy import QtWidgets
from qtpy import QtGui
from qtpy.QtCore import Signal, Slot
import pyqtgraph
from qtt import pgeometry as pmatlab
from functools import partial

#%%

class QCodesTimer(threading.Thread):

    def __init__(self, fn, dt=2, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.dt = dt

    def run(self):
        while 1:
            logging.debug('QCodesTimer: start sleep')
            time.sleep(self.dt)
            # do something
            logging.debug('QCodesTimer: run!')
            self.fn()


class ParameterViewer(QtWidgets.QTreeWidget):
    """ Simple class to show qcodes parameters """

    shared_kwargs = ['station', 'instrumentnames']

    def __init__(self, instruments, instrumentnames=None,
                 name='QuTech Parameter Viewer', **kwargs):
        """ Simple class to show qcodes parameters

        Args:
            instruments (list): list of Qcodes Instruments to show
            instrumentnames (None or list): optional list of names to show
            name (str, optional): string used in the window title

        """
        super().__init__(**kwargs)
        w = self
        w.setGeometry(1700, 50, 300, 600)
        w.setColumnCount(3)
        self.verbose = 1
        header = QtWidgets.QTreeWidgetItem(["Parameter", "Value"])
        w.setHeaderItem(header)
        w.setWindowTitle(name)

        if instrumentnames is None:
            instrumentnames = [i.name for i in instruments]
        self._instruments = instruments
        self._instrumentnames = instrumentnames
        self._itemsdict = dict()
        for i in instrumentnames:
            self._itemsdict[i] = dict()
        self._timer = None
        self.init()
        self.show()

        self.callbacklist = []

        self.update_field.connect(self._set_field)

    def init(self):
        """ Initialize parameter viewer

        This function created all the GUI elements.
        """
        for ii, iname in enumerate(self._instrumentnames):
            instr = self._instruments[ii]
            pp = instr.parameters
            ppnames = sorted(instr.parameters.keys())

            ppnames = [p for p in ppnames if instr.parameters[p].has_get]
            gatesroot = QtWidgets.QTreeWidgetItem(self, [iname])
            for g in ppnames:
                # ww=['gates', g]
                si = min(sys.getswitchinterval(), 0.1)  # hack to make this semi thread-safe
                sys.setswitchinterval(100)  # hack to make this semi thread-safe
                sys.setswitchinterval(si)  # hack to make this semi thread-safe
                box = QtWidgets.QDoubleSpinBox()
                box.setKeyboardTracking(False)  # do not emit signals when still editing
                box.setMinimum(-10000)
                box.setMaximum(10000)
                box.setSingleStep(5)

                v = ''
                A = QtWidgets.QTreeWidgetItem(gatesroot, [g, v])
                self._itemsdict[iname][g] = A

                if pp[g].has_set:
                    qq = A
                    self.setItemWidget(qq, 1, box)
                    self._itemsdict[iname][g] = box

                box.valueChanged.connect(partial(self.valueChanged, iname, g))

        self.setSortingEnabled(True)
        self.expandAll()
        # self.label.setStyleSheet("QLabel { background-color : #baccba; margin: 2px; padding: 2px; }");

    def setSingleStep(self, value, instrument_name=None):
        """ Set the default step size for parameters in the viewer """
        if instrument_name is None:
            names = self._instrumentnames
        else:
            names = [instrument_name]
        for iname in names:
            lst = self._itemsdict[iname]
            for p in lst:
                box = lst[p]
                try:
                    box.setSingleStep(value)
                except:
                    pass

    def valueChanged(self, iname, param, value, *args, **kwargs):
        """ Callback used to update values in an instrument """
        instr = self._instruments[self._instrumentnames.index(iname)]
        logging.info('set %s.%s to %s' % (iname, param, value))
        instr.set(param, value)

    def updatecallback(self, start=True, dt=3):
        if self._timer is not None:
            del self._timer

        self.updatedata()

        if start:
            self._timer = QCodesTimer(fn=self.updatedata, dt=dt)
            self._timer.start()
        else:
            self._timer = None

    update_field = Signal(str, str, object, bool)

    @Slot(str, str, object, bool)
    def _set_field(self, iname, g, value, force_update):
        """ Helper function

        Update field of parameter viewer with a string value
        """
        if self.verbose >= 2:
            print('_set_field: %s %s: %s' % (iname, g, str(value)))
        sb = self._itemsdict[iname][g]

        if isinstance(sb, QtWidgets.QTreeWidgetItem):
            sb.setText(1, str(value))
        else:
            # update a float value
            if np.abs(sb.value() - value) > 1e-9 or force_update:
                if not sb.hasFocus():  # do not update when editing
                    logging.debug('update %s to %s' % (g, value))
                    try:
                        oldstate = sb.blockSignals(True)
                        sb.setValue(value)
                        sb.blockSignals(oldstate)
                    except Exception as e:
                        pass

    def updatedata(self, force_update=False):
        """ Update data in viewer using station.snapshow """
        logging.debug('ParameterViewer: update values')
        for iname in self._instrumentnames:
            instr = self._instruments[self._instrumentnames.index(iname)]

            pp = instr.parameters
            ppnames = sorted(instr.parameters.keys())

            si = sys.getswitchinterval()

            for g in ppnames:
                sys.setswitchinterval(100)  # hack to make this semi thread-safe
                value = pp[g].get()
                sys.setswitchinterval(si)

                self.update_field.emit(iname, g, value, force_update)

        for f in self.callbacklist:
            try:
                f()
            except Exception as e:
                logging.debug('update function failed')
                logging.debug(str(e))


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
        p = mp.Process(target=createParameterWidget, args=(instruments, doexec))
        p.start()
        return p

    instrumentnames = [i.name for i in instruments]
    app = pyqtgraph.mkQApp()

    ms = pmatlab.monitorSizes()[-1]
    p = ParameterViewer(instruments=instruments,
                        instrumentnames=instrumentnames)
    p.setGeometry(ms[0] + ms[2] - 320, 30, 300, 600)
    p.show()
    p.updatecallback()

    logging.info('created update widget...')

    if doexec:
        app.exec()
    return p

#%% Debugging code

if __name__ == '__main__':
    import qcodes
    import time
    import pdb
    from qtt.instrument_drivers.virtual_instruments import VirtualIVVI

    ivvi = VirtualIVVI(name='dummyivvi', model=None)
    p = ParameterViewer(instruments=[ivvi], instrumentnames=['ivvi'])
    p.show()
    self = p
    p.updatecallback()

    p.setGeometry(1640, 60, 240, 600)

    time.sleep(.1)
    ivvi.dac1.set(101)

    ivvi.dac2.set(102)

#%%
if __name__ == '__main__' and 0:
    box = QtWidgets.QDoubleSpinBox()
    box.setMaximum(10)
    qq = self.topLevelItem(0).child(2)
    self.setItemWidget(qq, 1, box)
