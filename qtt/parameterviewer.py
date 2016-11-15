#%%

import time
import threading
import logging
import numpy as np
import sys

import multiprocessing as mp

from qtpy.QtCore import Qt
from qtpy import QtWidgets
from qtpy import QtGui
import pyqtgraph
from qtt import pmatlab
from functools import partial


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


    shared_kwargs=['station', 'instrumentnames']
    
    ''' Simple class to show qcodes parameters '''
    def __init__(self, instruments, instrumentnames=None, name='QuTech Parameter Viewer', **kwargs):
        ''' Simple class to show qcodes parameters

        Arguments:
            instruments (list): list of Qcodes Instruments to show
            name (str): string used in the window title

        '''
        super().__init__(**kwargs)
        w = self
        w.setGeometry(1700, 50, 300, 600)
        w.setColumnCount(3)
        header = QtWidgets.QTreeWidgetItem(["Parameter", "Value"])
        w.setHeaderItem(header)
        w.setWindowTitle(name)

        if instrumentnames is None:
            instrumentnames = [i.name for i in instruments]
        self._instruments=instruments
        self._instrumentnames=instrumentnames
        self._itemsdict = dict()
        for i in instrumentnames:
            self._itemsdict[i] = dict()
        self._timer = None
        # self._station = station
        self.init()
        self.show()

        self.callbacklist = []

    def init(self):
        ''' Initialize parameter viewer '''
        for ii, iname in enumerate(self._instrumentnames):
            instr = self._instruments[ii]
            # pp=instr.parameters.keys()
            pp = instr.parameters
            ppnames = sorted(instr.parameters.keys())

            ppnames = [p for p in ppnames if instr.parameters[p].has_get]
            # pp = dd['instruments'][iname]['parameters']
            gatesroot = QtWidgets.QTreeWidgetItem(self, [iname])
            for g in ppnames:
                # ww=['gates', g]
                si = min(sys.getswitchinterval(), 0.1)  # hack to make this semi thread-safe
                sys.setswitchinterval(100)  # hack to make this semi thread-safe
                #value = pp[g].get()
                sys.setswitchinterval(si)  # hack to make this semi thread-safe
                box = QtWidgets.QDoubleSpinBox()
                box.setKeyboardTracking(False)  # do not emit signals when still editing
                box.setMinimum(-1000)
                box.setMaximum(1000)
                box.setSingleStep(5)

                # A = QtGui.QTreeWidgetItem(gatesroot, [g, box])
                v = ''
                A = QtWidgets.QTreeWidgetItem(gatesroot, [g, v])
                # qq=self.topLevelItem(0).child(2)
                self._itemsdict[iname][g] = A

                if pp[g].has_set:
                    qq = A
                    self.setItemWidget(qq, 1, box)
                    self._itemsdict[iname][g] = box

                box.valueChanged.connect(partial(self.valueChanged, iname, g))

        self.setSortingEnabled(True)
        self.expandAll()

        # self.label.setStyleSheet("QLabel { background-color : #baccba; margin: 2px; padding: 2px; }");

    def setSingleStep(self, instrument_name, value):
        """ Set the default step size for parameters in the viewer """
        lst = self._itemsdict[instrument_name]
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

    def updatecallback(self, start=True):
        if self._timer is not None:
            del self._timer

        self.updatedata()

        if start:
            self._timer = QCodesTimer(fn=self.updatedata, dt=4)
            self._timer.start()
        else:
            self._timer = None

    def updatedata(self):
        ''' Update data in viewer using station.snapshow '''
        # pp = gates['parameters']
        # gatesroot = QtGui.QTreeWidgetItem(w, ["gates"])
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
                # value = pp[g]['value']
                sb = self._itemsdict[iname][g]

                if isinstance(sb, QtWidgets.QTreeWidgetItem):
                    sb.setText(1, str(value))
                else:
                    # update a float value
                    if np.abs(sb.value() - value) > 1e-9:
                        if not sb.hasFocus():  # do not update when editing
                            logging.debug('update %s to %s' % (g, value))
                            try:
                                oldstate = sb.blockSignals(True)
                                sb.setValue(value)
                                sb.blockSignals(oldstate)
                            except Exception as e:
                                pass

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


def createParameterWidget(instruments, doexec=True):
    instrumentnames = [i.name for i in instruments]
    app = pyqtgraph.mkQApp()

    ms = pmatlab.monitorSizes()[-1]
    p = ParameterViewer(instruments=instruments, instrumentnames=instrumentnames)
    p.setGeometry(ms[0] + ms[2] - 280, 20, 260, 600)
    p.show()
    p.updatecallback()

    logging.info('created update widget...')

    if doexec:
        app.exec()
    return p

#%% Debugging code

if __name__ == '__main__':
    import qcodes
    p = ParameterViewer(instruments=[gates], instrumentnames=['ivvi'])
    p.show()
    self = p
    p.updatecallback()

    p.setGeometry(1640, 60, 240, 600)


#%%
if __name__ == '__main__':
    box = QtWidgets.QDoubleSpinBox()
    box.setMaximum(10)
    qq = self.topLevelItem(0).child(2)
    self.setItemWidget(qq, 1, box)
