#%%

import time
import threading
import pyqtgraph
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui # as QtGui

import logging

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
            
class ParameterViewer(Qt.QtGui.QTreeWidget):

    ''' Simple class to show qcodes parameters '''
    def __init__(self, station, instrumentnames=['gates'], name='QuTech Parameter Viewer', **kwargs):
        super().__init__(**kwargs)
        w = self
        w.setGeometry(1700, 50, 300, 600)
        w.setColumnCount(3)
        header = QtGui.QTreeWidgetItem(["Parameter", "Value"])
        w.setHeaderItem(header)
                        # Another alternative is
                        # setHeaderLabels(["Tree","First",...])
        w.setWindowTitle(name)

        self._instrumentnames=instrumentnames
        self._itemsdict = dict()
        self._itemsdict['gates'] = dict()
        self._timer = None
        self._station = station
        self.init()
        self.show()

        self.callbacklist=[]
        
    def init(self):
        ''' Initialize parameter viewer '''
        if self._station==None:
            return
        dd = self._station.snapshot()
        #x = 
        for iname in self._instrumentnames:
            pp = dd['instruments'][iname]['parameters']
            gatesroot = QtGui.QTreeWidgetItem(self, [iname])
            for g in pp:
                # ww=['gates', g]
                value = pp[g]['value']
                A = QtGui.QTreeWidgetItem(gatesroot, [g, str(value)])
                self._itemsdict['gates'][g] = A
        self.setSortingEnabled(True)
        self.expandAll()

    def updatecallback(self, start=True):
        if self._timer is not None:
            del self._timer
            
        if start:
            self._timer = QCodesTimer(fn=self.updatedata)
            self._timer.start()
        else:
            self._timer = None
        
        
    def updatedata(self):
        ''' Update data in viewer using station.snapshow '''
        dd = self._station.snapshot()
        gates = dd['instruments']['gates']
        pp = gates['parameters']
        # gatesroot = QtGui.QTreeWidgetItem(w, ["gates"])
        for g in pp:
            #ww = ['gates', g]
            value = pp[g]['value']
            x = self._itemsdict['gates'][g]
            logging.debug('update %s to %s' % (g, value))
            x.setText(1, str(value))

        for f in self.callbacklist:
            try:
                f()
            except Exception as e:
                logging.debug('update function failed')                  
                logging.debug(str(e))
                
            