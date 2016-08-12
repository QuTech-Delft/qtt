#%%

import time
import threading
import logging

from qtpy.QtCore import Qt
from qtpy import QtWidgets
from qtpy import QtGui
import pyqtgraph
import math

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

from functools import partial

class ParameterViewer(QtWidgets.QTreeWidget):

    shared_kwargs=['station', 'instrumentnames']
    
    ''' Simple class to show qcodes parameters '''
    def __init__(self, instruments, instrumentnames=['gates'], name='QuTech Parameter Viewer', **kwargs):
        super().__init__(**kwargs)
        w = self
        w.setGeometry(1700, 50, 300, 600)
        w.setColumnCount(3)
        header = QtWidgets.QTreeWidgetItem(["Parameter", "Value"])
        w.setHeaderItem(header)
                        # Another alternative is
                        # setHeaderLabels(["Tree","First",...])
        w.setWindowTitle(name)

        self._instruments=instruments
        self._instrumentnames=instrumentnames
        self._itemsdict = dict()
        for i in instrumentnames:
            self._itemsdict[i] = dict()
        self._timer = None
        #self._station = station
        self.init()
        self.show()

        self.callbacklist=[]

    def init(self):
        ''' Initialize parameter viewer '''
        #if self._station==None:
        #    return
        #dd = self._station.snapshot()
        #x =
        for ii,iname in enumerate(self._instrumentnames):
            instr = self._instruments[ii]
            #pp=instr.parameters.keys()
            pp=instr.parameters
            ppnames=sorted(instr.parameters.keys())
            
            ppnames=[p for p in ppnames if instr.parameters[p].has_get]
            #pp = dd['instruments'][iname]['parameters']
            gatesroot = QtWidgets.QTreeWidgetItem(self, [iname])
            for g in ppnames:
                # ww=['gates', g]
                value = pp[g].get()
                box=QtWidgets.QDoubleSpinBox()
                box.setMinimum(-1000)
                box.setMaximum(1000)
                

                #A = QtGui.QTreeWidgetItem(gatesroot, [g, box])
                A = QtWidgets.QTreeWidgetItem(gatesroot, [g, str(value)])
                ##qq=self.topLevelItem(0).child(2)
                ##qq=self.topLevelItem(0).child(2)
                self._itemsdict[iname][g] = A
                
                if pp[g].has_set:
                    qq=A
                    self.setItemWidget(qq, 1, box)
                    self._itemsdict[iname][g] = box
                
                box.valueChanged.connect(partial(self.valueChanged, iname, g) )
                
        self.setSortingEnabled(True)
        self.expandAll()

        #self.label.setStyleSheet("QLabel { background-color : #baccba; margin: 2px; padding: 2px; }");
        #self.value.setStyleSheet("QLabel { background-color : #cadaca; margin: 2px; padding: 2px; }");
        #self.plus.setStyleSheet("QPushButton { margin: 0px; padding: 0px; }");
        #self.minus.setStyleSheet("QPushButton { margin: 0px; padding: 0px; }");

    def valueChanged(self, iname, param, value, *args, **kwargs):
        #print([iname, param, value])
        #print(args)
        #print(kwargs)

        instr = self._instruments[self._instrumentnames.index(iname)]
        logging.info('set %s.%s to %s' % (iname, param, value))
        instr.set(param, value)
        #v=self.gates.get(self.name)+self.delta
        #self.gates.set(self.name, v)

        
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
        #dd = self._station.snapshot()
        #gates = dd['instruments']['gates']
        #pp = gates['parameters']
        # gatesroot = QtGui.QTreeWidgetItem(w, ["gates"])
        logging.debug('ParameterViewer: update values')
        for iname in self._instrumentnames:
            instr = self._instruments[self._instrumentnames.index(iname)]
            
            pp=instr.parameters
            ppnames=sorted(instr.parameters.keys())

            for g in ppnames:
                value=pp[g].get()
                #value = pp[g]['value']
                sb = self._itemsdict[iname][g]
                if math.abs(sb.value()-value)>1e-9:
                
                    logging.debug('update %s to %s' % (g, value))
                    try:
                        
                        oldstate=sb.blockSignals(True)
                        sb.setValue( value )
                        sb.blockSignals(oldstate)
                    except Exception as e:
                        pass

        for f in self.callbacklist:
            try:
                f()
            except Exception as e:
                logging.debug('update function failed')
                logging.debug(str(e))


def createUpdateWidget(instruments, doexec=True):
    instrumentnames=[i.name for i in instruments]
    #qtt.tools.dumpstring('createUpdateWidget: start')
    app=pyqtgraph.mkQApp()
    
    p=ParameterViewer(instruments=instruments, instrumentnames=instrumentnames)
    p.setGeometry(1640,60,240,600)
    p.show()
    p.updatecallback()
    


    logging.info('created update widget...')
    
    if doexec:
        app.exec()
    return p
    
#%%

if __name__=='__main__':
    import qcodes
    station=qcodes.station.Station()
    station.add_component(gates)
    station.gates=gates
    p=ParameterViewer(instruments=[gates], instrumentnames=['ivvi'])
    p.show()
    self=p    
    p.updatecallback()
    
    p.setGeometry(1640,60,240,600)

    
#%%
if __name__=='__main__':
    box=QtWidgets.QDoubleSpinBox()
    box.setMaximum(10)
    qq=self.topLevelItem(0).child(2)
    self.setItemWidget(qq, 1, box)
     #ui->treeWidget->setItemWidget(ui->treeWidget->topLevelItem(2)->child(0) , 1 , _spin_angle);

    
    