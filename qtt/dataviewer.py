#%% Load packages

import sys,os
import re
import logging

import qcodes
import qcodes as qc
from qcodes.plots.pyqtgraph import QtPlot # explicit import

import qtpy.QtCore as QtCore
import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets
import pyqtgraph as pg
import traceback

def findfilesR(p, patt):
    """ Get a list of files (recursive)

    Arguments
    ---------

    p (string): directory
    patt (string): pattern to match

    """
    lst = []
    rr = re.compile(patt)
    for root, dirs, files in os.walk(p, topdown=False):
        lst += [os.path.join(root, f) for f in files if re.match(rr, f)]
    return lst

#%% Main class
class DataViewer(QtWidgets.QWidget):

    ''' Simple viewer for Qcodes data
    
    Arugments
    ---------
    
        default_parameter : string
            name of default parameter to plot
    '''
    def __init__(self, window_title='Log Viewer', default_extension='hdf5', default_parameter='amplitude', debugdict=dict()):
        super(DataViewer, self).__init__()

        self.default_parameter = default_parameter
        self.default_extension= default_extension
        
        # setup GUI
        self.text= QtWidgets.QLabel()
        self.text.setText('Log files at %s' %  qcodes.DataSet.default_io.base_location)
        self.logtree= QtWidgets.QTreeView() # QTreeWidget
        self.logtree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._treemodel = QtGui.QStandardItemModel()
        self.logtree.setModel(self._treemodel)
        self.__debug = debugdict
        self.qplot= QtPlot(remote=False, interval=0) # FIXME: interval
        self.plotwindow= self.qplot.win
        #self.plotwindow = pg.GraphicsWindow(title='dummy')

        vertLayout = QtWidgets.QVBoxLayout()
        vertLayout.addWidget(self.text)
        vertLayout.addWidget(self.logtree)
        vertLayout.addWidget(self.plotwindow)
        self.setLayout(vertLayout)

        self._treemodel.setHorizontalHeaderLabels(['Log', 'Comments'])
        self.setWindowTitle(window_title)        
        self.logtree.header().resizeSection(0, 240)


        # disable edit
        self.logtree.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.logtree.doubleClicked.connect(self.logCallback)

        # get logs from disk             
        self.updateLogs()
        
        
    def updateLogs(self):
        ''' Update list of data files 

        extension : string
            can be hdf5 or dat        
            
        '''

        model=self._treemodel
        dd=findfilesR(qcodes.DataSet.default_io.base_location, '.*' + self.default_extension)        
        print(dd)
        
        logs=dict()        
        for i, d in enumerate(dd):
            tag= os.path.basename(d)
            datetag, logtag=d.split('/')[-3:-1]
            if not datetag in logs:
                logs[datetag]=dict()
            logs[datetag][logtag]=d

        self.logs = logs
        
        for i, datetag in enumerate(sorted(logs.keys())[::-1]):             
            parent1 = QtGui.QStandardItem(datetag)
            for j, logtag in enumerate(logs[datetag]):
                child1 = QtGui.QStandardItem(logtag)
                child2 = QtGui.QStandardItem('info about plot')
                child3 = QtGui.QStandardItem(os.path.join(datetag, logtag) )
                parent1.appendRow([child1, child2, child3])
            model.appendRow(parent1)
            # span container columns
            self.logtree.setFirstColumnSpanned(i, self.logtree.rootIndex(), True)

    ''' Return parameter to be plotted '''
    def plot_parameter(self, data):
        
        if self.default_parameter in data.arrays.keys():
            return self.default_parameter
        if 'amplitude' in data.arrays.keys():
            return 'amplitude'

        try:
            key= next(iter (data.arrays.keys()))
            return key
        except:
            return None
            
    def logCallback(self, index):
        logging.debug('dataviewer callback: index %s'% str(index))
        self.__debug['last']=index
        pp=index.parent()
        row=index.row()

        
        tag=pp.child(row,2).data()
        
        # load data
        if tag is not None:
            logging.info('logCallback! tag %s' % tag)
            try:
                logging.debug('load tag %s' % tag) 
                data=qc.load_data(tag)
        
                self.qplot.clear(); 
 
                param_name=self.plot_parameter(data)
                
                if param_name is not None:                    
                    logging.info('using parameter %s for plotting' % param_name)
                    self.qplot.add( getattr(data, param_name) ); 
                else:
                    logging.info('could not find parameter for DataSet')
            except Exception as ex:
                print('logCallback! error in index %s' % index )
                raise ex
                logging.debug(traceback.format_exception(ex))
                pass
        pass


#%% Testing

if __name__=='__main__':
    dataviewer = DataViewer()
    dataviewer.setGeometry(1920+1280,60, 700,800)
    dataviewer.qplot.win.setMaximumHeight(400)
    dataviewer.show()
    self=dataviewer
