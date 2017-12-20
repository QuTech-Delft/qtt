# %% Load packages

import os
import re
import logging

import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets
from qtpy.QtWidgets import QWidget

import pyqtgraph as pg
import argparse

import qcodes
from qcodes.plots.pyqtgraph import QtPlot

import qtt

# %% Main class


class DataViewer(QtWidgets.QWidget):

    def __init__(self, datadir=None, window_title='Data browser',
                 default_parameter='amplitude', extensions=['dat', 'hdf5'],
                 verbose=1):
        ''' Simple viewer for Qcodes data

        Arugments
        ---------

            datadir (string or None): directory to scan for experiments
            default_parameter (string): name of default parameter to plot
        '''
        super(DataViewer, self).__init__()
        self.verbose = verbose  # for debugging
        self.default_parameter = default_parameter
        if datadir is None:
            datadir = qcodes.DataSet.default_io.base_location

        self.extensions = extensions

        # setup GUI

        self.dataset = None
        self.text = QtWidgets.QLabel()
        
        #logtree
        self.logtree = QtWidgets.QTreeView()  # QTreeWidget
        self.logtree.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._treemodel = QtGui.QStandardItemModel()
        self.logtree.setModel(self._treemodel)
        
        #metatree
        self.metatree = QtWidgets.QTreeView()
        self._metamodel = QtGui.QStandardItemModel()
        self.metatree.setModel(self._metamodel)
        
        self.__debug = dict()
        if isinstance(QtPlot, QWidget):
            self.qplot = QtPlot()  # remote=False, interval=0)
        else:
            self.qplot = QtPlot(remote=False)  # remote=False, interval=0)
        if isinstance(self.qplot, QWidget):
            self.plotwindow = self.qplot
        else:
            self.plotwindow = self.qplot.win
        topLayout = QtWidgets.QHBoxLayout()
        self.select_dir = QtWidgets.QPushButton()
        self.select_dir.setText('Select directory')

        self.reloadbutton = QtWidgets.QPushButton()
        self.reloadbutton.setText('Reload data')

        self.outCombo = QtWidgets.QComboBox()

        topLayout.addWidget(self.text)
        topLayout.addWidget(self.select_dir)
        topLayout.addWidget(self.reloadbutton)

        treesLayout = QtWidgets.QHBoxLayout()
        treesLayout.addWidget(self.logtree)
        treesLayout.addWidget(self.metatree)

        vertLayout = QtWidgets.QVBoxLayout()

        vertLayout.addItem(topLayout)
        vertLayout.addItem(treesLayout)
        vertLayout.addWidget(self.plotwindow)

        self.pptbutton = QtWidgets.QPushButton()
        self.pptbutton.setText('Send data to powerpoint')
        self.clipboardbutton = QtWidgets.QPushButton()
        self.clipboardbutton.setText('Copy image to clipboard')

        bLayout = QtWidgets.QHBoxLayout()
        bLayout.addWidget(self.outCombo)
        bLayout.addWidget(self.pptbutton)
        bLayout.addWidget(self.clipboardbutton)

        vertLayout.addItem(bLayout)

        self.setLayout(vertLayout)

        self.setWindowTitle(window_title)
        self.logtree.header().resizeSection(0, 280)

        # disable edit
        self.logtree.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)

        self.setDatadir(datadir)

        self.logtree.doubleClicked.connect(self.logCallback)
        self.outCombo.currentIndexChanged.connect(self.comboCallback)
        self.select_dir.clicked.connect(self.selectDirectory)
        self.reloadbutton.clicked.connect(self.updateLogs)
        self.pptbutton.clicked.connect(self.pptCallback)
        self.clipboardbutton.clicked.connect(self.clipboardCallback)
        if self.verbose >= 2:
            print('created gui...')
        # get logs from disk
        self.updateLogs()
        self.datatag = None

        self.show()

    def setDatadir(self, datadir):
        self.datadir = datadir
        self.io = qcodes.DiskIO(datadir)
        logging.info('DataViewer: data directory %s' % datadir)
        self.text.setText('Log files at %s' %
                          self.datadir)

    def pptCallback(self):
        if self.dataset is None:
            print('no data selected')
            return
        qtt.tools.addPPT_dataset(self.dataset,customfig=self.qplot)

    def clipboardCallback(self):
        self.qplot.copyToClipboard()

    def selectDirectory(self):
        from qtpy.QtWidgets import QFileDialog
        d = QtWidgets.QFileDialog(caption='Select data directory')
        d.setFileMode(QFileDialog.Directory)
        d.exec()
        datadir = d.selectedFiles()[0]
        self.setDatadir(datadir)
        print('update logs')
        self.updateLogs()

    def updateLogs(self):
        ''' Update the list of measurements '''
        model = self._treemodel
        dd = []
        for e in self.extensions:
            dd += qtt.pgeometry.findfilesR(self.datadir, '.*%s' %
                                           e, show_progress=True)
        if self.verbose:
            print('DataViewer: found %d files' % (len(dd)))

        self.datafiles = sorted(dd)
        self.datafiles = [os.path.join(self.datadir, d)
                          for d in self.datafiles]

        model.clear()
        model.setHorizontalHeaderLabels(['Log', 'Comments'])

        logs = dict()
        for i, d in enumerate(dd):
            try:
                datetag, logtag = d.split(os.sep)[-3:-1]
                if datetag not in logs:
                    logs[datetag] = dict()
                logs[datetag][logtag] = d
            except Exception:
                pass
        self.logs = logs

        if self.verbose >= 2:
            print('DataViewer: create gui elements')
        for i, datetag in enumerate(sorted(logs.keys())[::-1]):
            if self.verbose >= 2:
                print('DataViewer: datetag %s ' % datetag)

            parent1 = QtGui.QStandardItem(datetag)
            for j, logtag in enumerate(sorted(logs[datetag])):
                filename = logs[datetag][logtag]
                child1 = QtGui.QStandardItem(logtag)
                child2 = QtGui.QStandardItem('info about plot')
                if self.verbose >= 2:
                    print('datetag %s, logtag %s' % (datetag, logtag))
                child3 = QtGui.QStandardItem(os.path.join(datetag, logtag))
                child4 = QtGui.QStandardItem(filename)
                parent1.appendRow([child1, child2, child3, child4])
            model.appendRow(parent1)
            # span container columns
            self.logtree.setFirstColumnSpanned(
                i, self.logtree.rootIndex(), True)
        if self.verbose >= 2:
            print('DataViewer: updateLogs done')

             
    def updateMetatree(self):
        ''' Update metadata tree '''
        self._metamodel.clear()
        self._metamodel.setHorizontalHeaderLabels(['meatadata', 'value'])
        try:
            self.fill_item(self._metamodel, self.dataset.metadata)
        except Exception as ex:
            print(ex)
    
    def fill_item(self, item, value):
        ''' recursive population of tree structure with a dict '''
        def new_item(parent, text, val=None):
            child = QtGui.QStandardItem(text)
            self.fill_item(child, val)
            parent.appendRow(child)

        if value is None: return
        elif isinstance(value, dict):                    
            for key, val in sorted(value.items()):
                if type(val) in [str, float, int]:
                    child = [QtGui.QStandardItem(str(key)),QtGui.QStandardItem(str(val))]
                    item.appendRow(child)
                else:
                    new_item(item, str(key), val)
        else:
            new_item(item, str(value))
                

    def getPlotParameter(self):
        ''' Return parameter to be plotted '''
        param_name = self.outCombo.currentText()
        if param_name is not '':
            return param_name
        parameters = self.dataset.arrays.keys()
        if self.default_parameter in parameters:
            return self.default_parameter
        return self.dataset.default_parameter_name()

    def selectedDatafile(self):
        return self.datatag

    def comboCallback(self, index):
        if not self._update_plot_:
            return
        param_name = self.getPlotParameter()
        if self.dataset is not None:
            self.updatePlot(param_name)

    def logCallback(self, index):
        """ Function called when. a log entry is selected """
        logging.info('logCallback: index %s' % str(index))
        self.__debug['last'] = index
        pp = index.parent()
        row = index.row()
        tag = pp.child(row, 2).data()
        filename = pp.child(row, 3).data()
        self.filename = filename
        self.datatag = tag
        if tag is None:
            return
        if self.verbose >= 2:
            print('DataViewer logCallback: tag %s, filename %s' %
                  (tag, filename))
        try:
            logging.debug('DataViewer: load tag %s' % tag)
            data = self.loadData(filename, tag)
            self.dataset = data
            self.updateMetatree()
            
            data_keys = data.arrays.keys()
            infotxt = 'arrays: ' + ', '.join(list(data_keys))
            q = pp.child(row, 1).model()
            q.setData(pp.child(row, 1), infotxt)
            self.resetComboItems(data, data_keys)
            param_name = self.getPlotParameter()
            self.updatePlot(param_name)
        except Exception as e:
            print('logCallback! error: %s' % str(e))
            logging.exception(e)
        return

    def resetComboItems(self, data, keys):
        old_key = self.outCombo.currentText()
        self._update_plot_ = False
        self.outCombo.clear()
        for key in keys:
            if not getattr(data, key).is_setpoint:
                self.outCombo.addItem(key)
        if old_key in keys:
            self.outCombo.setCurrentIndex(self.outCombo.findText(old_key))
            
        self._update_plot_ = True
        return

    def loadData(self, filename, tag):
        location = os.path.split(filename)[0]
        try:
            from qcodes.data.hdf5_format import HDF5Format
            hformatter = HDF5Format()
            data = qcodes.load_data(location, formatter=hformatter, io=self.io)
            logging.debug('loaded HDF5 dataset %s' % tag)
        except Exception as ex:
            from qcodes.data.gnuplot_format import GNUPlotFormat
            hformatter = GNUPlotFormat()
            data = qcodes.load_data(location, formatter=hformatter, io=self.io)
            logging.debug('loaded GNUPlotFormat dataset %s' % tag)
        return data

    def updatePlot(self, parameter):
        self.qplot.clear()
        if parameter is None:
            logging.info('could not find parameter for DataSet')
            return
        else:
            logging.info('using plotting parameter %s' % parameter)
            self.qplot.add(getattr(self.dataset, parameter))

# %% Run the GUI as a standalone program

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        sys.argv += ['-d', os.path.join(os.path.expanduser('~'),
                     'tmp', 'qdata')]

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', default=1, help="verbosity level")
    parser.add_argument(
        '-d', '--datadir', type=str, default=None, help="data directory")
    args = parser.parse_args()
    verbose = args.verbose
    datadir = args.datadir

    app = pg.mkQApp()

    dataviewer = DataViewer(datadir=datadir, extensions=['dat', 'hdf5'])
    dataviewer.verbose = 5
    dataviewer.setGeometry(1280, 60, 700, 900)
    dataviewer.plotwindow.setMaximumHeight(400)
    dataviewer.show()
    self = dataviewer

    # app.exec()


# %%

if 0:
    tag = list(list(dataviewer.logs.items())[0][1].items())[0][1]
    data = qcodes.load_data(tag)

    l = data.location

    data.formatter = HDF5Format()
    data.write()
    data._h5_base_group.close()
