# %% Load packages

import os
import logging
import argparse

import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets
from qtpy.QtWidgets import QWidget

import pyqtgraph as pg

import qcodes
from qcodes.plots.pyqtgraph import QtPlot
import qtt

# %% Main class


class DataViewer(QtWidgets.QWidget):

    def __init__(self, datadir=None, window_title='Data browser',
                 default_parameter='amplitude', extensions=['dat', 'hdf5'],
                 verbose=1):
        """ Simple viewer for Qcodes data

        Args:

            datadir (string or None): directory to scan for experiments
            default_parameter (string): name of default parameter to plot
        """
        super(DataViewer, self).__init__()
        self.verbose = verbose  # for debugging
        self.default_parameter = default_parameter
        if datadir is None:
            datadir = qcodes.DataSet.default_io.base_location

        self.extensions = extensions

        # setup GUI

        self.dataset = None
        self.text = QtWidgets.QLabel()

        # logtree
        self.logtree = QtWidgets.QTreeView()  # QTreeWidget
        self.logtree.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._treemodel = QtGui.QStandardItemModel()
        self.logtree.setModel(self._treemodel)

        # metatabs
        self.meta_tabs = QtWidgets.QTabWidget()
        self.meta_tabs.addTab(QtWidgets.QWidget(), 'metadata')

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

        self.loadinfobutton = QtWidgets.QPushButton()
        self.loadinfobutton.setText('Preload info')

        self.outCombo = QtWidgets.QComboBox()

        topLayout.addWidget(self.text)
        topLayout.addWidget(self.select_dir)
        topLayout.addWidget(self.reloadbutton)
        topLayout.addWidget(self.loadinfobutton)

        treesLayout = QtWidgets.QHBoxLayout()
        treesLayout.addWidget(self.logtree)
        treesLayout.addWidget(self.meta_tabs)

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
        self.loadinfobutton.clicked.connect(self.loadInfo)
        self.pptbutton.clicked.connect(self.pptCallback)
        self.clipboardbutton.clicked.connect(self.clipboardCallback)
        if self.verbose >= 2:
            print('created gui...')
        # get logs from disk
        self.updateLogs()
        self.datatag = None

        self.logtree.setColumnHidden(2, True)
        self.logtree.setColumnHidden(3, True)

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
        qtt.tools.addPPT_dataset(self.dataset, customfig=self.qplot)

    def clipboardCallback(self):
        self.qplot.copyToClipboard()

    def getArrayStr(self, metadata):
        params = []
        try:
            if 'loop' in metadata.keys():
                sv = metadata['loop']['sweep_values']
                params.append('%s [%.2f to %.2f %s]' % (sv['parameter']['label'],
                                                        sv['values'][0]['first'],
                                                        sv['values'][0]['last'],
                                                        sv['parameter']['unit']))

                for act in metadata['loop']['actions']:
                    if 'sweep_values' in act.keys():
                        sv = act['sweep_values']
                        params.append('%s [%.2f - %.2f %s]' % (sv['parameter']['label'],
                                                               sv['values'][0]['first'],
                                                               sv['values'][0]['last'],
                                                               sv['parameter']['unit']))
                infotxt = ' ,'.join(params)
                infotxt = infotxt + '  |  ' + ', '.join([('%s' % (v['label'])) for (
                    k, v) in metadata['arrays'].items() if not v['is_setpoint']])

            elif 'scanjob' in metadata.keys():
                sd = metadata['scanjob']['sweepdata']
                params.append(
                    '%s [%.2f to %.2f]' %
                    (sd['param'], sd['start'], sd['end']))
                if 'stepdata' in metadata['scanjob']:
                    sd = metadata['scanjob']['stepdata']
                    params.append(
                        '%s [%.2f to %.2f]' %
                        (sd['param'], sd['start'], sd['end']))
                infotxt = ' ,'.join(params)
                infotxt = infotxt + '  |  ' + \
                    ', '.join(metadata['scanjob']['minstrument'])
            else:
                infotxt = 'info about plot'

        except BaseException:
            infotxt = 'info about plot'

        return infotxt

    def loadInfo(self):
        try:
            for row in range(self._treemodel.rowCount()):
                index = self._treemodel.index(row, 0)
                i = 0
                while (index.child(i, 0).data() is not None):
                    filename = index.child(i, 3).data()
                    loc = '\\'.join(filename.split('\\')[:-1])
                    tempdata = qcodes.DataSet(loc)
                    tempdata.read_metadata()
                    infotxt = self.getArrayStr(tempdata.metadata)
                    self._treemodel.setData(index.child(i, 1), infotxt)
                    if 'comment' in tempdata.metadata.keys():
                        self._treemodel.setData(index.child(
                            i, 4), tempdata.metadata['comment'])
                    i = i + 1
        except Exception as e:
            print(e)

    def selectDirectory(self):
        from qtpy.QtWidgets import QFileDialog
        d = QtWidgets.QFileDialog(caption='Select data directory')
        d.setFileMode(QFileDialog.Directory)
        if d.exec():
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
        model.setHorizontalHeaderLabels(
            ['Log', 'Arrays', 'location', 'filename', 'Comments'])

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
#            self.logtree.setFirstColumnSpanned(
#                i, self.logtree.rootIndex(), True)
            self.logtree.setColumnWidth(0, 240)
            self.logtree.setColumnHidden(2, True)
            self.logtree.setColumnHidden(3, True)

        if self.verbose >= 2:
            print('DataViewer: updateLogs done')

    def _create_meta_tree(self, meta_dict):
        metatree = QtWidgets.QTreeView()
        _metamodel = QtGui.QStandardItemModel()
        metatree.setModel(_metamodel)
        metatree.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)

        _metamodel.setHorizontalHeaderLabels(['metadata', 'value'])

        try:
            self.fill_item(_metamodel, meta_dict)
            return metatree

        except Exception as ex:
            print(ex)

    def updateMetaTabs(self):
        ''' Update metadata tree '''
        meta = self.dataset.metadata

        self.meta_tabs.clear()
        if 'gates' in meta.keys():
            self.meta_tabs.addTab(self._create_meta_tree(meta['gates']),
                                  'gates')
        elif meta.get('station', dict()).get('instruments', dict()).get('gates', None) is not None:
            self.meta_tabs.addTab(self._create_meta_tree(meta['station']['instruments']['gates']),
                                  'gates')
        if meta.get('station', dict()).get('instruments', None) is not None:
            if 'instruments' in meta['station'].keys():
                self.meta_tabs.addTab(self._create_meta_tree(meta['station']['instruments']),
                                      'instruments')

        self.meta_tabs.addTab(self._create_meta_tree(meta), 'metadata')

    def fill_item(self, item, value):
        ''' recursive population of tree structure with a dict '''
        def new_item(parent, text, val=None):
            child = QtGui.QStandardItem(text)
            self.fill_item(child, val)
            parent.appendRow(child)

        if value is None:
            return
        elif isinstance(value, dict):
            for key, val in sorted(value.items()):
                if type(val) in [str, float, int]:
                    child = [QtGui.QStandardItem(
                        str(key)), QtGui.QStandardItem(str(val))]
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
        """ Return currently selected data file """
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
            self.updateMetaTabs()

            data_keys = data.arrays.keys()
            infotxt = self.getArrayStr(data.metadata)
            q = pp.child(row, 1).model()
            q.setData(pp.child(row, 1), infotxt)
            if 'comment' in data.metadata.keys():
                q.setData(pp.child(row, 2), data.metadata['comment'])
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
        data = qtt.data.load_dataset(location)
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
    dataviewer.logtree.setColumnWidth(0, 240)
    dataviewer.show()

    # app.exec()
