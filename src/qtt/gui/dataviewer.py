# %% Load packages

import argparse
import logging
import os

import pyqtgraph as pg
import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets
from qtpy.QtWidgets import QFileDialog, QWidget

import qcodes_loop
from qcodes_loop.data.data_set import DataSet
import qtt
from qcodes_loop.plots.pyqtgraph import QtPlot


# %% Main class


class DataViewer(QtWidgets.QMainWindow):

    def __init__(self, data_directory=None, window_title='Data browser',
                 default_parameter='amplitude', extensions=None, verbose=1):
        """ Contstructs a simple viewer for Qcodes data.

        Args:
            data_directory (string or None): The directory to scan for experiments.
            default_parameter (string): A name of default parameter to plot.
            extensions (list): A list with the data file extensions to filter.
            verbose (int): The logging verbosity level.
        """
        super(DataViewer, self).__init__()
        if extensions is None:
            extensions = ['dat', 'hdf5']

        self.verbose = verbose
        self.default_parameter = default_parameter
        self.data_directories = [None] * 2
        self.directory_index = 0
        if data_directory is None:
            data_directory = DataSet.default_io.base_location

        self.extensions = extensions

        # setup GUI
        self.dataset = None
        self.text = QtWidgets.QLabel()

        # logtree
        self.logtree = QtWidgets.QTreeView()
        self.logtree.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._treemodel = QtGui.QStandardItemModel()
        self.logtree.setModel(self._treemodel)

        # metatabs
        self.meta_tabs = QtWidgets.QTabWidget()
        self.meta_tabs.addTab(QtWidgets.QWidget(), 'metadata')

        self.__debug = dict()
        if isinstance(QtPlot, QWidget):
            self.qplot = QtPlot()
        else:
            self.qplot = QtPlot(remote=False)
        if isinstance(self.qplot, QWidget):
            self.plotwindow = self.qplot
        else:
            self.plotwindow = self.qplot.win

        topLayout = QtWidgets.QHBoxLayout()

        self.filterbutton = QtWidgets.QPushButton()
        self.filterbutton.setText('Filter data')
        self.filtertext = QtWidgets.QLineEdit()
        self.outCombo = QtWidgets.QComboBox()

        topLayout.addWidget(self.text)
        topLayout.addWidget(self.filterbutton)
        topLayout.addWidget(self.filtertext)

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
        widget = QtWidgets.QWidget()
        widget.setLayout(vertLayout)
        self.setCentralWidget(widget)

        self.setWindowTitle(window_title)
        self.logtree.header().resizeSection(0, 280)

        # disable edit
        self.logtree.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)

        self.set_data_directory(data_directory)
        self.logtree.doubleClicked.connect(self.log_callback)
        self.outCombo.currentIndexChanged.connect(self.combobox_callback)
        self.filterbutton.clicked.connect(
            lambda: self.update_logs(
                filter_str=self.filtertext.text()))
        self.pptbutton.clicked.connect(self.ppt_callback)
        self.clipboardbutton.clicked.connect(self.clipboard_callback)

        menuBar = self.menuBar()

        menuDict = {
            '&Data': {'&Reload Data': self.update_logs,
                      '&Preload all Info': self.load_info,
                      '&Quit': self.close},
            '&Folder': {'&Select Dir1': lambda: self.select_directory(index=0),
                        'Select &Dir2': lambda: self.select_directory(index=1),
                        '&Toggle Dirs': self.toggle_data_directory
                        },
            '&Help': {'&Info': self.show_help}
        }
        for (k, menu) in menuDict.items():
            mb = menuBar.addMenu(k)
            for (kk, action) in menu.items():
                act = QtWidgets.QAction(kk, self)
                mb.addAction(act)
                act.triggered.connect(action)

        if self.verbose >= 2:
            print('created gui...')

        # get logs from disk
        self.update_logs()
        self.datatag = None

        self.logtree.setColumnHidden(2, True)
        self.logtree.setColumnHidden(3, True)
        self.show()

    def set_data_directory(self, data_directory, index=0):
        self.data_directories[index] = data_directory
        self.data_directory = data_directory
        self.disk_io = qcodes_loop.data.io.DiskIO(data_directory)
        logging.info('DataViewer: data directory %s' % data_directory)
        self.text.setText('Log files at %s' % self.data_directory)

    def show_help(self):
        """ Show help dialog """
        self.infotext = "Dataviewer for qcodes datasets"
        QtWidgets.QMessageBox.information(self, 'qtt dataviwer control info', self.infotext)

    def toggle_data_directory(self):
        index = (self.directory_index + 1) % len(self.data_directories)
        self.directory_index = index
        self.data_directory = self.data_directories[index]
        self.disk_io = qcodes_loop.data.io.DiskIO(self.data_directory)
        logging.info('DataViewer: data directory %s' % self.data_directory)
        self.text.setText('Log files at %s' % self.data_directory)
        self.update_logs()

    def ppt_callback(self):
        if self.dataset is None:
            print('no data selected')
            return
        qtt.utilities.tools.addPPT_dataset(self.dataset, customfig=self.qplot)

    def clipboard_callback(self):
        self.qplot.copyToClipboard()

    @staticmethod
    def get_data_info(metadata):
        params = []
        try:
            if 'loop' in metadata.keys():
                sv = metadata['loop']['sweep_values']
                params.append(
                    '%s [%.2f to %.2f %s]' %
                    (sv['parameter']['label'],
                     sv['values'][0]['first'],
                     sv['values'][0]['last'],
                     sv['parameter']['unit']))

                for act in metadata['loop']['actions']:
                    if 'sweep_values' in act.keys():
                        sv = act['sweep_values']
                        params.append(
                            '%s [%.2f - %.2f %s]' %
                            (sv['parameter']['label'],
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

    def load_info(self):
        try:
            for row in range(self._treemodel.rowCount()):
                index = self._treemodel.index(row, 0)
                i = 0
                while (index.child(i, 0).data() is not None):
                    filename = index.child(i, 3).data()
                    loc = '\\'.join(filename.split('\\')[:-1])
                    tempdata = DataSet(loc)
                    tempdata.read_metadata()
                    infotxt = DataViewer.get_data_info(tempdata.metadata)
                    self._treemodel.setData(index.child(i, 1), infotxt)
                    if 'comment' in tempdata.metadata.keys():
                        self._treemodel.setData(index.child(
                            i, 4), tempdata.metadata['comment'])
                    i = i + 1
        except Exception as e:
            print(e)

    def select_directory(self, index=0):
        d = QtWidgets.QFileDialog(caption='Select data directory')
        d.setFileMode(QFileDialog.Directory)
        if d.exec():
            datadir = d.selectedFiles()[0]
            self.set_data_directory(datadir, index)
            print('update logs')
            self.update_logs()

    @staticmethod
    def find_datafiles(datadir, extensions=None, show_progress=True):
        """ Find all datasets in a directory with a given extension """
        if extensions is None:
            extensions = ['dat', 'hdf5']
        dd = []
        for e in extensions:
            dd += qtt.pgeometry.findfilesR(datadir, '.*%s' %
                                           e, show_progress=show_progress)

        datafiles = sorted(dd)
        return datafiles

    @staticmethod
    def _filename2datetag(filename):
        """ Parse a filename to a date tag and base filename """
        if filename.endswith('.json'):
            datetag = filename.split(os.sep)[-1].split('_')[0]
            logtag = filename.split(os.sep)[-1][:-5]
        else:
            # other formats, assumed to be in normal form
            datetag, logtag = filename.split(os.sep)[-3:-1]
        return datetag, logtag

    def update_logs(self, filter_str=None):
        ''' Update the list of measurements '''
        model = self._treemodel

        self.datafiles = self.find_datafiles(self.data_directory, self.extensions)
        dd = self.datafiles

        if filter_str:
            dd = [s for s in dd if filter_str in s]

        if self.verbose:
            print('DataViewer: found %d files' % (len(dd)))

        model.clear()
        model.setHorizontalHeaderLabels(
            ['Log', 'Arrays', 'location', 'filename', 'Comments'])

        logs = dict()
        for _, filename in enumerate(dd):
            try:
                datetag, logtag = self._filename2datetag(filename)
                if datetag not in logs:
                    logs[datetag] = dict()
                logs[datetag][logtag] = filename
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
            self.logtree.setColumnWidth(0, 240)
            self.logtree.setColumnHidden(2, True)
            self.logtree.setColumnHidden(3, True)

        if self.verbose >= 2:
            print('DataViewer: update_logs done')

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

    def update_meta_tabs(self):
        ''' Update metadata tree '''
        meta = self.dataset.metadata

        self.meta_tabs.clear()
        if 'gates' in meta.keys():
            self.meta_tabs.addTab(self._create_meta_tree(meta['gates']),
                                  'gates')
        elif meta.get('station', dict()).get('instruments', dict()).get('gates', None) is not None:
            self.meta_tabs.addTab(
                self._create_meta_tree(
                    meta['station']['instruments']['gates']),
                'gates')
        if meta.get('station', dict()).get('instruments', None) is not None:
            if 'instruments' in meta['station'].keys():
                self.meta_tabs.addTab(
                    self._create_meta_tree(
                        meta['station']['instruments']),
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

    def get_plot_parameter(self):
        ''' Return parameter to be plotted '''
        param_name = self.outCombo.currentText()
        if param_name != '':
            return param_name
        parameters = self.dataset.arrays.keys()
        if self.default_parameter in parameters:
            return self.default_parameter
        return self.dataset.default_parameter_name()

    def selected_data_file(self):
        """ Return currently selected data file """
        return self.datatag

    def combobox_callback(self, index):
        if not self._update_plot_:
            return
        param_name = self.get_plot_parameter()
        if self.dataset is not None:
            self.update_plot(param_name)

    def log_callback(self, index):
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
            data = DataViewer.load_data(filename, tag)
            if not data:
                raise ValueError('File invalid (%s) ...' % filename)
            self.dataset = data
            self.update_meta_tabs()

            data_keys = data.arrays.keys()
            infotxt = DataViewer.get_data_info(data.metadata)
            q = pp.child(row, 1).model()
            q.setData(pp.child(row, 1), infotxt)
            if 'comment' in data.metadata.keys():
                q.setData(pp.child(row, 2), data.metadata['comment'])
            self.reset_combo_items(data, data_keys)
            param_name = self.get_plot_parameter()
            self.update_plot(param_name)
        except Exception as e:
            print('logCallback! error: %s' % str(e))
            logging.exception(e)
        return

    def reset_combo_items(self, data, keys):
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

    @staticmethod
    def load_data(filename, tag):
        if filename.endswith('.json'):
            location = filename
        else:
            # qcodes datasets are found by filename, but should be loaded by directory...
            location = os.path.split(filename)[0]
        data = qtt.data.load_dataset(location)
        return data

    def update_plot(self, parameter):
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

    dataviewer = DataViewer(data_directory=datadir, extensions=['dat', 'hdf5'])
    dataviewer.verbose = 5
    dataviewer.setGeometry(1280, 60, 700, 900)
    dataviewer.logtree.setColumnWidth(0, 240)
    dataviewer.show()
