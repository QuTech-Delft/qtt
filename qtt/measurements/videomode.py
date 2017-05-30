# -*- coding: utf-8 -*-
"""
Contains code to do live plotting 

"""
#%%
import datetime
import threading
from qcodes.instrument.parameter import StandardParameter
from qcodes.utils.validators import Numbers
from qtt.live_plotting import livePlot, fpgaCallback_1d, fpgaCallback_2d
from qtt.tools import connect_slot
import qtpy.QtWidgets as QtWidgets
from qtt.measurements.scans import plotData, makeDataset_sweep, makeDataset_sweep_2D


#%%


class VideoMode:
    """ Controls the videomode tuning.

    Attributes:
        station (qcodes station): contains all the information about the set-up
        sweepparams (string, 1 x 2 list or dict): the parameter(s) to be swept
        sweepranges (int or 1 x 2 list): the range(s) to be swept over
        minstrument (int): the channel of the FPGA
        Naverage (int): the number of times the FPGA averages
        resolution (1 x 2 list): for 2D the resolution
    """
    # TODO: implement optional sweep directions, i.e. forward and backward
    # TODO: implement virtual gates functionality
    def __init__(self, station, sweepparams, sweepranges, minstrument, Naverage=25, resolution=[90, 90], diff_dir=None, verbose=1):
        self.station = station
        self.verbose=verbose
        self.sweepparams = sweepparams
        self.sweepranges = sweepranges
        self.fpga_ch = minstrument
        self.Naverage = StandardParameter('Naverage', get_cmd=self._get_Naverage, set_cmd=self._set_Naverage, vals=Numbers(1, 1023))
        self._Naverage_val = Naverage
        self.resolution = resolution
        self.diff_dir = None
        self.lp = livePlot(None, self.station.gates, self.sweepparams, self.sweepranges)

        self.lp.win.start_button.clicked.connect(connect_slot(self.run))
        self.lp.win.stop_button.clicked.connect(connect_slot(self.stop))

        self.lp.win.single_button = QtWidgets.QPushButton('Single')
        self.lp.win.layout().children()[0].insertWidget(1, self.lp.win.single_button)
        self.lp.win.single_button.clicked.connect(connect_slot(self.single))

        box = QtWidgets.QSpinBox()
        box.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        box.setKeyboardTracking(False)  # do not emit signals when still editing
        box.setMinimum(1)
        box.setMaximum(1023)
        box.setPrefix('Naverage: ')
        box.setMaximumWidth(120)
        box.valueChanged.connect(self.Naverage.set)
        self.box = box
        self.lp.win.layout().children()[0].addWidget(self.box)
        
        self.run()
        
        self.datalock = threading.Lock()

    def close(self):
        self.lp.close()
        
    def get_dataset(self, run=False):
        """ Return latest recorded dataset """
        with self.datalock:
            if run:
                data=self.lp.datafunction()
            else:
                data=self.lp.datafunction_result
            self.alldata = self.makeDataset(data, Naverage=None)
            #self.alldata.metadata['idx']=self.lp.idx
            return self.alldata
    def run(self):
        """ Programs the AWG, starts the read-out and the plotting. """
        if type(self.sweepranges) is int:
            if type(self.sweepparams) is str:
                waveform, _ = self.station.awg.sweep_gate(self.sweepparams, self.sweepranges, period=1e-3)
            elif type(self.sweepparams) is dict:
                waveform, _ = self.station.awg.sweep_gate_virt(self.sweepparams, self.sweeprange, period=1e-3)
            self.datafunction = fpgaCallback_1d(self.station, waveform, self.Naverage.get(), self.fpga_ch)
        elif type(self.sweepranges) is list:
            if type(self.sweepparams) is list:
                waveform, _ = self.station.awg.sweep_2D(self.station.fpga.sampling_frequency.get(), self.sweepparams, self.sweepranges, self.resolution)
            elif type(self.sweepparams) is dict:
                waveform, _ = self.station.awg.sweep_2D_virt(self.station.fpga.sampling_frequency.get(), self.sweepparams['gates_horz'], self.sweepparams['gates_vert'], self.sweepranges, self.resolution)
            self.datafunction = fpgaCallback_2d(self.station, waveform, self.Naverage.get(), self.fpga_ch, self.resolution, self.diff_dir)

        self.lp.datafunction = self.datafunction
        self.lp.startreadout()
        self.box.setValue(self.Naverage.get())

        if hasattr(self.station, 'RF'):
            self.station.RF.on()

    def stop(self):
        """ Stops the plotting, AWG(s) and if available RF. """
        self.lp.stopreadout()
        if self.station is not None:
            if hasattr(self.station, 'awg'):
                self.station.awg.stop()
            if hasattr(self.station, 'RF'):
                self.station.RF.off()

    def single(self):
        """Do a single scan with a lot averaging.

        Note: this does not yet support the usage of linear combinations of 
        gates (a.k.a. virtual gates).
        """
        Naverage = 1000
        Naverage_old = self.Naverage.get()
        waittime_old = self.lp.datafunction.waittime
        self.lp.datafunction.waittime = Naverage * self.lp.data.size / self.station.fpga.sampling_frequency.get()
        self.Naverage.set(Naverage)
        self.data = self.lp.datafunction()
        self.Naverage.set(Naverage_old)
        self.lp.datafunction.waittime = waittime_old
        alldata = self.makeDataset(self.data, Naverage=Naverage)
        alldata.write(write_metadata=True)
        plotData(alldata, fig=None)
        with self.datalock:
            self.alldata = alldata
        if self.verbose:
            print('videomode: recorded single shot measurement')
            
    def makeDataset(self, data, Naverage=None):
        if data.ndim == 1:
            alldata, _ = makeDataset_sweep(data, self.sweepparams, self.sweepranges, gates=self.station.gates, loc_record={'label': 'videomode_1d_single'})
        if data.ndim == 2:
            alldata, _ = makeDataset_sweep_2D(data, self.station.gates, self.sweepparams, self.sweepranges, loc_record={'label': 'videomode_2d_single'})
        alldata.metadata = {'scantime': str(datetime.datetime.now()), 'station': self.station.snapshot(), 'allgatevalues': self.station.gates.allvalues()}
        alldata.metadata['Naverage'] = Naverage
        if hasattr(self.lp.datafunction, 'diff_dir'):
            alldata.metadata['diff_dir'] = self.lp.datafunction.diff_dir
        return alldata
    
    def _get_Naverage(self):
        return self._Naverage_val

    def _set_Naverage(self, value):
        self._Naverage_val = value
        self.lp.datafunction.Naverage = value
        self.box.setValue(value)
