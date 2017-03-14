# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:43:47 2017

@author: diepencjv
"""
#%%
from qtt.live_plotting import livePlot, fpgaCallback_1d, fpgaCallback_2d
from qtt.tools import connect_slot
import qtpy.QtWidgets as QtWidgets
from qtt.scans import plotData, scan1Dfast, scan2Dturbo, makeDataset_sweep, makeDataset_sweep_2D
import datetime

#%%


class VideoMode:
    """ Controls the videomode tuning.

    Attributes:
        station (qcodes station): contains all the information about the set-up
        sweepparams (string, 1 x 2 list or dict): the parameter(s) to be swept
        sweepranges (int or 1 x 2 list): the range(s) to be swept over
        fpga_ch (int): the channel of the FPGA
        Naverage (int): the number of times the FPGA averages
        resolution (1 x 2 list): for 2D the resolution
    """
    # TODO: implement optional sweep directions, i.e. forward and backward
    # TODO: implement virtual gates functionality
    def __init__(self, station, sweepparams, sweepranges, fpga_ch, Naverage=25, resolution=[90, 90], diff_dir=None):
        self.station = station
        self.sweepparams = sweepparams
        self.sweepranges = sweepranges
        self.fpga_ch = fpga_ch
        self.Naverage = Naverage
        self.resolution = resolution
        self.diff_dir = None
        self.lp = livePlot(None, self.station.gates, self.sweepparams, self.sweepranges)

        self.lp.win.start_button.clicked.connect(connect_slot(self.run))
        self.lp.win.stop_button.clicked.connect(connect_slot(self.stop))

        self.lp.win.single_button = QtWidgets.QPushButton('Single')
        self.lp.win.layout().children()[0].removeWidget(self.lp.win.stop_button)
        self.lp.win.layout().children()[0].addWidget(self.lp.win.single_button)
        self.lp.win.layout().children()[0].addWidget(self.lp.win.stop_button)
        self.lp.win.single_button.clicked.connect(connect_slot(self.single))

        box = QtWidgets.QDoubleSpinBox()
        box.setKeyboardTracking(False)  # do not emit signals when still editing
        box.setMinimum(1)
        box.setMaximum(1023)
        box.setSingleStep(1)
        box.setPrefix('Naverage: ')
        box.setDecimals(0)
        box.setMaximumWidth(120)
        box.valueChanged.connect(self.Naverage_changed)
        self.lp.win.layout().children[0].addWidget(box)

        self.run()

    def run(self):
        """ Programs the AWG, starts the read-out and the plotting. """
        if type(self.sweepranges) is int:
            if type(self.sweepparams) is str:
                waveform, _ = self.station.awg.sweep_gate(self.sweepparams, self.sweepranges, period=1e-3)
            elif type(self.sweepparams) is dict:
                waveform, _ = self.station.awg.sweep_gate_virt(self.sweepparams, self.sweeprange, period=1e-3)
            self.datafunction = fpgaCallback_1d(self.station, waveform, self.Naverage, self.fpga_ch)
        elif type(self.sweepranges) is list:
            if type(self.sweepparams) is list:
                waveform, _ = self.station.awg.sweep_2D(self.station.fpga.sampling_frequency.get(), self.sweepparams, self.sweepranges, self.resolution)
            elif type(self.sweepparams) is dict:
                waveform, _ = self.station.awg.sweep_2D_virt(self.station.fpga.sampling_frequency.get(), self.sweepparams['gates_horz'], self.sweepparams['gates_vert'], self.sweepranges, self.resolution)
            self.datafunction = fpgaCallback_2d(self.station, waveform, self.Naverage, self.fpga_ch, self.resolution, self.diff_dir)

        self.lp.datafunction = self.datafunction
        self.lp.startreadout()

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
        Naverage_old = self.lp.datafunction.Naverage
        waittime_old = self.lp.datafunction.waittime
        self.lp.datafunction.waittime = Naverage * self.lp.data.size / self.station.fpga.sampling_frequency.get()
        self.lp.datafunction.Naverage = Naverage
        self.data = self.lp.datafunction()
        self.lp.datafunction.Naverage = Naverage_old
        self.lp.datafunction.waittime = waittime_old
        if self.data.ndim == 1:
            alldata, _ = makeDataset_sweep(self.data, self.sweepparams, self.sweepranges, gates=self.station.gates, loc_record={'label': 'videomode_1d_single'})
        if self.data.ndim == 2:
            alldata, _ = makeDataset_sweep_2D(self.data, self.station.gates, self.sweepparams, self.sweepranges, loc_record={'label': 'videomode_2d_single'})
        alldata.metadata = {'scantime': str(datetime.datetime.now()), 'station': self.station.snapshot(), 'allgatevalues': self.station.gates.allvalues()}
        alldata.metadata['Naverage'] = Naverage
        if hasattr(self.lp.datafunction, 'diff_dir'):
            alldata.metadata['diff_dir'] = self.lp.datafunction.diff_dir
        alldata.write(write_metadata=True)
        plotData(alldata, fig=None)
        self.alldata = alldata

    def Naverage_changed(self, value):
        """Set the value of Naverage."""
        self.lp.datafunction.Naverage = value
