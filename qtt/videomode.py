# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:43:47 2017

@author: diepencjv
"""
#%%
from qtt.live_plotting import livePlot, fpgaCallback_1d, fpgaCallback_2d
from qtt.tools import connect_slot
import qtpy.QtWidgets as QtWidgets
from qtt.scans import scan1Dfast, scan2Dturbo

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
        """Do a single scan with 1000 averaging.

        Do a single scan at high averaging. Note: this does not yet support the
        usage of linear combinations of gates (a.k.a. virtual gates).    
        """
        scanjob = {'Naverage': 1000}
        class dummy(object):
            pass
        dummy_sd = dummy()
        dummy_sd.fpga_ch = self.fpga_ch
        scanjob['sd'] = dummy_sd
        if type(self.sweepparams) is str:
            scanjob['sweepdata'] = {'param': self.sweepparams}
            scanjob['sweepdata']['start'] = self.station.gates.get(self.sweepparams) - self.sweepranges / 2
            scanjob['sweepdata']['end'] = scanjob['sweepdata']['start'] + self.sweepranges
            self.alldata = scan1Dfast(self.station, scanjob)
        if type(self.sweepparams) is list:
            scanjob['stepdata'] = {'param': self.sweepparams[0]}
            scanjob['stepdata']['start'] = self.station.gates.get(self.sweepparams[0]) - self.sweepranges[0] / 2
            scanjob['stepdata']['end'] = scanjob['stepdata']['start'] + self.sweepranges[0]
            scanjob['sweepdata'] = {'param': self.sweepparams[1]}
            scanjob['sweepdata']['start'] = self.station.gates.get(self.sweepparams[1]) - self.sweepranges[1] / 2
            scanjob['sweepdata']['end'] = scanjob['sweepdata']['start'] + self.sweepranges[1]
            scanjob['resolution'] = self.resolution
            self.alldata = scan2Dturbo(self.station, scanjob)

        if type(self.sweepparams) is dict:
            raise NotImplementedError('Single scan with linear combinations of gates is not implemented.')
