# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:43:47 2017

@author: diepencjv
"""
#%%
from qtt.live_plotting import livePlot, fpgaCallback_1d, fpgaCallback_2d

#%%


class VideoMode:
    """ Controls the videomode tuning.

    Attributes:
        station (qcodes station): 
        sweepparams (string, list or dict)
        sweepranges (int or list)
        fpga_ch (int): 
        resolution (list): 
    """
    # TODO: implement optional sweep directions, i.e. forward and backward
    def __init__(self, station, sweepparams, sweepranges, fpga_ch, Naverage=25, resolution=[90, 90], diff_dir=None):
        self.station = station
        self.sweepparams = sweepparams
        self.sweepranges = sweepranges
        self.fpga_ch = fpga_ch
        self.Naverage = Naverage
        self.resolution = resolution
        self.diff_dir = None

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

        self.lp = livePlot(self.datafunction, self.station.gates, self.sweeparams, self.sweepranges)
        self.lp.startreadout()

        if hasattr(self.station, 'RF'):
            self.station.RF.on()

    def stop(self):
        """ Stops the plotting, AWG(s) and possibly RF. """
        self.lp.stopreadout()
        if self.station is not None:
            if hasattr(self.station, 'awg'):
                self.station.awg.stop()
            if hasattr(self.station, 'RF'):
                self.station.RF.off()
