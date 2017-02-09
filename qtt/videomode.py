# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:43:47 2017

@author: diepencjv
"""
#%%
from qtt.live_plotting import livePlot, fpgaCallback_2d

#%%
class VideoMode:
    #TODO: implement optional sweep directions, i.e. forward and backward
    def __init__(self, station, datafunction):
        self.lp = livePlot(station.gates)
        self.lp.datafunction = datafunction
        self.station = station
        
    def run(self):
        self.lp.startreadout()
            
    def stop(self):
        self.lp.stopreadout()
        self.station.awg.stop()
        self.station.RF.off()