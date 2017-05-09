# -*- coding: utf-8 -*-
""" ???
Created on Wed Aug 31 16:19:30 2016

@author: diepencjv
"""

#%%


def sweep_horz(station, gate, sweeprange, risetime, Naverage):
    station.awg.sweep_gate(gate, sweeprange, risetime)

    totalpoints, DataRead_ch1, DataRead_ch2 = station.fpga.readFPGA(Naverage=Naverage)

    station.awg.stop()

    return totalpoints, DataRead_ch1, DataRead_ch2
