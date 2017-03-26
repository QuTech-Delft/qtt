# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:36:56 2016

@author: eendebakpt
"""

#%% Start the control widget
import sys
import platform
import qcodes
import qcodes as qc
from qtt.live_plotting import MeasurementControl
from qcodes.tests.instrument_mocks import DummyInstrument
import pyqtgraph as pg

mc = MeasurementControl()
mc.setGeometry(1700, 50, 300, 400)
mc.install_qcodes_hook()
mc.enable_measurements()


#%% Start a loop


instr = DummyInstrument()


qapp = pg.mkQApp()

loop = qc.Loop(instr.dac1[0:40:1], progress_interval=1, delay=.25).with_bg_task(task=pg.QtGui.QApplication.processEvents, min_delay=.05).each(instr.dac2)
ds = loop.run()
