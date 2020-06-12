# -*- coding: utf-8 -*-
""" Example script to show QTT capabilities

The script should be executed from an interactive environment such as Spyder.
An event loop should be running for the GUI elements.

@author: eendebakpt
"""

# %% Load packages
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from collections import OrderedDict
import pyqtgraph
_ = pyqtgraph.mkQApp()

import qcodes
from qcodes.data.data_set import DataSet
import qtt
from qtt.gui.parameterviewer import createParameterWidget
from qtt.algorithms.gatesweep import analyseGateSweep
from qtt.measurements.scans import scanjob_t
from qtt.instrument_drivers.virtual_gates import VirtualGates
from qtt import save_state
import qtt.measurements.videomode

import qtt.simulation.virtual_dot_array

datadir = tempfile.mkdtemp(prefix='qtt_example')
DataSet.default_io = qcodes.data.io.DiskIO(datadir)

# %% Create a virtual model for testing
#
# The model resembles the spin-qubit dot setup. The hardware consists of a virtual
# keithley, IVVI racks and a virtual gates object

nr_dots = 3
station = qtt.simulation.virtual_dot_array.initialize(reinit=True, nr_dots=nr_dots, maxelectrons=2)

keithley1 = station.keithley1
keithley3 = station.keithley3

# virtual gates for the model
gates = station.gates
model = station.model

# %% Setup measurement windows


mwindows = qtt.gui.live_plotting.setupMeasurementWindows(station, create_parameter_widget=False)
pv = createParameterWidget([gates, ])

logviewer = qtt.gui.dataviewer.DataViewer()
logviewer.show()

# %% Read out instruments
print('value: %f' % keithley3.readnext())
snapshotdata = station.snapshot()


# %% Simple 1D scan loop

param_left = station.model.bottomgates[0]
param_right = station.model.bottomgates[-1]
scanjob = scanjob_t({'sweepdata': dict({'param': param_right, 'start': -500, 'end': 0,
                                        'step': .8, 'wait_time': 3e-3}), 'minstrument': ['keithley3.amplitude']})
data1d = qtt.measurements.scans.scan1D(station, scanjob, location=None, verbose=1)


# %% Save the current state of the system to disk

save_state(station)

# %% Print the scanned data

print(data1d.default_parameter_name())

# %% Make a 2D scan
start = -500
scanjob = scanjob_t()
scanjob.add_sweep(param_right, start=start, end=start + 400, step=4., wait_time=0.)
scanjob.add_sweep(param_left, start=start, end=start + 400, step=5)
scanjob.add_minstrument(['keithley1.amplitude'])
data = qtt.measurements.scans.scan2D(station, scanjob)

gates.set(param_right, -300)
gates.set(param_left, -300)
gv = gates.allvalues()

# %% Fit 1D pinch-off scan:

adata = analyseGateSweep(data1d, fig=100)


# %% Make virtual gates
np.set_printoptions(precision=2, suppress=True)

crosscap_map = OrderedDict((
    ('VP1', OrderedDict((('P1', 1), ('P2', 0.56), ('P3', 0.15)))),
    ('VP2', OrderedDict((('P1', 0.62), ('P2', 1), ('P3', 0.593)))),
    ('VP3', OrderedDict((('P1', 0.14), ('P2', 0.62), ('P3', 1))))
))
virts = VirtualGates(qtt.measurements.scans.instrumentName('vgates'), gates, crosscap_map)
virts.print_matrix()

gates.resetgates(gv, gv, verbose=0)

virts.VP2.set(-60)

cc1 = virts.VP1()
cc2 = virts.VP2()
r = 80
scanjob = scanjob_t({'sweepdata': dict({'param': virts.VP1, 'start': cc1 - 100, 'end': cc1 +
                                        100, 'step': 4.}), 'minstrument': ['keithley1.amplitude'], 'wait_time': 0.})
scanjob['stepdata'] = dict({'param': virts.VP2, 'start': cc2 - r, 'end': cc2 + r, 'step': 2.})
data = qtt.measurements.scans.scan2D(station, scanjob)
gates.resetgates(gv, gv, verbose=0)

vgates = ['vSD1b'] + virts.vgates() + ['vSD1a']
pgates = ['SD1b'] + virts.pgates() + ['SD1a']
virts2 = qtt.instrument_drivers.virtual_gates.extend_virtual_gates(vgates, pgates, virts, name='vgates')

# %% Send data to powerpoint
print('add copy data to Powerpoint use the following:')
print('   qtt.utilities.tools.addPPT_dataset(data);')
if 0:
    qtt.utilities.tools.addPPT_dataset(data)

# %% Start videomode

digitizer = station.sdigitizer
station.awg = station.vawg


print('starting videomode in background...')
gates.P3.increment(40)
vm = qtt.measurements.videomode.VideoMode(station, ['P1', 'P2'], [160] * 2,
                                          minstrument=(digitizer.name, [1, 1]), resolution=[96, 96],
                                          diff_dir=[None, 'g'], name='physical gates')
vm.crosshair(True)
vm.stopreadout()
vm.updatebg()


# %%

s1 = qtt.measurements.scans.create_vectorscan(virts.VP1, 160)
s2 = qtt.measurements.scans.create_vectorscan(virts.VP2, 160)
scan_parameters = {'gates_horz': s1['param'], 'gates_vert': s2['param']}
vm_virtual = qtt.measurements.videomode.VideoMode(station, scan_parameters, [200, 180],
                                                  minstrument=(digitizer.name, [1, 1]), resolution=[96, 96],
                                                  diff_dir=[None, 'g'], name='virtual gates')
vm_virtual.crosshair(True)
vm_virtual.stopreadout()
vm_virtual.updatebg()


# %% Close all GUI elements
if not qtt.utilities.tools.is_spyder_environment():
    print('close all GUI elements')
    vm.close()
    vm_virtual.close()
    plt.close('all')
    pv.close()
    logviewer.close()

    qtt.gui.live_plotting.liveplotwindow.win.close()
    print('close all GUI elements: done')
