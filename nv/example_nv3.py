# -*- coding: utf-8 -*-
"""
Load data and visualize results

@author: eendebakpt
"""


# coding: utf-8

#%% Load packages
from imp import reload
import sys
import os
import numpy as np
from matplotlib import pyplot as plt

from nvtools.nvtools import extract_data
from qtt import pgeometry
import nvtools
from nvtools.nvtools import extract_data, remove_empty_intervals

#%% Load data

datadir = '/home/eendebakpt/data/qutech/nv/nv3'
folder = 'FrequencylogPippinSIL3'


files = sorted(pgeometry.findfiles(os.path.join(datadir, folder), '.*dat'))
files = [os.path.join(os.path.join(datadir, folder), f) for f in files]
print(files)

gate_scaling = 1

data = list(range(len(files)))
for i in range(len(files)):
    data[i] = extract_data(files[i], gate_scaling)
    print('data %d: length %s' % (i, data[i][0].shape))


#%% Remove empty intervals (i.e. idle time) from data
allData, _ = remove_empty_intervals(data)
adata = np.array(allData).T

#%% Show some data

fig = plt.figure(1000, figsize=(17, 6))
plt.clf()
ax = plt.subplot(211)
plt.plot(allData[0], allData[2])
ax.set_xlabel('elapsed time (s)')
ax.set_ylabel('Gate voltage (mV)')
ax2 = plt.subplot(212)
plt.plot(allData[0], allData[1])
ax2.set_xlabel('elapsed time (s)')
ax2.set_ylabel('Yellow frequency (GHz)')
plt.show()

#%%
reload(nvtools.nvtools)
from nvtools.nvtools import plotSection, nv_plot_callback

f = lambda plotidx, **kwargs: nv_plot_callback(plotidx, adata, **kwargs)
f(0)
pc = pgeometry.plotCallback(func=f, xdata=allData[0], ydata=allData[2])
pc.connect(fig)

pgeometry.tilefigs([1000, 100], [1, 2])

#%%
if 0:
    plt.figure(300)
    plt.clf()
    offset = 2
    si = 3000
    plotSection(allData, list(range(si - offset, si - offset + 100)), None, mode='gate')
    #cid = ScatterFig.canvas.mpl_connect('button_press_event', pc)
