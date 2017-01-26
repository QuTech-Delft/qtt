# -*- coding: utf-8 -*-
"""
Load data and visualize results

@author: eendebakpt
"""


# coding: utf-8

#%% Load packages
import sys,os
import numpy as np
from matplotlib import pyplot as plt
import copy

from nvtools.nvtools import extract_data
try:
    from qtt import pgeometry
except:
    import pgeometry

#%% Load data

datadir='/home/eendebakpt/data/qutech/nv/nv3'
folder='FrequencylogPippinSIL3'


files=sorted(pgeometry.findfiles(os.path.join(datadir, folder), '.*dat'))
files=[os.path.join(os.path.join(datadir, folder), f) for f in files]
print(files)
    
# files[0] = '20160805_160044_frequency_logger.dat'
# files[1] = '20160809_132333_frequency_logger.dat'
# files[2] = '20160811_145538_frequency_logger.dat'

gate_scaling = 1

data = list(range(len(files)))
for i in range(len(files)):
    data[i] = extract_data(files[i], gate_scaling)
    print('data %d: length %s' % (i, data[i][0].shape))


#%% Remove empty intervals (i.e. idle time) from data 
stitchIndices,stitchTimeDiff = list(range(len(data))),list(range(len(data)))

for i,d in zip(range(len(data)),data):
    timeDiff = np.diff(d)
    stitchTimeDiff[i] = np.append(np.array([0]),timeDiff[timeDiff > 30*60])
    stitchTimeDiff[i] = np.cumsum(stitchTimeDiff[i])
    # find times whenever we were idle for more than 30 minutes
    ind1,ind2 = np.where( timeDiff > 30*60 )
    stitchIndices[i] = np.append(ind2[ind1==0],np.array([len(d[0])-1]))
    stitchIndices[i] = np.append(stitchIndices[i][0],np.diff(stitchIndices[i]))
    
    # create an array that corrects for the idle times
    subtraction_arr = np.array([0])
    for j,inds,diff in zip(range(len(stitchIndices[i])),stitchIndices[i],stitchTimeDiff[i]):
        subtraction_arr = np.append(subtraction_arr,np.ones(inds)*diff)
    
    # manipulate the original time series by setting it initially to 0 and adjusting the idle time with the subtraction array
    data[i][0] = np.subtract(data[i][0],subtraction_arr)-data[i][0][0]
    


#%% Stitch all datafiles together
allData = [np.array([]),np.array([]),np.array([]),np.array([])]
allStitchIndices = []
for i in range(len(data)):
    if i ==0:
        allData = copy.deepcopy(data[i])
    else:
        for j,d in zip(range(len(data[i])),data[i]):
            if j ==0:
                addtotime = allData[0][-1]
            else:
                addtotime = 0
            allData[j] = np.append(allData[j],d+addtotime)
#     allData = [allData; [(cut{i}(:,1) - cut{i}(1,1) + allData(end,1)) cut{i}(:,2:end)]];

# allData = allData(2:end,:);
# allStitchIndices = allStitchIndices(2:end);

adata=np.array(allData).T
             
#%% Show some data

fig = plt.figure(1000, figsize=(17,6));plt.clf()
ax = plt.subplot(211)
plt.plot(allData[0],allData[2])
ax.set_xlabel('elapsed time (s)')
ax.set_ylabel('Gate voltage (mV)')
ax2 = plt.subplot(212)
plt.plot(allData[0],allData[1])
ax2.set_xlabel('elapsed time (s)')
ax2.set_ylabel('Yellow frequency (GHz)')
plt.show()

#%%
from nvtools.nvtools import plotSection

def f(plotidx, fig=100, *args, **kwargs):
    verbose = kwargs.get('verbose', 1)
    if verbose:
        print('plotidx = %s' % plotidx)
    plt.figure(fig)
    plt.clf()
    #dataidx = int(jumpdata[plotidx, 6])
    dataidx=plotidx
    plotSection(adata, list(range(dataidx - 60, dataidx + 100)), jumps=None, si=dataidx)
    plt.pause(1e-4)
    plt.figure(fig+1);  plt.clf()
    plotSection(adata, list(range(dataidx - 60, dataidx + 100)), jumps=None, mode='freq', si=dataidx)
    plt.pause(1e-4)
    
    
pc = pgeometry.plotCallback(func=f, xdata=allData[0], ydata=allData[2])
pc.connect(fig)

#%%
if 0:
    plt.figure(300); plt.clf()
    offset=2
    si=3000
    plotSection(allData, list(range(si-offset, si-offset+100)), None, mode='gate')
    #cid = ScatterFig.canvas.mpl_connect('button_press_event', pc)
