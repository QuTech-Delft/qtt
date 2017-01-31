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
import qcodes
import nvtools
from nvtools.nvtools import extract_data, remove_empty_intervals
from qtt import pgeometry

import qcodes
#%% Load data

os.chdir(qcodes.config['user']['nvDataDir'])
files = []
NV1 = False
if NV1:
    timestamp_list = ['033735','144747','145532','222427','234411']  
    folder = r'FrequencylogPippinSIL2//'
    gate_scaling = 1
    
    attractmV = 15 # mV
    attractFreq = 40e-3 # MHz
else:
    timestamp_list = ['115652','132333','145538','160044']
    folder = r'Frequencylog111no2SIL2//'
    gate_scaling = 1e3
    
for t in timestamp_list:
    files.append(folder+t+'_frequency_logger.dat')
    
print( files)

data = list(range(len(files)))
for i in range(len(files)):
    data[i] = extract_data(files[i], gate_scaling)
    print('data %d: length %s' % (i, data[i][0].shape))


#%% Remove empty intervals (i.e. idle time) from data and merge datasets

allData,_=remove_empty_intervals(data)
adata=np.array(allData).T

#%% Show some data

fig = plt.figure(figsize=(17,6))
ax = plt.subplot(211)
plt.plot(allData[0],allData[2])
ax.set_xlabel('elapsed time (s)')
ax.set_ylabel('Gate voltage (mV)')
ax2 = plt.subplot(212)
plt.plot(allData[0],allData[1])
ax2.set_xlabel('elapsed time (s)')
ax2.set_ylabel('Yellow frequency (GHz)')
plt.show()

#reload(nvtools.nvtools)
from nvtools.nvtools import plotSection, nv_plot_callback

f = lambda plotidx, **kwargs: nv_plot_callback(plotidx, adata, **kwargs)
pc = pgeometry.plotCallback(func=f, xdata=allData[0], ydata=allData[2])
pc.connect(fig)

# In[8]:

# % Analyze jumps --> create jumps from the data
time = allData[0]
yellow = allData[1]
gate = allData[2]
newfocus = allData[3]
dt = np.median(np.diff(time))

jumpSelect = (np.diff(time)>3*dt) & (np.diff(time)<30*60) # take three times the median as a qualifier for a jump
jumpIndex = np.nonzero(jumpSelect)[0]
jumpStart = time[np.append(jumpSelect, False)]
jumpEnd = time[np.append(False, jumpSelect)]
jumpGate = gate[np.append(False, jumpSelect)] - gate[np.append(jumpSelect, False)]
jumpYellow = yellow[np.append(False, jumpSelect)] - yellow[np.append(jumpSelect, False)]
jumpRelative = jumpGate / jumpYellow
jumpSelect=np.append(jumpSelect, False)

#%% Save data for other scripts

xx=np.vstack( ( time[jumpSelect], gate[jumpSelect], yellow[jumpSelect], newfocus[jumpSelect], jumpGate, jumpYellow, jumpIndex) )

if 1:                 
    # save data
    #np.save('/home/eendebakpt/tmp/jdata.npy', xx)
    np.save(os.path.join(qcodes.config['user']['nvDataDir'],'jdata2.npy'), xx)
    np.save(os.path.join(qcodes.config['user']['nvDataDir'],'jdata-alldata2.npy'), allData)

    plt.figure(1); plt.clf()
    plt.plot( xx[0,:], xx[5,:], '.b')

# In[9]:

fig=  plt.figure()
ax = plt.subplot(111)


plt.plot(yellow, newfocus, 'x')

fig=plt.figure()
plt.plot(newfocus, '.b'); plt.ylabel('Newfocus')

fig=plt.figure()
plt.plot(time, '.b'); plt.ylabel('Time')



#%% do we identify jumps correctly?
#
# Plot a section to look at the selected jumps
plot_range = [300,600]
x = allData[0][plot_range[0]:plot_range[1]]
y = allData[1][plot_range[0]:plot_range[1]]
y2 = allData[2][plot_range[0]:plot_range[1]]
plot_select = jumpSelect & (allData[0]<x[-1]) & (allData[0]>x[0])
fig = plt.figure(figsize=(17,6))
ax = plt.subplot(211)
plt.plot(x,y2,'x-')
plt.plot(allData[0][plot_select],allData[2][plot_select],'ro')
# ax.set_xlim([0,1000])
ax.set_xlabel('elapsed time (s)')
ax.set_ylabel('Gate voltage (V)')
ax2 = plt.subplot(212)
plt.plot(x,y,'x-')
plt.plot(allData[0][plot_select],allData[1][plot_select],'ro')
ax2.set_xlabel('elapsed time (s)')
# ax2.set_xlim([0,1000])
ax2.set_ylabel('Yellow frequency (GHz)')
plt.show()

#%% Plot correlations between gate and yellow jumps

from nvtools.nvtools import add_attraction_grid

fig=  plt.figure()
ax = plt.subplot(111)
add_attraction_grid(ax, attractmV, attractFreq, zorder=0)

b = jumpGate/jumpYellow
plt.plot(jumpGate, jumpYellow, '.', zorder=3)
plt.xlabel('Gate [mV]');plt.ylabel('Frequency jump [GHz]')

ax.set_xlabel('Voltage jump on gate (mV)')
ax.set_ylabel('Frequency jump on yellow (GHz)')
plt.title('Correlation between gate and yellow jumps.')



# green points

xx=np.vstack((jumpGate, jumpYellow) )

from qtt import pmatlab
from qtt.pmatlab import points_in_polygon
if 0:
    rr=np.array([[-24.2,7.25],[0.6796,.4297]])
    pmatlab.plotPoints(pmatlab.region2poly(rr), '.-g')
    pp=pmatlab.region2poly(rr)
    idx=points_in_polygon(xx.T, pp.T)==1
    pmatlab.plotPoints(xx[:, idx], '.g')
    
    print('# green: %d' % np.sum(idx==1))


#%% Correlation between a jump and the next jump

fig=  plt.figure()
ax = plt.subplot(211)
plt.plot(jumpYellow[0:-1], jumpYellow[1:], 'x')

ax.set_xlabel('Previous jump')
ax.set_ylabel('Frequency jump on yellow (GHz)')

ax = plt.subplot(212)
plt.plot(jumpGate[0:-1], jumpGate[1:], 'x')

ax.set_xlabel('Previous jump')
ax.set_ylabel('Frequency jump on gate')
plt.title('Correlation betweenyellow jumps.')


#%% Select small jumps 

print(jumpGate.shape)
smallIdx=np.nonzero(np.abs(np.array(jumpGate))<15 & ( np.abs(np.array(jumpYellow) ) <2) )[0]
jumpSelect
jumpIdx=np.nonzero(jumpSelect)[0]
#print(jumpIdx)
#smallIdx=jumpIdx[idx]
print(smallIdx)

si=24
jumpGate[si]


# In[14]:
from imp import reload
import nvtools
reload(nvtools)
reload(nvtools.nvtools)
from nvtools.nvtools import plotSection

       
offset=10
fig=plt.figure(figsize=(16,12))
for ii in range(16):
    ax=plt.subplot(4,4,ii+1)
    si=smallIdx[ii]
    plotSection(allData, list(range(si-offset, si-offset+100) ), jumpSelect, mode='gate')
    plt.plot(allData[0][si], allData[2][si], '.y', markersize=12)

#%%
fig=plt.figure(100, figsize=(12,8)) ; plt.clf()
offset=20
ji=smallIdx[4]
si=jumpIdx[ji]
print('index %d' % si)
plt.subplot(211)
plotSection(allData, range(si-offset, si-offset+100), mode='gate')
plt.plot(allData[0][si], allData[2][si], '.y', markersize=12)
plt.title('jump %.1f mV, %.2f GHz' % (jumpGate[ji], jumpYellow[ji]))
plt.subplot(212)
plotSection(allData, range(si-offset, si-offset+100), mode='yellow')
plt.plot(allData[0][si], yellow[si], '.y', markersize=12)

print('gate: %f' % (gate[si]-gate[si+1], ))

#%% Notes

# * To correctly predict jumps we need the ground truth. The real ground truth is not available, only the results after several tries. Maybe we can extrapolate back in time?
# * What is the newfocus variable?
# * The actual number of jumps (304) is not very high for machine learning...
# * The correlation plot shows structure between the gate and yellow jumps. There are many data points near (0,0). How is this possible? These should not have been jumps?
# The plot is symmetric. Can we distinct based on the previous absolute value?
# * What is the domain of attraction? This is needed to determinate either a search strategy or a list of best locations.
# * What is our goal? E.g. number of attempts before find a correct pair of frequencies smaller then 20?
# 

# In[12]:

print(gate.shape)
print(gate[jumpSelect].shape)
print(jumpGate.shape)


# In[84]:

jgate=gate[jumpSelect]

fig=  plt.figure()
ax = plt.subplot(211)
plt.plot(jgate, jumpGate, 'xb')
ax = plt.subplot(212)
plt.plot(jgate, jumpYellow, 'xr')


# In[ ]:


