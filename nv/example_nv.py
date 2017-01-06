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

#%% Load data

os.chdir('C:\\Users\\Laurens\\Documents\\qtechData')
files = []
NV1 = True
if NV1:
    timestamp_list = ['033735','144747','145532','222427','234411']  
    folder = r'FrequencylogPippinSIL2//'
    gate_scaling = 1
else:
    timestamp_list = ['115652','132333','145538','160044']
    folder = r'Frequencylog111no2SIL2//'
    gate_scaling = 1e3
    
for t in timestamp_list:
    files.append(folder+t+'_frequency_logger.dat')
    
# files[0] = '20160805_160044_frequency_logger.dat'
# files[1] = '20160809_132333_frequency_logger.dat'
# files[2] = '20160811_145538_frequency_logger.dat'
print( files)

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


# In[8]:

# % Analyze jumps --> create jumps from the data
time = allData[0]
yellow = allData[1]
gate = allData[2]
newfocus = allData[3]
dt = np.median(np.diff(time))

jumpSelect = (np.diff(time)>3*dt) & (np.diff(time)<30*60) # take three times the median as a qualifier for a jump
jumpStart = time[np.append(jumpSelect, False)]
jumpEnd = time[np.append(False, jumpSelect)]
jumpGate = gate[np.append(False, jumpSelect)] - gate[np.append(jumpSelect, False)]
jumpYellow = yellow[np.append(False, jumpSelect)] - yellow[np.append(jumpSelect, False)]
jumpRelative = jumpGate / jumpYellow


#%% Save data for other scripts

xx=np.vstack( ( time[jumpSelect], gate[jumpSelect], yellow[jumpSelect], newfocus[jumpSelect], jumpGate, jumpYellow) )

if 0:                 
    # save data
    #np.save('/home/eendebakpt/tmp/jdata.npy', xx)
    np.save('jdata.npy', xx)

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
plot_length = 100
x = allData[0][:plot_length]
y = allData[1][:plot_length]
y2 = allData[2][:plot_length]
plot_select = jumpSelect & (allData[0]<x[-1])[:-1]
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
fig=  plt.figure()
ax = plt.subplot(111)

b = jumpGate/jumpYellow

plt.plot(jumpGate, jumpYellow, 'x')

ax.set_xlabel('Voltage jump on gate (mV)')
ax.set_ylabel('Frequency jump on yellow (GHz)')
plt.title('Correlation between gate and yellow jumps.')

# green points

xx=np.vstack((jumpGate, jumpYellow) )

from qtt import pmatlab
from pmatlab import points_in_polygon
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

def plotSection(allData, idx, mode='gate'):
    """ Helper function to plot a section of data """
    x = allData[0][idx]
    y = allData[1][idx]
    y2 = allData[2][idx]
    v=np.zeros( len(allData[0]) ).astype(bool); v[idx]=1
    plot_select = jumpSelect & v[:-1]
    ax=plt.gca()
    if mode=='gate':
        plt.plot(x,y2,'x-')
        plt.plot(allData[0][plot_select],allData[2][plot_select],'ro')
        ax.set_xlabel('elapsed time (s)')
        ax.set_ylabel('Gate voltage (V)')
    else:
        plt.plot(x,y,'x-')
        plt.plot(allData[0][plot_select],allData[1][plot_select],'ro')
        ax2.set_xlabel('elapsed time (s)')
        # ax2.set_xlim([0,1000])
        ax2.set_ylabel('Yellow frequency (GHz)')
        
offset=10
fig=plt.figure(figsize=(16,12))
for ii in range(16):
    ax=plt.subplot(4,4,ii+1)
    si=smallIdx[ii]
    plotSection(allData, range(si-offset, si-offset+100), mode='gate')
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


