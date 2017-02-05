# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:50:35 2017

@author: Laurens
"""

from nv.StatsPredictor import StatsPredictor
import numpy as np
import pandas as pd
import qcodes
import os
import copy
from nvtools.nvtools import extract_data

#%% Load data for the jumps
#print('Generating Data')
#data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T
#data=data[:,0:6]
#df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump'])
#jumps = df[['gate jump', 'yellow jump']]
#
#labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))
#labels[labels==-1] = 5 #this makes it a bit nicer to handle

#%% Load raw data
os.chdir(qcodes.config['user']['nvDataDir'])
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

print( files)

data = list(range(len(files)))
for i in range(len(files)):
    data[i] = extract_data(files[i], gate_scaling)
    print('data %d: length %s' % (i, data[i][0].shape))


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

time = allData[0]
yellow = allData[1]
gate = allData[2]

    
#%% basic use
d=np.vstack((time,yellow,gate)).T
trainData=d[:10000,:]
testData=d[10000:,:]
propperFitted=False

#%% Test fitting (This takes a while!)
sp=StatsPredictor(10,verbose=True)
sp.fit(trainData)
propperFitted=True

#%% Actual usecase test
if not propperFitted:
    sp=StatsPredictor(1,verbose=True)
    sp.fit(trainData,quick=True)

for i in range(testData.shape[0]):
    sp.predictNext()
    #Here you would normally use this prediction to find the new value
    sp.foundNextValue(testData[i,:])

#%% Find best possible clustering using all data
sp=StatsPredictor(1)
sp.fit(d)
sp.plotClustering()