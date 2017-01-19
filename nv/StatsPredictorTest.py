# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:50:35 2017

@author: Laurens
"""

import StatsPredictor
import numpy as np
import pandas as pd
import qcodes
import os

#%%
print('Generating Data')
data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T
data=data[:,0:6]
df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump'])
jumps = df[['gate jump', 'yellow jump']]

labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))
labels[labels==-1] = 5 #this makes it a bit nicer to handle

#%% Test with ignore 0
testSize=len(labels)-50
trainLabels = labels[:-testSize]
testLabels = labels[-testSize:]
sp=StatsPredictor(1,trainLabels)
hitTime = np.zeros((testSize,1))
for i in range(len(testLabels)):
    if testLabels[i] == 0:
        hitTime[i] = 1
    else:        
        hitTime[i] = list(sp.predictNext().argsort()[::-1]).index(testLabels[i])+2
        sp.foundNextCluster(testLabels[i])
print('Average search \"time\": ', str(np.mean(hitTime)))

#%% Test without ignore 0
trainLabels = labels[:-testSize]
testLabels = labels[-testSize:]
sp=StatsPredictor(1,trainLabels,False)
hitTime = np.zeros((testSize,1))
for i in range(len(testLabels)):
        hitTime[i] = list(sp.predictNext().argsort()[::-1]).index(testLabels[i])+1
        sp.foundNextCluster(testLabels[i])
print('Average search \"time\": ', str(np.mean(hitTime)))

#%% Basic commands for testing
from StatsPredictor import StatsPredictor
sp = StatsPredictor(40,labels)
preds = sp.predictNext()
preds2=sp.foundNextCluster(4)
preds3=sp.foundNextCluster(0)

