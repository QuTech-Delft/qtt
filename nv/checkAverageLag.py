# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:21:08 2017

@author: Laurens
"""

#%% Load packages
import numpy as np
import random
import sys,os
from theano import tensor as T
import numpy
from matplotlib import pyplot as plt
import qcodes
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score
from pandas.tools.plotting import andrews_curves


#%% Global settings
dataSelection = ['yellow jump','gate jump','yellow','gate']
lblsAsInput = True
lag = 100
keepNoClass = True


#%%
print('Generating Data')
data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T
data=data[:,0:6]
df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump'])
jumps = df[['gate jump', 'yellow jump']]

labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))
labels[labels==-1] = 5 #this makes it a bit nicer to handle

#%% Data needs to be scaled for almost any machine learning algorithm to work

# translate by mean and scale with std
datascaler= StandardScaler()
dataS = datascaler.fit_transform(data)
dfS=df.copy()
dfS[:]=datascaler.transform(df)

#%% Select the subset of data to use:
selectedData = dfS[dataSelection]
ran = range(0,selectedData.shape[0])

laggedData = np.zeros((len(dataSelection),selectedData.shape[0]-lag,lag))
for i in range(len(dataSelection)):
    d = pd.DataFrame(selectedData.iloc[:,i])
    laggedData[i,:,:] = np.concatenate([d.shift(i) for i in ran],axis=1)[lag:,:lag]

dataSet = np.dstack(laggedData)[:-1,:,:] # Remove final entry as we have no label for it
 
if lblsAsInput: # If we also want to use the clusters labels as input
    labelData = np.concatenate([pd.DataFrame(labels).shift(i) for i in ran],axis=1)[lag:,:lag]
    labelData = np.reshape(labelData,(labelData.shape[0],labelData.shape[1],1))[:-1,:,:]
    dataSet = np.dstack((dataSet,labelData))
    
#%% Create the labels
lbls = labels[lag+1:] #Crop the part of the labels we are not using
#lbls[lbls==-1] = 5 #Setting this class to 5 so it can be one hot encoded and more easily be cut off

if not keepNoClass: # to make training a little bit easier for now
    print('this is happening!')
    dataSet = dataSet[lbls<5,:,:] # Remove all the points that do not belong to a class
    lbls = lbls[lbls<5]
    
#lbls = OneHotEncoder(sparse=False).fit_transform(lbls.reshape(-1,1))

#lblCount = lbls.shape[1]

lblCount=len(np.unique(lbls))

#%% plot average cluster sequence
avgClustSeq = np.zeros((lblCount,lag))
sdClustSeq = np.zeros((lblCount,lag))
for i in range(lblCount):
    avgClustSeq[i,:] = np.mean(dataSet[lbls==i,:,4],axis=0)
    sdClustSeq[i,:] = np.sqrt(np.var(dataSet[lbls==i,:,4],axis=0))

plt.figure()
colors=['r','b','g','y']
for i in [1,3]:#range(lblCount):
    plt.plot(avgClustSeq[i],'-'+colors[i],label=i)
    plt.plot(avgClustSeq[i]+sdClustSeq[i],'--'+colors[i])
    plt.plot(avgClustSeq[i]-sdClustSeq[i],'--'+colors[i])
plt.legend()

#df=pd.DataFrame(dataSet[lbls%2==1,:,4])
#df['labels'] = pd.Series(lbls[lbls%2==1])
#
#plt.figure()
#andrews_curves(df,'labels')

#%% plot average actual values
avgYfSeq = np.zeros((lblCount,lag))
sdYfSeq = np.zeros((lblCount,lag))
avgGvSeq = np.zeros((lblCount,lag))
sdGvSeq = np.zeros((lblCount,lag))
for i in range(lblCount):
    avgYfSeq[i,:] = np.mean(dataSet[lbls==i,:,2],axis=0)
    sdYfSeq[i,:] = np.sqrt(np.var(dataSet[lbls==i,:,2],axis=0))
    avgGvSeq[i,:] = np.mean(dataSet[lbls==i,:,3],axis=0)
    sdGvSeq[i,:] = np.sqrt(np.var(dataSet[lbls==i,:,3],axis=0))
    
plt.figure()
plt.title('Average Yellow Frequency Sequence')
colors=['r','b','g','y']
for i in [1,3]:#range(lblCount):
    plt.plot(avgYfSeq[i],'-'+colors[i],label=i)
    plt.plot(avgYfSeq[i]+sdYfSeq[i],'--'+colors[i])
    plt.plot(avgYfSeq[i]-sdYfSeq[i],'--'+colors[i])
plt.legend()

plt.figure()
plt.title('Average Gate Voltage Sequence')
colors=['r','b','g','y']
for i in [1,3]:#range(lblCount):
    plt.plot(avgGvSeq[i],'-'+colors[i],label=i)
    plt.plot(avgGvSeq[i]+sdGvSeq[i],'--'+colors[i])
    plt.plot(avgGvSeq[i]-sdGvSeq[i],'--'+colors[i])
plt.legend()


