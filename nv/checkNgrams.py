# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:21:58 2017

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
colors=['r','b','g','y']

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

#%% Show preceding histograms
df=pd.DataFrame()
prevLbls = dataSet[:,0,4]

plt.figure()
for i in range(4):
    p=plt.subplot(2,2,i+1)
    p.set_title('cluster ' + str(i+1) + ' followed by:')
    plt.hist(lbls[prevLbls==(i+1)],bins=np.arange(lblCount+2)-.5,color=colors[i])
    plt.xlim([1,5])
    plt.xticks(range(1,lblCount))
    plt.xlim([-1, lblCount])

plt.figure()
p=plt.subplot(121)
p.set_title('cluster 0 followed by:')
plt.hist(lbls[prevLbls==0],bins=np.arange(lblCount+2)-.5)
plt.xticks(range(lblCount+1))
plt.xlim([-1, lblCount])
p=plt.subplot(122)
p.set_title('cluster -1 followed by:')
plt.hist(lbls[prevLbls==5],bins=np.arange(lblCount+2)-.5,color='r')
plt.xticks(range(lblCount+1))
plt.xlim([-1, lblCount])

#%% Same plot but without the 0 cluster
lbls=labels[labels>0]
prevLbls=lbls[:-1]
lbls = lbls[1:]

plt.figure()
for i in range(4):
    p=plt.subplot(2,2,i+1)
    p.set_title('cluster ' + str(i+1) + ' followed by:')
    plt.hist(lbls[prevLbls==(i+1)],bins=np.arange(lblCount+2)-.5,color=colors[i])
    plt.xlim([1,5])
    plt.xticks(range(1,lblCount))
    plt.xlim([-1, lblCount])

plt.figure()
p=plt.subplot(121)
p.set_title('cluster 0 followed by:')
plt.hist(lbls[prevLbls==0],bins=np.arange(lblCount+2)-.5)
plt.xticks(range(lblCount+1))
plt.xlim([-1, lblCount])
p=plt.subplot(122)
p.set_title('cluster -1 followed by:')
plt.hist(lbls[prevLbls==5],bins=np.arange(lblCount+2)-.5,color='r')
plt.xticks(range(lblCount+1))
plt.xlim([-1, lblCount])
