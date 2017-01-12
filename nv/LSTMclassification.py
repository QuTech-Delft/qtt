# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:50:54 2017

@author: Laurens
"""

#%% Load packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys,os
from theano import tensor as T


import numpy
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, SimpleRNN
from keras.utils import np_utils
from keras.optimizers import Adam

from matplotlib import pyplot as plt
import qcodes
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score

#%% Global settings
dataSelection = ['yellow jump','gate jump']
lblsAsInput = True
lag = 100
keepNoClass = False
keepZeroCluster = True
zeroOrNotZero = False # Classify only zero cluster vs not-zero cluster (so don't remove that zero cluster)
sequentialTesting = False # The stateful LSTM will likely work better with this set to True
LSTMtype = 0

batch_size = 1
nb_epochs = 300
learningRate = 0.00001

#%%
print('Generating Data')
data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T

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
lbls[lbls==-1] = 5 #Setting this class to 5 so it can be one hot encoded and more easily be cut off

if ~keepNoClass: # to make training a little bit easier for now
    dataSet = dataSet[lbls<5,:,:] # Remove all the points that do not belong to a class
    lbls = lbls[lbls<5]

if ~zeroOrNotZero: # Only classify 0 cluster vs not 0 cluster
    if ~keepZeroCluster:
        print('You should keep the zero cluster, otherwise this will not work')
        print('I am keeping the zero cluster for you!!')
        keepZeroCluster=True
        
    lbls[lbls>0] = 1

if ~keepZeroCluster: # Throw out the 0 cluster
    dataSet = dataSet[lbls>0,:,:]
    lbls=lbls[lbls>0]
    
lbls = OneHotEncoder(sparse=False).fit_transform(lbls.reshape(-1,1))

lblCount = lbls.shape[1]

#%% Split into a train- and testset
if ~sequentialTesting:
    X_train, X_test, y_train, y_test = train_test_split(dataSet,lbls, stratify=lbls)
else: 
    X_train = dataSet[:1000,:,:]
    X_valid = dataSet[1000:1100,:,:] #Since I cannot take a random subset as validation set in the case of sequential testing
    X_test = dataSet[1100:,:,:]
    y_train = lbls[:1000,:]
    y_valid = dataSet[1000:1100,:,:]
    y_test = lbls[1100:,:]

#%% All the different LSTM's

'''' WIP
TODO:
    Add the LSTM's
    Add the testing
    Add the plotting
    Add a part that stores all the parameters in a CSV file
    Add a PCA part, perhaps that will improve learning (as the clusters are diagonal)
    Add more regularisation (dropout?)
''''

