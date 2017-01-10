# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:16:13 2017

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

labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))

#%%
print('Generating Data')
data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T

df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump'])
jumps = df[['gate jump', 'yellow jump']]
#plt.figure(300); plt.clf()
#df.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), linewidths=0)

#%% Data needs to be scaled for almost any machine learning algorithm to work

# translate by mean and scale with std
datascaler= StandardScaler()
dataS = datascaler.fit_transform(data)
dfS=df.copy()
dfS[:]=datascaler.transform(df)

Xbase=dataS[:,4:] # base data
datascalerBase = StandardScaler().fit(data[:,4:])
x=dataS[:,4]
y=dataS[:,5]

#%% Create data set with 100 data points -> 1 label
lag = 100
ran = range(0,len(jumps[['gate jump']]))
lagSquare = np.concatenate([jumps[['gate jump']].shift(i) for i in ran],axis=1)
gateSet=lagSquare[lag:,:lag]
lagSquare = np.concatenate([jumps[['yellow jump']].shift(i) for i in ran],axis=1)
yellowSet=lagSquare[lag:,:lag]
#%%
dataSet = np.dstack((gateSet,yellowSet))[:-1,:,:] #I don't know the label for the final sequence, so drop it
lbls = labels[101:]
lbls[lbls==-1] = 5 #Setting this class to 5 so it can be one hot encoded and more easily be cut off
if 0: # to make training a little bit easier for now
    dataSet = dataSet[lbls<5,:,:] # Remove all the points that do not belong to a class
    lbls = lbls[lbls<5]

if 0: # Throw out the 0 cluster
    dataSet = dataSet[lbls>0,:,:]
    lbls=lbls[lbls>0]

if 1: # Only classify 0 cluster vs not 0 cluster
    lbls[lbls>0] = 1
    
lbls = OneHotEncoder(sparse=False).fit_transform(lbls.reshape(-1,1))

lblCount = lbls.shape[1]

#%% Create a train and test set
X_train, X_test, y_train, y_test = train_test_split(dataSet,lbls, stratify=lbls)



#%% Define & train a basic LSTM
model = Sequential()
model.add(GRU(10, input_shape=(100,2)))
model.add(Dense(lblCount, activation='softmax'))

optimiser = Adam(lr=0.001)
model.compile(optimiser, loss='binary_crossentropy', metrics=['accuracy'])
hist=model.fit(X_train, y_train, batch_size=1, nb_epoch=10, validation_split=0.2, verbose=2).history

#%% Test the basic LSTM
yhat = model.predict_proba(X_test)
ap = average_precision_score(y_test,yhat)
yhat = model.predict_classes(X_test)
ac = accuracy_score(np.argmax(y_test,axis=1),yhat)