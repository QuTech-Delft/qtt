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
data=data[:, 0:6]

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
ran = range(0,len(dfS[['gate jump']]))
lagSquare = np.concatenate([dfS[['gate jump']].shift(i) for i in ran],axis=1)
gateSet=lagSquare[lag:,:lag]
lagSquare = np.concatenate([dfS[['yellow jump']].shift(i) for i in ran],axis=1)
yellowSet=lagSquare[lag:,:lag]
#%%
labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))
dataSet = np.dstack((gateSet,yellowSet))[:-1,:,:] #I don't know the label for the final sequence, so drop it
lbls = labels[lag+1:]
lbls[lbls==-1] = 5 #Setting this class to 5 so it can be one hot encoded and more easily be cut off
if 1: # to make training a little bit easier for now
    dataSet = dataSet[lbls<5,:,:] # Remove all the points that do not belong to a class
    lbls = lbls[lbls<5]

if 0: # Throw out the 0 cluster
    dataSet = dataSet[lbls>0,:,:]
    lbls=lbls[lbls>0]

if 0: # Only classify 0 cluster vs not 0 cluster
    lbls[lbls>0] = 1
    
lbls = OneHotEncoder(sparse=False).fit_transform(lbls.reshape(-1,1))

lblCount = lbls.shape[1]

#%% Create a train and test set
X_train, X_test, y_train, y_test = train_test_split(dataSet,lbls, stratify=lbls)

#%% Define & train a basic LSTM
model = Sequential()
model.add(GRU(10, input_shape=(lag,2), return_sequences=True))
model.add(GRU(10))
model.add(Dense(lblCount, activation='softmax'))

optimiser = Adam(lr=0.000003)
model.compile(optimiser, loss='binary_crossentropy', metrics=['accuracy'])
hist=model.fit(X_train, y_train, batch_size=5, nb_epoch=500, validation_split=0.2, verbose=2).history

#%% Test the basic LSTM
yhat = model.predict_proba(X_test)
ap = average_precision_score(y_test,yhat)
yhatt = model.predict_classes(X_test)
ac = accuracy_score(np.argmax(y_test,axis=1),yhatt)
print('\nAP: ', ap,'   ac: ', ac)

#%% Plot the history
plt.figure()
plt.plot(hist['loss'],label='Train loss')
plt.plot(hist['val_loss'],label='Validation loss')
plt.legend()

#%% LSTM using both the jumps and the absolute values

#%% Create data set with 100 data points -> 1 label
lag = 25
ran = range(0,len(dfS[['gate jump']]))
lagSquare = np.concatenate([dfS[['gate jump']].shift(i) for i in ran],axis=1)
gateJumpSet=lagSquare[lag:,:lag]
lagSquare = np.concatenate([dfS[['yellow jump']].shift(i) for i in ran],axis=1)
yellowJumpSet=lagSquare[lag:,:lag]
lagSquare = np.concatenate([dfS[['gate']].shift(i) for i in ran],axis=1)
gateSet=lagSquare[lag:,:lag]
lagSquare = np.concatenate([dfS[['yellow']].shift(i) for i in ran],axis=1)
yellowSet=lagSquare[lag:,:lag]
#%%
labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))
dataSet = np.dstack((gateSet,yellowSet,gateJumpSet,yellowJumpSet))[:-1,:,:] #I don't know the label for the final sequence, so drop it
lbls = labels[lag+1:]
lbls[lbls==-1] = 5 #Setting this class to 5 so it can be one hot encoded and more easily be cut off
if 1: # to make training a little bit easier for now
    dataSet = dataSet[lbls<5,:,:] # Remove all the points that do not belong to a class
    lbls = lbls[lbls<5]

if 0: # Throw out the 0 cluster
    dataSet = dataSet[lbls>0,:,:]
    lbls=lbls[lbls>0]

if 0: # Only classify 0 cluster vs not 0 cluster
    lbls[lbls>0] = 1
    
lbls = OneHotEncoder(sparse=False).fit_transform(lbls.reshape(-1,1))

lblCount = lbls.shape[1]

#%% Create a train and test set
X_train, X_test, y_train, y_test = train_test_split(dataSet,lbls, stratify=lbls)

#%% Define & train a jump+absolute LSTM
model = Sequential()
model.add(GRU(4, input_shape=(lag,4)))#, return_sequences=True))
#model.add(GRU(5))
model.add(Dense(lblCount, activation='softmax'))

optimiser = Adam(lr=0.00001)
model.compile(optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
hist=model.fit(X_train, y_train, batch_size=5, nb_epoch=500, validation_split=0.2, verbose=2).history

#%% Test the jump+absolute LSTM
yhat = model.predict_proba(X_test)
ap = average_precision_score(y_test,yhat)
yhatt = model.predict_classes(X_test)
ac = accuracy_score(np.argmax(y_test,axis=1),yhatt)
print('\nAP: ', ap,'   ac: ', ac)

#%% Plot the history
plt.figure()
plt.plot(hist['loss'],label='Train loss')
plt.plot(hist['val_loss'],label='Validation loss')
plt.legend()

#%% Statefull LSTM

#%% Create data set with 1 data points -> 1 label
lag = 100
ran = range(0,len(df[['gate jump']]))
lagSquare = np.concatenate([dfS[['gate jump']].shift(i) for i in ran],axis=1)
gateJumpSet=lagSquare[lag:,:lag]
lagSquare = np.concatenate([dfS[['yellow jump']].shift(i) for i in ran],axis=1)
yellowJumpSet=lagSquare[lag:,:lag]
#%%
#labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))
dataSet = np.dstack((gateJumpSet,yellowJumpSet))[:-1,:,:] #I don't know the label for the final sequence, so drop it
lbls = labels[lag+1:]
lbls[lbls==-1] = 5 #Setting this class to 5 so it can be one hot encoded and more easily be cut off
if 1: # to make training a little bit easier for now
    dataSet = dataSet[lbls<5,:,:] # Remove all the points that do not belong to a class
    lbls = lbls[lbls<5]

if 0: # Throw out the 0 cluster
    dataSet = dataSet[lbls>0,:,:]
    lbls=lbls[lbls>0]

if 0: # Only classify 0 cluster vs not 0 cluster
    lbls[lbls>0] = 1
    
lbls = OneHotEncoder(sparse=False).fit_transform(lbls.reshape(-1,1))

lblCount = lbls.shape[1]

#%% Create a train and test set
#X_train, X_test, y_train, y_test = train_test_split(dataSet,lbls, stratify=lbls)
X_train = dataSet[:1000,:,:]
X_valid = dataSet[1000:1100,:,:]
X_test = dataSet[1100:,:,:]
y_train = lbls[:1000,:]
y_valid = dataSet[1000:1100,:,:]
y_test = lbls[1100:,:]

#%% Define & train a stateful LSTM
batchSize = 10
nbEpochs = 300
loss=[]

model = Sequential()
model.add(GRU(10, batch_input_shape=(batchSize,lag,2), stateful=True))#, return_sequences=True))
#model.add(GRU(5, stateful=True))
model.add(Dense(lblCount, activation='softmax'))

optimiser = Adam(lr=0.00001)
model.compile(optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
for i in range(nbEpochs): # Number of epochs
    loss = np.append(loss,model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=2, shuffle=False).history['loss'])
    #For some reason the batchsize is ignored in the model.evaluate
    #val_acc, val_loss = model.evaluate(X_valid, y_valid, batch_size=batchSize)
    #print('Epoch ',i,'/',nbEpochs,': val_acc: ',val_acc,'   val_loss: ',val_loss)
    print('Epoch',i+1,'/',nbEpochs)
    model.reset_states()

#%% Test the stateful LSTM
yhat = model.predict_proba(X_test[:150,:,:], batch_size=batchSize)
ap = average_precision_score(y_test[:150,:],yhat)
yhatt = model.predict_classes(X_test[:150,:,:], batch_size=batchSize)
ac = accuracy_score(np.argmax(y_test[:150,:],axis=1),yhatt)
print('\nAP: ', ap,'   ac: ', ac)

#%% Plot the history
plt.figure()
plt.plot(loss,label='Train loss')
#plt.plot(hist['val_loss'],label='Validation loss')
plt.legend()

#%% LSTM using the jump cluster labels as input

#%%
lag = 25
labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))
#dataSet = np.dstack((gateSet,yellowSet,gateJumpSet,yellowJumpSet))[:-1,:,:] #I don't know the label for the final sequence, so drop it
lbls = labels[lag+1:]
lbls[lbls==-1] = 5 #Setting this class to 5 so it can be one hot encoded and more easily be cut off
labels[labels==-1] = 5

if 0: # to make training a little bit easier for now
    dataSet = dataSet[lbls<5,:] # Remove all the points that do not belong to a class
    lbls = lbls[lbls<5]
    
if 0: # Throw out the 0 cluster
    dataSet = dataSet[lbls>0,:,:]
    lbls=lbls[lbls>0]

if 0: # Only classify 0 cluster vs not 0 cluster
    lbls[lbls>0] = 1
    
lbls = OneHotEncoder(sparse=False).fit_transform(lbls.reshape(-1,1))
lblCount = lbls.shape[1]

ran = range(0,len(dfS[['gate jump']]))
lagSquare = np.concatenate([dfS[['gate jump']].shift(i) for i in ran],axis=1)
gateJumpSet=lagSquare[lag:,:lag]
lagSquare = np.concatenate([dfS[['yellow jump']].shift(i) for i in ran],axis=1)
yellowJumpSet=lagSquare[lag:,:lag]
#lagSquare = np.concatenate([dfS[['gate']].shift(i) for i in ran],axis=1)
#gateSet=lagSquare[lag:,:lag]
#lagSquare = np.concatenate([dfS[['yellow']].shift(i) for i in ran],axis=1)
#yellowSet=lagSquare[lag:,:lag]
labels = pd.DataFrame(labels)
lagSquare = np.concatenate([labels.shift(i) for i in ran],axis=1)
labelSet = lagSquare[lag:,:lag]
dataSet = np.dstack((labelSet,gateJumpSet,yellowJumpSet))[:-1,:,:] #I don't know the label for the final sequence, so drop it
#dataSet = np.reshape(np.array(labelSet),(1497,25,1))[:-1,:,:] #I don't know the label for the final sequence, so drop it

#%% Create a train and test set
X_train, X_test, y_train, y_test = train_test_split(dataSet,lbls, stratify=lbls)

#%% Define & train a clusterLbl LSTM
model = Sequential()
model.add(GRU(5, input_shape=(lag,3)))#, return_sequences=True))
#model.add(GRU(5))
model.add(Dense(lblCount, activation='softmax'))

optimiser = Adam(lr=0.00001)
model.compile(optimiser, loss='binary_crossentropy', metrics=['accuracy'])
hist=model.fit(X_train, y_train, batch_size=1, nb_epoch=300, validation_split=0.2, verbose=2).history

#%% Test the clusterLbl LSTM
yhat = model.predict_proba(X_test)
ap = average_precision_score(y_test,yhat)
yhatt = model.predict_classes(X_test)
ac = accuracy_score(np.argmax(y_test,axis=1),yhatt)
print('\nAP: ', ap,'   ac: ', ac)

#%% Plot the history
plt.figure()
plt.plot(hist['loss'],label='Train loss')
plt.plot(hist['val_loss'],label='Validation loss')
plt.legend()
