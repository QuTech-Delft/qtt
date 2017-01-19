# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:50:54 2017

@author: Laurens
"""

''' WIP
TODO:
    Add a part that stores all the parameters in a CSV file
    Add a PCA part, perhaps that will improve learning (as the clusters are diagonal)
    Add more regularisation (dropout?)
'''

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
from sklearn.decomposition import PCA

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
LSTMtype = 3
doPCA = True

batchSize = 1
nbEpochs = 300
learningRate = 0.00001

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

if doPCA:
    pca = PCA(n_components=2)
    selectedData = pd.DataFrame(pca.fit_transform(selectedData))

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

if not keepNoClass: # to make training a little bit easier for now
    dataSet = dataSet[lbls<5,:,:] # Remove all the points that do not belong to a class
    lbls = lbls[lbls<5]

if zeroOrNotZero: # Only classify 0 cluster vs not 0 cluster
    if not keepZeroCluster:
        print('You should keep the zero cluster, otherwise this will not work')
        print('I am keeping the zero cluster for you!!')
        keepZeroCluster=True
        
    lbls[lbls>0] = 1

if not keepZeroCluster: # Throw out the 0 cluster
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

if LSTMtype==0:    # Basic LSTM
    model = Sequential()
    model.add(GRU(10, input_shape=(lag,X_train.shape[2])))
    model.add(Dense(lblCount, activation='softmax'))
    
    optimiser = Adam(lr=learningRate)
    model.compile(optimiser, loss='binary_crossentropy', metrics=['accuracy'])
    hist=model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=nbEpochs, validation_split=0.2, verbose=2).history
elif LSTMtype==1:    # Double LSTM layer
    model = Sequential()
    model.add(GRU(10, input_shape=(lag,X_train.shape[2]), return_sequences=True))
    model.add(GRU(10))
    model.add(Dense(lblCount, activation='softmax'))
    
    optimiser = Adam(lr=learningRate)
    model.compile(optimiser, loss='binary_crossentropy', metrics=['accuracy'])
    hist=model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=nbEpochs, validation_split=0.2, verbose=2).history
elif LSTMtype==2:    # Stateful LSTM
    print('nbEpochs is set to 1 so it will work with the stateful LSTM')
    npEpochs=1
    loss=[]
    
    model = Sequential()
    model.add(GRU(10, batch_input_shape=(batchSize,lag,X_train.shape[2]), stateful=True))#, return_sequences=True))
    #model.add(GRU(5, stateful=True))
    model.add(Dense(lblCount, activation='softmax'))
    
    optimiser = Adam(lr=learningRate)
    model.compile(optimiser, loss='binary_crossentropy', metrics=['accuracy'])
    for i in range(nbEpochs): # Number of epochs
        loss = np.append(loss,model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=npEpochs, verbose=2, shuffle=False).history['loss'])
        #For some reason the batchsize is ignored in the model.evaluate
        #val_acc, val_loss = model.evaluate(X_valid, y_valid, batch_size=batchSize)
        #print('Epoch ',i,'/',nbEpochs,': val_acc: ',val_acc,'   val_loss: ',val_loss)
        print('Epoch',i+1,'/',nbEpochs)
        model.reset_states()
elif LSTMtype==3: #Dropout
    model = Sequential()
    model.add(Dropout(0.3, input_shape=(lag,X_train.shape[2])))
    model.add(GRU(10))
    model.add(Dense(lblCount, activation='softmax'))
    
    optimiser = Adam(lr=learningRate)
    model.compile(optimiser, loss='binary_crossentropy', metrics=['accuracy'])
    hist=model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=nbEpochs, validation_split=0.2, verbose=2).history

#%% Test the LSTMs
if LSTMtype!=2:
    yhat = model.predict_proba(X_test)
    ap = average_precision_score(y_test,yhat)
    yhatt = model.predict_classes(X_test)
    ac = accuracy_score(np.argmax(y_test,axis=1),yhatt)
    print('\nAP: ', ap,'   ac: ', ac)
else: # Test the stateful LSTM
    sel = int(X_test.shape[0]/batchSize)*batchSize # The amount of data should be divisible by the batch size
    yhat = model.predict_proba(X_test[:sel,:,:], batch_size=batchSize)
    ap = average_precision_score(y_test[:sel,:],yhat)
    yhatt = model.predict_classes(X_test[:sel,:,:], batch_size=batchSize)
    ac = accuracy_score(np.argmax(y_test[:sel,:],axis=1),yhatt)
    print('\nAP: ', ap,'   ac: ', ac)
        
#%% Plot the history (usefull for checking whether the learning rate is any good) 
if LSTMtype!=2:
    plt.figure()
    plt.plot(hist['loss'],label='Train loss')
    plt.plot(hist['val_loss'],label='Validation loss')
    plt.legend()
else: # Since the stateful LSTM doesn't have validation
    plt.figure()
    plt.plot(loss,label='Train loss')
    plt.legend()