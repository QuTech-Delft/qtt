# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:07:12 2017

@author: Laurens
"""

import sys,os
import numpy as np
from matplotlib import pyplot as plt
import copy
import pandas as pd
from sklearn.metrics import mean_squared_error
from nvtools.nvtools import extract_data
import qcodes
from statsmodels.tsa.ar_model import AR

#%%
print('Generating Data')
data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T

df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump'])
#plt.figure(300); plt.clf()
#df.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), linewidths=0)

#%% Visually confirm autocorrelation
laserFreq = df['yellow']
gateV = df['gate']
fig = plt.figure(figsize=(13,6))
plt1 = plt.subplot(121)
plt1.set_title('Yellow frequency lagplot')
pd.tools.plotting.lag_plot(laserFreq)
plt2 = plt.subplot(122)
plt2.set_title('Gate voltage lagplot')
pd.tools.plotting.lag_plot(gateV)
#How does this plot look with only the non zero jumps?

fig = plt.figure()
pd.tools.plotting.autocorrelation_plot(laserFreq, label='Yellow frequency')
pd.tools.plotting.autocorrelation_plot(gateV, label='Gate voltage')
plt.legend()

laserFreqVals = pd.DataFrame(laserFreq.values)
laserFraqDataframe = pd.concat([laserFreqVals.shift(1), laserFreqVals], axis=1)
laserFraqDataframe.columns = ['t-1', 't+1']
result = laserFraqDataframe.corr()
print(result)

gateVvals = pd.DataFrame(gateV.values)
gateVdataframe = pd.concat([gateVvals.shift(1), gateVvals], axis=1)
gateVdataframe.columns = ['t-1', 't+1']
result = gateVdataframe.corr()
print(result)

#%% Persistance model (baseline)
testSetSize = 500
#Laser frequency
# split lagged datasets into train and test sets
X = laserFraqDataframe.values
lFtrain, lFtest = X[1:len(X)-testSetSize], X[len(X)-testSetSize:]
train_X, train_y = lFtrain[:,0], lFtrain[:,1]
test_X, test_y = lFtest[:,0], lFtest[:,1]
 
# persistence model
def model_persistence(x):
	return x
 
# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE Yellow frequency: %.3f' % test_score)
# plot predictions vs expected
fig = plt.figure()
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()

# split lagged datasets into train and test sets
X = gateVdataframe.values
gVtrain, gVtest = X[1:len(X)-testSetSize], X[len(X)-testSetSize:]
train_X, train_y = gVtrain[:,0], gVtrain[:,1]
test_X, test_y = gVtest[:,0], gVtest[:,1]

# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE gate voltage: %.3f' % test_score)
# plot predictions vs expected
fig = plt.figure()
plt.plot(test_y)
plt.plot(predictions, color='red')
plt.show()

#%% Train an autoregression model
testSetSize = 10
# split dataset yellow frequency
X = laserFreq.values
train, test = X[1:len(X)-testSetSize], X[len(X)-testSetSize:]
# train autoregression model
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
fig = plt.figure()
plt.plot(test, label='testData')
plt.plot(predictions, color='red', label = 'predictions')
plt.legend()

X = gateV.values
train, test = X[1:len(X)-testSetSize], X[len(X)-testSetSize:]
# train autoregression model
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
fig = plt.figure()
plt.plot(test, label='testData')
plt.plot(predictions, color='red', label = 'predictions')
plt.legend()

#The clear conclusion is that autoregression is not a great idea