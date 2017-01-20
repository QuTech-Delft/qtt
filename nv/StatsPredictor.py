# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:18:07 2017

@author: Laurens
"""

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

class StatsPredictor():
    def __init__(self,lookback=1, jumpClusters=[], ignoreZero=True, minimumData=10):
        self.lookback = lookback
        self.ignoreZero=ignoreZero
        self.minimumData = minimumData
        if len(jumpClusters)>0:
            self.fit(jumpClusters)
    
    def _fitClustered(self, jumpClusters):
        if self.ignoreZero:
            jumpClusters = jumpClusters[jumpClusters>0]
        self.labels = jumpClusters[self.lookback+1:]
        ran = range(0,len(jumpClusters))
        self.laggedData = np.zeros((len(jumpClusters)-self.lookback,self.lookback))
        d = pd.DataFrame(jumpClusters)
        self.laggedData[:,:] = np.concatenate([d.shift(i) for i in ran],axis=1)[self.lookback:,:self.lookback]
        self.lastSeq = self.laggedData[-1,:]
        self.laggedData = self.laggedData[:-1,:]

    def predictNext(self):
        if not hasattr(self, 'laggedData'):
            print("This instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")
            return np.array([])
        index = self.__getIndex(self.laggedData,self.lastSeq)
        i=0
        while np.sum(index)<self.minimumData:
            i+=1
            index = self.__getIndex(self.laggedData[:,:-i],self.lastSeq[:-i])
        if i>0:
            print('Prediction made based on lookback of ', self.lookback-i)
        prediction=np.bincount(self.labels[index])/np.sum(index)
        while len(prediction)<len(self.labels):
            prediction = np.append(prediction,0)
        #print(len(prediction))
        return prediction
        
    def foundNextCluster(self, newClust):
        if self.ignoreZero and newClust==0:
            print('ignoring 0')
            return self.predictNext()
        self.laggedData = np.append(self.laggedData,np.reshape(self.lastSeq,(1,len(self.lastSeq))),axis=0)
        self.labels = np.append(self.labels,newClust)
        self.lastSeq = np.append(self.lastSeq,newClust)[1:]
        return self.predictNext()
    
    def fit(self, absVals):
        #Convert absolute values to jumps
        self._fitClustered(absVals)
        return
    
    def foundNextValue(self):
        return
    
    def __getIndex(self,lagDat,lastSeq):
        index=np.array([],dtype='bool')
        for i in range(lagDat.shape[0]):
            index = np.append(index,np.array_equal(lagDat[i,:],lastSeq))
        return index
        
        
        