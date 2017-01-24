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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.neighbors import KernelDensity,KDTree

class StatsPredictor():
    '''
    The statspredictor predicts the next cluster based on the previous occurences of the last #lookback sequence
    int lookback - The number of previous values it uses for the prediction
    [time, yellow, gate] data - The data, used to fit the StatsPredictor
    Boolean ignoreZero - If true the 0 jumps are removed thus causing the lookback to skip 0
    int minimumData - The minimum amount of data needed to make a prediction, if this value
        cannot be reached, the lookback is lowered for that prediction
    Boolean verbose - Functions might print more if True
    '''
    def __init__(self,lookback=1, data=[], ignoreZero=True, minimumData=10,verbose=False):
        self.lookback = lookback
        self.ignoreZero=ignoreZero
        self.minimumData = minimumData
        self.verbose = verbose
        self.online = False #Online not currently implemented
        if len(data)>0:
            self.fit(data)
    
    '''
    Fits the predictor using the clusters
    [] jumpClusters - the sequence of cluster labels
    '''
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

    '''
    Returns probabilities per cluster
    '''
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

    #Does not currently work well for very small X values
    def _clustScale(self,jmpGate,jmpYellow,init=False):
        if init or not hasattr(self,'scG') or not hasattr(self,'scY'):
            self.scG = StandardScaler()
            self.scY = StandardScaler()
            return [self.scG.fit_transform(jmpGate),self.scY.fit_transform(jmpYellow)]
        return [self.scG.transform(jmpGate),self.scY.transform(jmpYellow)]
    
    #Cutoff as percentage of data in the 0 cluster
    def filterZero(self,jumps,cutoff=0.65):
        self.densityKern = KernelDensity().fit(jumps)
        s = self.densityKern.score_samples(jumps)
        if self.verbose:
            plt.figure()
            plt.subplot(121)
            plt.title('Density per Gate Voltage')
            plt.scatter(jumps[:,0],s)
            plt.subplot(122)
            plt.title('Density per Yellow Frequency')
            plt.scatter(jumps[:,1],s)
        self.densVal=np.sort(s)[::-1][np.floor(len(jumps)*cutoff)]
        return [jumps[0][s<self.densVal],jumps[1][s<self.densVal]]
    
    def evaluate(self,jumps,clust,zeroDens='auto'):
        trainCut = np.floor(0.8*len(jumps))
        trainJumps = jumps[:trainCut,:]
        testJumps = jumps[trainCut:,:]
        labels = self.getLabel(jumps,clust.labels_)
        trainLabels = labels[:trainCut,:]
        testLabels = labels[trainCut:,:]
        sp = StatsPredictor()
        sp._fitClustered(trainLabels)
        hitTime = np.zeros((len(testJumps),1))
        
        #Get both gate and yellow jump values on a scale from 0 to 15
        oldGateRange = np.max(jumps[0,:])-np.min(jumps[0,:])
        oldYellowRange = np.max(jumps[1,:])-np.min(jumps[1,:])
        sclJumps = jumps
        sclJumps[0,:] = ((jumps[0,:]-np.min(jumps[0,:]))*15)/oldGateRange
        sclJumps[1,:] = ((jumps[1,:]-np.min(jumps[1,:]))*15)/oldYellowRange

        #find cluster centres
        clustCentres = self.getClusterCentres()
        hitTime = np.zeros((len(testLabels),1))
        for i in range(len(testLabels)):
            #For each jump first check the distance from the centre of the predicted cluster
            dist = np.linalg.norm(clustCentres[np.argmax(sp.predictNext()),:]-testJumps[i,:])
            hitTime[i] = dist
            #Maybe some way of taking the correctness of the clustering into account?
        return np.mean(hitTime)
    
    def getClusterCentres(self,jumps,clust='auto'):
        if clust=='auto':
            clust=self.clusterer
        allLabels = np.unique(clust.labels_)
        clustCentres = np.zeros((len(allLabels),2))
        for l in range(len(allLabels)):
            c = jumps[labels==allLabels[l]]
            clustCentres[l,0] = np.min(c[0]) + (np.max(c[0]) - np.min(c[0]))/2
            clustCentres[l,1] = np.min(c[1]) + (np.max(c[1]) - np.min(c[1]))/2
        return clustCentres
    
    def findClustering(self,jumps):
        spectBands=[3,5,7]
        epsilon=range(0.1,0.7,0.1)
        minSamples=range(10,100,10)
        #Try all DBSCAN
        jumps0 = self.filterZero(jumps)
        bestScore = 100
        bestClust = 0
        self.zeroDens=False
        for e in epsilon:
            for samps in minSamples:
                #With 0 cluster
                clust = DBSCAN(eps=e,min_samples=samps).fit(jumps)
                score = self.evaluate(jumps,False)
                if score<bestScore:
                    bestClust = clust
                    bestScore=score
                    self.zeroDens=False
                #Without 0 cluster
                clust = DBSCAN(eps=e,min_samples=samps).fit(jumps0)
                score = self.evaluate(jumps,True)
                if score<bestScore:
                    bestClust = clust
                    bestScore=score
                    self.zeroDens=True
        #Try all SpectralClustering (CHECK FOR VARYING GAMMA???)
        for spect in spectBands:
            #With 0 cluster
            clust = SpectralClustering(spectBands,gamma=0.2).fit(jumps)
            score = self.evaluate(jumps,False)
            if score<bestScore:
                bestClust = clust
                bestScore=score
                self.zeroDens=False
            #Without 0 cluster
            clust = SpectralClustering(spectBands,gamma=0.2).fit(jumps0)
            score = self.evaluate(jumps,True)
            if score<bestScore:
                bestClust = clust
                bestScore=score
                self.zeroDens=True
        self.clusterer = bestClust
        if self.zeroDens:
            self.clusterer = self.clusterer.fit(jumps0)
        else:
            self.clusterer = self.clusterer.fit(jumps)
        print('Best cluster performance: ', score)
        return self.getLabel(jumps)

    '''
    Assigns labels to the given jumps based on the label of the nearest neighbour
    
    [gate jump, yellow jump] jumps - The jumps to be labelled
    lbls - if 'auto' use the labels as assigned by the clustering of this instance
            otherwise use provided labels (provided in order)
    '''
    def getLabel(self,jumps,lbls='auto'):
        if lbls == 'auto':
            lbls=self.labels
        labels = lbls[self.kdTree.query(jumps,1,False)[1,:]]
        if self.zeroDens:#Handle zero
            s = self.densityKern.score_samples(jumps)
            labels[s>=self.densVal] = 0
        return labels
    
    def _toJumps(self,X):
        time = X[0]
        yellow = X[1]
        gate = X[2]
        dt = np.median(np.diff(time))
        
        jumpSelect = (np.diff(time)>3*dt) & (np.diff(time)<30*60) # take three times the median as a qualifier for a jump
        jumpGate = gate[np.append(False, jumpSelect)] - gate[np.append(jumpSelect, False)]
        jumpYellow = yellow[np.append(False, jumpSelect)] - yellow[np.append(jumpSelect, False)]
        return np.hstack((jumpGate,jumpYellow)).T
    
    def fit(self, X):
        #Convert absolute values to jumps
        self.jumpGate, self.jumpYellow = self._toJumps(X)
        #Form the kdTree used to classify new points
        self.kdTree = KDTree(np.hstack((self.jumpGate,self.jumpYellow)).T)        
        #Find the clustering to use for this dataset
        labels = self.findClustering(self._clustScale(self.jumpGate,self.jumpYellow,True))
        self._fitClustered(labels)
        return
    
    def foundNextValue(self,X):
        jumps = self._toJumps(X)
        self.jumpGate = np.append(self.jumpGate,jumps[:,0])
        self.jumpYellow = np.append(self.jumpYellow,jumps[:,1])
        newLab = self.getLabel(jumps)
        return self.foundNextCluster(newLab)
    
    def __getIndex(self,lagDat,lastSeq):
        index=np.array([],dtype='bool')
        for i in range(lagDat.shape[0]):
            index = np.append(index,np.array_equal(lagDat[i,:],lastSeq))
        return index
    
    '''
    After x new values the predictor should probably be fit again to improve performance
    '''
    def _refit():
       return
        