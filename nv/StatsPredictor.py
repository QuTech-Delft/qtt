# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:18:07 2017

@author: Laurens


===== TODO ======
Is cluster 0 always the centre cluster??
No, it appears to always be the cluster with the most points, not neccesarily the middle one...
So skipping it for prediction does not make sense in that case, and thus the prediction *should* be worse
For now test each clustering with and without skipping 0, but something to keep an eye on

How does a statspredictor fitted without 0 centre handle 0 centre inputs? <- getLabels checks the density


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
import matplotlib.cm as cm

class StatsPredictor():
    '''
    The statspredictor predicts the next cluster based on the previous occurences of the last #lookback sequence
    int lookback - The number of previous values it uses for the prediction
    [time, gate, yellow] data - The data, used to fit the StatsPredictor
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
    Fits the predictor using the absolute values
    [time, gate, yellow] X - the absolute values used to fit the predictor
    '''
    def fit(self, X):
        #Convert absolute values to jumps
        self.jumpGate, self.jumpYellow = self.toJumps(X)
        #Form the kdTree used to classify new points
        self.kdTree = KDTree(np.vstack((self.jumpGate,self.jumpYellow)).T)
        self.kdTree0 = KDTree(self.filterZero(np.vstack((self.jumpGate,self.jumpYellow)).T))
        #Find the clustering to use for this dataset
        labels = self.findClustering(np.hstack(self.clustScale(self.jumpGate,self.jumpYellow,True)))
        self.fitClustered(labels)
        return
    
    '''
    Fits the predictor using the clusters
    [gate jump, yellow jump] jumpClusters - the sequence of cluster labels
    '''
    def fitClustered(self, jumpClusters):
        jumpClusters=self.moveNonCluster(jumpClusters)
        if self.ignoreZero:
            jumpClusters = jumpClusters[jumpClusters>0]
        else:
            jumpClusters = jumpClusters.flatten()
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
            if self.verbose:
                print('Prediction made based on lookback of ', self.lookback-i)
        prediction=np.bincount(self.labels[index])/np.sum(index)
        while len(prediction)<len(self.labels):
            prediction = np.append(prediction,0)
        if self.ignoreZero: #If we ignore 0, always check 0 first, i.e. make it the first prediction
            prediction = np.append(10,prediction)
        
        return prediction
        
    def foundNextCluster(self, newClust):
        if self.ignoreZero and newClust==0:
            if self.verbose:
                print('ignoring 0')
            return self.predictNext()
        self.laggedData = np.append(self.laggedData,np.reshape(self.lastSeq,(1,len(self.lastSeq))),axis=0)
        self.labels = np.append(self.labels,newClust)
        self.lastSeq = np.append(self.lastSeq,newClust)[1:]
        return self.predictNext()

    #Does not currently work well for very small X values
    def clustScale(self,jmpGate,jmpYellow,init=False):
        if init or not hasattr(self,'scG') or not hasattr(self,'scY'):
            self.scG = StandardScaler()
            self.scY = StandardScaler()
            return [self.scG.fit_transform(jmpGate.reshape(-1,1)),self.scY.fit_transform(jmpYellow.reshape(-1,1))]
        return [self.scG.transform(jmpGate.reshape(-1,1)),self.scY.transform(jmpYellow).reshape(-1,1)]
    
    #Ignore 0 if it is included in the clustering
    #If 0 was already removed before clustering, don't ignore 0, since it's a normal cluster in that case
    def evaluate(self,jumps,clust='auto',ignoreZero=True):
        if clust is 'auto':
            clust = self.clusterer
        trainCut = int(np.floor(0.8*len(jumps)))
        #trainJumps = jumps[:trainCut,:]
        labels = clust.labels_
        trainLabels = labels[:trainCut]
        #testLabels = labels[trainCut:]
        if len(np.unique(trainLabels))<2: #It can happen that there is only cluster 0 in the train set
              #In that case the classifier won't be great anyway, so skip it
            return -1
        sp = StatsPredictor(ignoreZero=ignoreZero)
        sp.fitClustered(self.moveNonCluster(trainLabels))
        
        hitTime = self.clusterDistanceMetric(jumps,trainCut,clust,sp)            
        return np.mean(hitTime)
    
    '''
    For this metric the score for a points is:
        dist=distance(newPoint,predictedClusterCenter)
    Final score is the average of all these distances
    
    prediction = list(np.argsort(prediction)[::-1])
    dist=0
    for j in range(prediction.index(testLabels[i])): #For each cluster to check
        dist += np.linalg.norm(clustCentres[np.argmax(prediction[j]),:]-testJumps[i,:])

    '''
    def clusterDistanceMetric(self,jumps,trainCut,clust,statsPred):
        labels = clust.labels_
        labels = self.moveNonCluster(labels)
        testLabels = labels[trainCut:]
        #Get both gate and yellow jump values on a scale from 0 to 15
        oldGateRange = np.max(jumps[:,0])-np.min(jumps[:,0])
        oldYellowRange = np.max(jumps[:,1])-np.min(jumps[:,1])
        sclJumps = np.zeros(jumps.shape)
        sclJumps[:,0] = ((jumps[:,0]-np.min(jumps[:,0]))*15)/oldGateRange
        sclJumps[:,1] = ((jumps[:,1]-np.min(jumps[:,1]))*15)/oldYellowRange
        testJumps = sclJumps[trainCut:,:]        
        
        #find cluster centres
        clustCentres = statsPred.getClusterCentres(jumps,clust)
        #Scale the cluster centres:
        clustCentres[:,0] =  ((clustCentres[:,0]-np.min(jumps[:,0]))*15)/oldGateRange
        clustCentres[:,1] =  ((clustCentres[:,1]-np.min(jumps[:,1]))*15)/oldYellowRange
        
        hitTime = np.zeros((len(testLabels),1))
        for i in range(len(testLabels)):
            #For each jump first check the distance from the centre of the predicted cluster
            prediction = statsPred.predictNext()
            dist = np.linalg.norm(clustCentres[np.argmax(prediction),:]-testJumps[i,:])
            hitTime[i] = dist
            statsPred.foundNextCluster(testLabels[i])
            #Add this when fixed:
            #statsPred.foundNextValue(testJumps[i,:])
        
        return hitTime
    
    def getClusterCentres(self,jumps='auto',clust='auto'):
        if jumps is 'auto':
            np.vstack((self.jumpGate,self.jumpYellow)).T
        if clust is 'auto':
            clust=self.clusterer
        allLabels = np.unique(clust.labels_)
        clustCentres = np.zeros((len(allLabels),2))
        labels = clust.labels_
        for lab in range(len(allLabels)):
            c = jumps[(labels==allLabels[lab]),:]
            clustCentres[lab,0] = np.min(c[0]) + (np.max(c[0]) - np.min(c[0]))/2
            clustCentres[lab,1] = np.min(c[1]) + (np.max(c[1]) - np.min(c[1]))/2
        return clustCentres
    
    def findClustering(self,jumps):
        spectBands=[3,5,7,9]
        gamma=np.concatenate((np.arange(0.1,1,0.1),np.arange(1,3,0.2),np.arange(3,10,1)))
        epsilon=np.arange(0.1,0.7,0.1)
        minSamples=range(10,100,10)
        #Try all DBSCAN
        jumps0 = self.filterZero(jumps)
        bestScore = 100
        bestClust = 0
        clustString=''
        self.zeroDens=False
        for e in epsilon:
            for samps in minSamples:
                #With 0 cluster
                clust = DBSCAN(eps=e,min_samples=samps).fit(jumps)
                if len(np.unique(clust.labels_))>1:
                    if self.verbose:
                        self.plotClustering(clust,jumps)
                    score = self.evaluate(jumps,clust,True)
                    if score > 0:
                        if score<bestScore:
                            bestClust = clust
                            bestScore=score
                            self.zeroDens=False
                            clustString='DBSCAN   e: '+str(e)+' s: '+str(samps)+' Score: '+str(score)+' Cluster count: '+str(len(np.unique(clust.labels_)))
                        if self.verbose:
                            print('DBSCAN   e: ',e,' s: ',samps,' Score: ',score,' Cluster count: ',len(np.unique(clust.labels_)))    
                    elif self.verbose:
                        print('---Only 0 in trainset--- DBSCAN   e: ',e,' s: ',samps,' Score: ',score,' Cluster count: ',len(np.unique(clust.labels_)))    
                elif self.verbose:
                    print('---No clustering--- DBSCAN   e: ',e,' s: ',samps,' Score: ',score,' Cluster count: ',len(np.unique(clust.labels_)))    
                #With 0 cluster, don't ignore it in predictions
                clust = DBSCAN(eps=e,min_samples=samps).fit(jumps)
                if len(np.unique(clust.labels_))>1:
                    if self.verbose:
                        self.plotClustering(clust,jumps)
                    score = self.evaluate(jumps,clust,False)
                    if score > 0:
                        if score<bestScore:
                            bestClust = clust
                            bestScore=score
                            self.zeroDens=False
                            clustString='DBSCAN use0  e: '+str(e)+' s: '+str(samps)+' Score: '+str(score)+' Cluster count: '+str(len(np.unique(clust.labels_)))
                        if self.verbose:
                            print('DBSCAN use0  e: ',e,' s: ',samps,' Score: ',score,' Cluster count: ',len(np.unique(clust.labels_)))    
                    elif self.verbose:
                        print('---Only 0 in trainset--- DBSCAN use0  e: ',e,' s: ',samps,' Score: ',score,' Cluster count: ',len(np.unique(clust.labels_)))    
                elif self.verbose:
                    print('---No clustering--- DBSCAN use0  e: ',e,' s: ',samps,' Score: ',score,' Cluster count: ',len(np.unique(clust.labels_)))    
                #Without 0 cluster
                clust = DBSCAN(eps=e,min_samples=samps).fit(jumps0)
                if len(np.unique(clust.labels_))>1:
                    if self.verbose:
                        self.plotClustering(clust,jumps0)
                    score = self.evaluate(jumps0,clust,False)
                    if score > 0:
                        if score<bestScore:
                            bestClust = clust
                            bestScore=score
                            self.zeroDens=True
                            clustString='DBSCAN_ignore0   e: '+str(e)+' s: '+str(samps)+' Score: '+str(score)+' Cluster count: '+str(len(np.unique(clust.labels_)))
                        if self.verbose:
                            print('DBSCAN_ignore0   e: ', e, ' s: ', samps, ' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                    elif self.verbose:
                        print('---Only 0 in trainset--- DBSCAN_ignore0   e: ',e,' s: ',samps,' Score: ',score,' Cluster count: ',len(np.unique(clust.labels_)))    
                elif self.verbose:
                    print('---No clustering--- DBSCAN_ignore0   e: ',e,' s: ',samps,' Score: ',score,' Cluster count: ',len(np.unique(clust.labels_)))    
        #Check all spectral clustering
        for spect in spectBands:
            for gam in gamma:
                #With 0 cluster
                clust = SpectralClustering(spect,gamma=gam).fit(jumps)
                if len(np.unique(clust.labels_))>1:
                    if self.verbose:
                        self.plotClustering(clust,jumps)
                    score = self.evaluate(jumps,clust,True)
                    if score > 0:
                        if score<bestScore:
                            bestClust = clust
                            bestScore=score
                            self.zeroDens=False
                            clustString='Spectral   spect: '+ str(spect)+' gamma: '+gam+' Score: '+ str(score)+' Cluster count: '+str(len(np.unique(clust.labels_)))
                        if self.verbose:
                            print('Spectral   spect: ', spect,' gamma: ',gam,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                    elif self.verbose:
                        print('---Only 0 in trainset--- Spectral   spect: ', spect,' gamma: ',gam,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                elif self.verbose:
                    print('---No clustering--- Spectral   spect: ', spect,' gamma: ',gam,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                #With 0 cluster, don't ignore it in predictions
                clust = SpectralClustering(spect,gamma=gam).fit(jumps)
                if len(np.unique(clust.labels_))>1:
                    if self.verbose:
                        self.plotClustering(clust,jumps)
                    score = self.evaluate(jumps,clust,True)
                    if score > 0:
                        if score<bestScore:
                            bestClust = clust
                            bestScore=score
                            self.zeroDens=False
                            clustString='Spectral use0  spect: '+ str(spect)+' gamma: '+gam+' Score: '+ str(score)+' Cluster count: '+str(len(np.unique(clust.labels_)))
                        if self.verbose:
                            print('Spectral use0   spect: ', spect,' gamma: ',gam,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                    elif self.verbose:
                        print('---Only 0 in trainset--- Spectral use0  spect: ', spect,' gamma: ',gam,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                elif self.verbose:
                    print('---No clustering--- Spectral use0  spect: ', spect,' gamma: ',gam,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                #Without 0 cluster
                clust = SpectralClustering(spect,gamma=gam).fit(jumps0)
                if len(np.unique(clust.labels_))>1:
                    if self.verbose:
                        self.plotClustering(clust,jumps0)
                    score = self.evaluate(jumps0,clust,False)
                    if score > 0:
                        if score<bestScore:
                            bestClust = clust
                            bestScore=score
                            self.zeroDens=True
                            clustString='Spectral_ignore0   spect: '+ str(spect)+' Score: '+ str(score)+' Cluster count: '+str(len(np.unique(clust.labels_)))
                        if self.verbose:
                            print('Spectral_ignore0   spect: ', spect,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                    elif self.verbose:
                        print('---Only 0 in trainset--- Spectral_ignore0   spect: ', spect,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                elif self.verbose:
                    print('---No clustering--- Spectral_ignore0   spect: ', spect,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
        self.clusterer = bestClust
        if self.zeroDens:
            self.clusterer = self.clusterer.fit(jumps0)
        else:
            self.clusterer = self.clusterer.fit(jumps)
        print('======= Best cluster performance: ', bestScore)
        print(clustString)
        if self.verbose:
            self.plotClustering(figNum=201)
        return self.clusterer.labels_

    '''
    Assigns labels to the given jumps based on the label of the nearest neighbour
    
    [gate jump, yellow jump] jumps - The jumps to be labelled
    [] lbls - if 'auto' use the labels as assigned by the clustering of this instance
            otherwise use provided labels (provided in order)
    KdTree kdt - the kdTree to use
    '''
    def getLabel(self,jumps,lbls='auto'):
        kdt = self.kdTree
        if lbls is 'auto':
            lbls=self.labels
        if len(lbls) < len(self.jumpGate): #when we have been given fewer labels than the total set, it must be because the 0 cluster was removed
            kdt = self.kdTree0
        lbls = self.moveNonCluster(lbls)
        labels = lbls[kdt.query(jumps,1,False)]
        if self.zeroDens:#Handle zero
            s = self.densityKern.score_samples(jumps)
            labels[s>=self.densVal] = 0
        return labels
    
    def toJumps(self,X):
        time = X[:,0]
        yellow = X[:,1]
        gate = X[:,2]
        dt = np.median(np.diff(time))
        
        jumpSelect = (np.diff(time)>3*dt) & (np.diff(time)<30*60) # take three times the median as a qualifier for a jump
        jumpGate = gate[np.append(False, jumpSelect)] - gate[np.append(jumpSelect, False)]
        jumpYellow = yellow[np.append(False, jumpSelect)] - yellow[np.append(jumpSelect, False)]
        return [jumpGate,jumpYellow]
    
    def foundNextValue(self,X):
        jumps = self.toJumps(X)
        self.jumpGate = np.append(self.jumpGate,jumps[:,0])
        self.jumpYellow = np.append(self.jumpYellow,jumps[:,1])
        newLab = self.getLabel(jumps)
        return self.foundNextCluster(newLab)
    
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
        self.densVal=np.sort(s)[::-1][int(len(jumps)*cutoff)]
        return np.vstack((jumps[s<self.densVal,0],jumps[s<self.densVal,1])).T
    
    def __getIndex(self,lagDat,lastSeq):
        index=np.array([],dtype='bool')
        for i in range(lagDat.shape[0]):
            index = np.append(index,np.array_equal(lagDat[i,:],lastSeq))
        return index
    
    #Changes the non-clustered points to the last index rather than -1
    def moveNonCluster(self,labels):
        uniq = np.unique(labels)
        if uniq[0] == -1:
            labels[labels==-1] = np.max(uniq)+1
        return labels
    
    def plotClustering(self,clust='auto',jumps='auto',figNum=101):
        if clust is 'auto':
            clust=self.clusterer
        if jumps is 'auto':
            jumps=np.vstack((self.jumpGate,self.jumpYellow)).T
        plt.figure(figNum)
        plt.scatter(jumps[:,0],jumps[:,1], c=clust.labels_, cmap=cm.jet, linewidths=0)
        plt.show()
    
    '''
    After x new values the predictor should probably be fit again to improve performance
    '''
    def _refit():
       return
        