# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:18:07 2017

@author: Laurens

TODO:
There should probably be a function that returns the areas each cluster covers
Perhaps instead of scaling before clustering, the data should be scaled by the step size?

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
from sklearn.decomposition import PCA
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
    def __init__(self,lookback=1, data=[], ignoreZero=True, minimumData=10,verbose=False,attractmV=15,attractFreq=40e-3,removedZero=False):
        self.lookback = lookback
        self.ignoreZero=ignoreZero
        self.minimumData = minimumData
        self.verbose = verbose
        self.attractmV = attractmV
        self.attractFreq = attractFreq
        self.removedZero = removedZero
        self.prevX = 'init'
        self.online = False #Online not currently implemented
        if len(data)>0:
            self.fit(data)
    
    '''
    Fits the predictor using the absolute values
    [time, gate, yellow] X - the absolute values used to fit the predictor
    '''
    def fit(self, X, quick=False):
        #Convert absolute values to jumps
        self.X = X
        self.jumpGate, self.jumpYellow = self.toJumps(X)
        self.scaledJumpGate, self.scaledJumpYellow = self.clustScale(self.jumpGate,self.jumpYellow)
        #Form the kdTree used to classify new points
        self.kdTree = KDTree(np.vstack((self.scaledJumpGate,self.scaledJumpYellow)).T)
        self.kdTree0 = KDTree(self.filterZero(np.vstack((self.scaledJumpGate,self.scaledJumpYellow)).T))
        #Find the clustering to use for this dataset
        labels = self.findClustering(np.vstack((self.scaledJumpGate,self.scaledJumpYellow)).T,quick)
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
        while len(prediction)<len(np.unique(self.labels)):
            prediction = np.append(prediction,0)
        if self.ignoreZero: #If we ignore 0, always check 0 first, i.e. make it the first prediction
            prediction[0]=10
        if self.removedZero: #If the cluster is removed, it should still show up in the predictions
            prediction = np.append(10,prediction)
        return prediction
    
    '''
    Called when a new point is found, it updates the internal state so the predictions
    will now also include the new point.
    This is usually only called by self.foundNextValue()
    
    int newClust - The label of the cluster the new point is in
    '''
    def foundNextCluster(self, newClust):
        if self.ignoreZero and newClust==0:
            if self.verbose:
                print('ignoring 0')
            return self.predictNext()
        self.laggedData = np.append(self.laggedData,np.reshape(self.lastSeq,(1,len(self.lastSeq))),axis=0)
        self.labels = np.append(self.labels,newClust)
        self.lastSeq = np.append(self.lastSeq,newClust)[1:]
        return self.predictNext()

    '''
    Scales the jumps to so that they are on the same scale, this is important for the clustering,
    as the difference in scale between the gate voltage and the yellow frequency is quite large
    
    [float] jmpGate - all the values for jumps in the gate voltage
    [float] jmpYellow - all the values for jumps in the yellow frequency
    Boolean init - Set to true to initialise the scalers, when false it uses previously initialised ones.
    '''
    def clustScale(self,jmpGate,jmpYellow,init=False):
        if init or not hasattr(self,'scG') or not hasattr(self,'scY'):
            self.scG = StandardScaler()
            self.scY = StandardScaler()
            return [np.ravel(self.scG.fit_transform(jmpGate.reshape(-1,1))),np.ravel(self.scY.fit_transform(jmpYellow.reshape(-1,1)))]
        return [np.ravel(self.scG.transform(jmpGate.reshape(-1,1))),np.ravel(self.scY.transform(jmpYellow.reshape(-1,1)))]
    
    '''
    Evaluates a cluster
    [gate jump, yellow jump] jumps - The data for which the cluster is to be evaluated
    Clustering clust - The clustering to evaluate, when set to 'auto' it uses self.clusterer
    Boolean ignoreZero - Whether or not to skip the zero when making the predictions
    Boolean removedZero - Whether or not the zero was removed for the clustering
    String metric - When 'clusterSteps' use the cluster steps metric
                  - When 'distance' use the distance metric (not as good)
    '''
    def evaluate(self,jumps,clust='auto',ignoreZero=True,removedZero=False,metric='clusterSteps'):
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
        sp = StatsPredictor(ignoreZero=ignoreZero,removedZero=removedZero)
        if removedZero:
            sp.filterZero(jumps)
        sp.fitClustered(self.moveNonCluster(trainLabels))
        
        if metric is 'distance':
            hitTime = self.clusterDistanceMetric(jumps,trainCut,clust,sp)  
        elif metric is 'clusterSteps':
            hitTime = self.clusterStepsMetric(jumps,trainCut,clust,sp)
        return np.mean(hitTime)
    
    '''
    For this metric the score for a points is:
        dist=distance(newPoint,predictedClusterCenter)
    Final score is the average of all these distances
    
    NOTE: In reality this metric only takes the distance to the 0 cluster (the 
    cluster that is always checked first) into account. Adding together the distances
    to all tried clusters doesn't really make a lot of sense either, so this metric
    is just not very good. Use clusterStepsMetric instead.
    
    [gate jump, yellow jump] jumps - The data for which the cluster is to be evaluated
    int trainCut - The number of jumps in the trainSet
    Clustering clust - The clustering to be evaluated
    StatsPredictor statsPred - The StatsPredictor used for the predictions
    '''
    def clusterDistanceMetric(self,jumps,trainCut,clust,statsPred):
        labels = jumps.shape[0]
        if statsPred.removedZero:
            s = statsPred.densityKern.score_samples(jumps)
            labels[s<self.densVal] = (self.moveNonCluster(clust.labels_)+1)
        else:            
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
            #This only checks distance to 0 cluster, but adding distances untill the correct cluster
            #doesn't really make sense either            
            hitTime[i] = dist
            statsPred.foundNextCluster(testLabels[i])

        return hitTime


    '''
    This metric the score for a point is the total number of steps needed before the 
    cluster the point is in is completely searched. The final score is the average of these.
    
    NOTE: The estimate for the size is currently a bit rough, especially since clusters that 
    consist of two clusters (i.e. one on both sides of the plot) are heavily punished.
    Splitting up the two parts of the cluster when calculating the area could be a good improvement.
    
    [gate jump, yellow jump] jumps - The data for which the cluster is to be evaluated
    int trainCut - The number of jumps in the trainSet
    Clustering clust - The clustering to be evaluated
    StatsPredictor statsPred - The StatsPredictor used for the predictions
    '''
    def clusterStepsMetric(self,jumps,trainCut,clust,statsPred):
        labels = np.zeros(jumps.shape[0])
        if statsPred.removedZero:
            s = statsPred.densityKern.score_samples(jumps)
            labels[s<self.densVal] = self.moveNonCluster(clust.labels_)+1
        else:            
            labels = clust.labels_
            labels = self.moveNonCluster(labels)
        #Change space to step-scale
        scJumps = self.scaleClustSteps(jumps)
        #For each cluster determine the size, with pca and a rectangle
        classes = np.unique(labels)
        clustSteps = np.zeros(classes.shape) #clustSteps essentially holds our weight for each class
        for i,c in enumerate(classes):
            #TODO: Find possible gaps in the data to refine size estimate?
            newData = PCA(n_components=2).fit_transform(scJumps[labels==c,:])
            width = np.max(newData[:,0])-np.min(newData[:,0])
            height = np.max(newData[:,1])-np.min(newData[:,1])
            clustSteps[i] = width*height
        #For each step needed to find the correct cluster *cluster size
        testLabels = labels[trainCut:]
        hitTime = np.zeros((len(testLabels),1))
        for i in range(len(testLabels)):
            prediction = statsPred.predictNext()
            for p in np.argsort(prediction):
                hitTime[i]+=clustSteps[p]
                if p==np.argmax(prediction):
                    break
        return hitTime
    
    '''
    Returns the jumps scaled by the attraction sizes.
    
    [gate jump, yellow jump] jumps - The data for which the cluster is to be evaluated
    '''
    def scaleClustSteps(self,jumps):
        scJumps=jumps.copy()
        scJumps[:,0]=self.scG.inverse_transform(jumps[:,0])
        scJumps[:,1]=self.scY.inverse_transform(jumps[:,1])
        scJumps[:,0] = scJumps[:,0]/self.attractmV
        scJumps[:,1] = scJumps[:,1]/self.attractFreq
        return scJumps
    
    '''
    Returns the centres of the clusters
    [gate jump, yellow jump] jumps - The data for which the cluster centres need to be retrieved
            When set to auto, use the internal jumps
    Clusterer clust - The Clustering for which the centres are to be obtained
            When set to auto, use the internal clusterer
    '''
    def getClusterCentres(self,jumps='auto',clust='auto'):
        if jumps is 'auto':
            jumps=np.vstack((self.scaledJumpGate,self.scaledJumpYellow)).T
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
    
    '''
    This function tries out a wide range of clusterings and selects the one that
    allows for the best predictions.
    
    NOTE: This function takes a long time to run, fortunatly you should only need to do it once.
    To speed it up you could lower the amount of parameters tested, at the risk of skipping the best one.
    
    [gate jump, yellow jump] jumps - The data for which we need to find a clustering
    Boolean quick - When set to True, it uses a clustering found to be best for NV1, 
            rather than testing everything
    '''
    def findClustering(self,jumps,quick=False):
        self.zeroDens=False
        spectBands=np.arange(3,10,1)
        gamma=np.concatenate((np.arange(0.1,1,0.1),np.arange(1,3,0.2),np.arange(3,10,1)))
        epsilon=np.arange(0.1,0.7,0.1)
        minSamples=range(10,100,10)
        jumps0 = self.filterZero(jumps)
        bestScore = sys.maxsize
        bestClust = 0
        clustString=''
        
        if quick:
            self.zeroDens=True
            clust = SpectralClustering(8,gamma=0.3).fit(jumps0)
            self.clusterer=clust
            self.plotClustering(figNum=201)
            print(self.evaluate(jumps,clust,False,True))
            return self.getLabel(jumps,self.clusterer.labels_)        
        #Try all DBSCAN
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
                    score = self.evaluate(jumps,clust,False,True)
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
                            clustString='Spectral   spect: '+ str(spect)+' gamma: '+str(gam)+' Score: '+ str(score)+' Cluster count: '+str(len(np.unique(clust.labels_)))
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
                            clustString='Spectral use0  spect: '+ str(spect)+' gamma: '+str(gam)+' Score: '+ str(score)+' Cluster count: '+str(len(np.unique(clust.labels_)))
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
                    score = self.evaluate(jumps,clust,False,True)
                    if score > 0:
                        if score<bestScore:
                            bestClust = clust
                            bestScore=score
                            self.zeroDens=True
                            clustString='Spectral_ignore0   spect: '+ str(spect)+' gamma: '+str(gam)+' Score: '+ str(score)+' Cluster count: '+str(len(np.unique(clust.labels_)))
                        if self.verbose:
                            print('Spectral_ignore0   spect: ', spect,' gamma: ',gam,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                    elif self.verbose:
                        print('---Only 0 in trainset--- Spectral_ignore0   spect: ', spect,' gamma: ',gam,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
                elif self.verbose:
                    print('---No clustering--- Spectral_ignore0   spect: ', spect,' gamma: ',gam,' Score: ', score,' Cluster count: ',len(np.unique(clust.labels_)))    
        self.clusterer = bestClust
        if self.zeroDens:
            self.clusterer = self.clusterer.fit(jumps0)
        else:
            self.clusterer = self.clusterer.fit(jumps)
        print('======= Best cluster performance: ', bestScore)
        print(clustString)
        if self.verbose:
            self.plotClustering(figNum=201)
        return self.getLabel(jumps,self.clusterer.labels_)

    '''
    Assigns labels to the given jumps based on the label of the nearest neighbour
    
    [gate jump, yellow jump] jumps - The jumps to be labelled
    [] lbls - if 'auto' use the labels as assigned by the clustering of this instance
            otherwise use provided labels (provided in order)
    '''
    def getLabel(self,jumps,lbls='auto'):
        kdt = self.kdTree
        if lbls is 'auto':
            lbls=self.labels
        if len(lbls) < len(self.scaledJumpGate): #when we have been given fewer labels than the total set, it must be because the 0 cluster was removed
            kdt = self.kdTree0
        lbls = self.moveNonCluster(lbls)
        labels = np.ravel(lbls[kdt.query(jumps,1,False)])
        if self.zeroDens:#Handle zero
            labels = labels+1
            s = self.densityKern.score_samples(jumps)
            labels[s>=self.densVal] = 0
        return labels

    '''
    Converts the supplied data into relative jump sizes
    [time, gate, yellow] X - The data that needs to be transformed
    ''' 
    def toJumps(self,X):
        X=np.array(X)
        s=X.shape
        if len(s)<2:
            X=np.reshape(X,(1,3))
        if not self.prevX is 'init': #If this isn't the first entry, add the last value to the start
            X=np.append(self.prevX,X) #Otherwise jumps aren't well defined
            self.prevX = X[-1,:]
        time = X[:,0]
        yellow = X[:,1]
        gate = X[:,2]
        dt = np.median(np.diff(time))
        
        jumpSelect = (np.diff(time)>3*dt) & (np.diff(time)<30*60) # take three times the median as a qualifier for a jump
        jumpGate = gate[np.append(False, jumpSelect)] - gate[np.append(jumpSelect, False)]
        jumpYellow = yellow[np.append(False, jumpSelect)] - yellow[np.append(jumpSelect, False)]
        return [jumpGate,jumpYellow]
    
    '''
    Call when a new value is found, determines whether it is a jump or not,
    and if it is, it is added to the internal state and used in further predictions
    
    NOTE: A possible extension here is that after a significant amount of new jumps,
    refit is called, as it could potentially find a better suited cluster.
    
    [time, gate, yellow] X - a single new datapoint
    '''
    def foundNextValue(self,X):
        jumpGate,jumpYellow = self.toJumps(X)
        if not jumpGate.size == 0:
            sclGateJump, sclYellowJump = self.clustScale(jumpGate,jumpYellow)
            self.jumpGate = np.append(self.jumpGate,jumpGate)
            self.jumpYellow = np.append(self.jumpYellow,jumpYellow)
            self.scaledJumpGate = np.append(self.scaledJumpGate,sclGateJump)
            self.scaledJumpYellow = np.append(self.scaledJumpYellow,sclYellowJump)
            self.X = np.append(self.X,X)
            newLab = self.getLabel(np.vstack((sclGateJump,sclYellowJump)).T)
            return self.foundNextCluster(newLab)
    
    '''
    Filters the zero cluster (the densest cluster) out of the jumps
    
    [jumpGate, jumpYellow] jumps - The jumps from which the densest cluster needs to be removed
    float cutoff - The percentage of the data to be removed
    '''
    #Cutoff as percentage of data in the 0 cluster
    def filterZero(self,jumps,cutoff=0.65):
        self.densityKern = KernelDensity().fit(jumps)
        s = self.densityKern.score_samples(jumps)
        self.densVal=np.sort(s)[::-1][int(len(jumps)*cutoff)]
        return np.vstack((jumps[s<self.densVal,0],jumps[s<self.densVal,1])).T
    
    '''
    Gets the indices of all the occurences of lastSeq
    [cluster sequences] lagDat - A matrix containing all the sequences encountered thus far
    [cluster sequence] lastSeq - The sequence we are looking for
    '''
    def __getIndex(self,lagDat,lastSeq):
        index=np.array([],dtype='bool')
        for i in range(lagDat.shape[0]):
            index = np.append(index,np.array_equal(lagDat[i,:],lastSeq))
        return index
    
    '''
    Normally points that aren't clustered are given label -1, now they are given
    the highest label+1, as that is easier to deal with
    
    [labels] labels - The labels to be transformed
    '''
    def moveNonCluster(self,labels):
        uniq = np.unique(labels)
        if uniq[0] == -1:
            labels[labels==-1] = np.max(uniq)+1
        return labels
    
    '''
    Plots the supplied clustering
    
    Clustering clust - The clustering to be plotted, 'auto' uses self.clusterer
    [jumpGate, jumpYellow] jumps - the jump values to be used for the plot
    figNum - In case you want to plot it in a specific figure
    '''
    def plotClustering(self,clust='auto',jumps='auto',figNum=101):
        if clust is 'auto':
            clust=self.clusterer
        if jumps is 'auto':
            jumps=np.vstack((self.scaledJumpGate,self.scaledJumpYellow)).T
        plt.figure(figNum)
        labels = self.getLabel(jumps,clust.labels_)
        plt.scatter(jumps[:,0],jumps[:,1], c=labels, cmap=cm.jet, linewidths=0)
        plt.show()
    
    '''
    Every so often you might want to refit the model, as there might be a more optimal clustering.
    To speed the fitting up significantly you could do this by only checking a few variants close to the one currently in use.
    '''
    def refit(self):
        self.fit(self.X)
        