# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:16:50 2017

@author: Laurens
"""
#%% Load packages
from __future__ import print_function
from imp import reload
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

import pandas as pd
import seaborn as sns
import sklearn
import sklearn.cluster
from sklearn.cluster import DBSCAN, Birch, MiniBatchKMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import sklearn

import matplotlib.cm as cm
from qtt import pmatlab

import sklearn.manifold
from sklearn.manifold import TSNE

import nvtools
from nvtools.nvtools import labelMapping
from nvtools.nvtools import showModel

import qcodes
os.chdir(qcodes.config['user']['nvDataDir'])

labels=np.load('labels.npy')

#%%
print('Generating Data')
data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T

df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump'])
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

#%% Checking out t-SNE clustering

sklearn.manifold.TSNE(n_components=2)        
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
qq=model.fit_transform(Xbase)

#%% cluster on the t-SNE transformation
#db = DBSCAN(eps=0.8, min_samples=10).fit(qq) # fit centers
#db=Birch(threshold=0.4, branching_factor=2, compute_labels=True).fit(qq)
db=MiniBatchKMeans().fit(qq)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
try:
    core_samples_mask[db.core_sample_indices_] = True
except:
    pass
labels = db.labels_

encoder = sklearn.preprocessing.LabelEncoder ()
encoder.fit(labels)


# Number of clusters in labels, ignoring noise if present.
if 1:
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(qq, labels))

#plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)

plt.figure(); plt.clf();
if labels is None:
    plt.scatter(qq[:,0], qq[:,1])
else:
    plt.scatter(qq[:,0], qq[:,1], c=labels)
plt.title('t-SNE plot')

plt.figure(301); plt.clf(); plt.jet()
df.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), c=labels, cmap=cm.jet, linewidths=0, colorbar=False)

