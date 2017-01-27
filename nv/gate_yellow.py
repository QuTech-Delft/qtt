'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''

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
from sklearn.cluster import DBSCAN, Birch, KMeans, AffinityPropagation, MeanShift,SpectralClustering
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
import qcodes
import nvtools
from nvtools.nvtools import labelMapping
from nvtools.nvtools import showModel

from sklearn.neighbors import KernelDensity

# pip install statsmodels --user
from statsmodels.graphics.gofplots import qqplot

os.chdir(qcodes.config['user']['nvDataDir'])

#%%
print('Generating Data')
data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T
df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump','jump index'])
if 0:
    plt.figure(300); plt.clf()
    df.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), linewidths=0)
    plt.figure(300); plt.clf()
    df.plot(kind='scatter', x='gate', y='yellow', ax=plt.gca(), linewidths=0)
    plt.figure()
    plt.subplot(221)
    plt.title('Yellow frequency jumps over time')
    df.plot(kind='scatter', x='yellow jump', y='time', ax=plt.gca(), linewidths=0)
    plt.subplot(222)
    plt.title('Gate voltage jumps over time')
    df.plot(kind='scatter', x='gate jump', y='time', ax=plt.gca(), linewidths=0)
    plt.subplot(223)
    plt.title('Yellow frequency over time')
    df.plot(kind='scatter', x='yellow', y='time', ax=plt.gca(), linewidths=0)
    plt.subplot(224)
    plt.title('Gate voltage over time')
    df.plot(kind='scatter', x='gate', y='time', ax=plt.gca(), linewidths=0)

#%% Data needs to be scaled for almost any machine learning algorithm to work

#data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata2.npy')).T
df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump','jump index'])

# translate by mean and scale with std
datascaler= StandardScaler()
dataS = datascaler.fit_transform(data)
dfS=df.copy()
dfS[:]=datascaler.transform(df)

Xbase=dataS[:,4:6] # base data
datascalerBase = StandardScaler().fit(data[:,4:])
x=dataS[:,4]
y=dataS[:,5]

#plt.figure(100); plt.clf(); plt.plot(x,y, '.b'); plt.axis('image')

#%% Learn clusters
X=Xbase
db = DBSCAN(eps=0.2, min_samples=10).fit(X) # fit centers
#db=Birch(threshold=0.15, branching_factor=3, compute_labels=True).fit(X)
#db=SpectralClustering(5,gamma=0.2).fit(X)
#db=KMeans(n_clusters=7).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
try:
    core_samples_mask[db.core_sample_indices_] = True
except:
    pass
labels = db.labels_

encoder = sklearn.preprocessing.LabelEncoder()
encoder.fit(labels)
    
# Number of clusters in labels, ignoring noise if present.
if 0:
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

#plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
plt.figure(301); plt.clf(); plt.jet()
df.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), c=labels, cmap=cm.jet, linewidths=0, colorbar=False)

np.save(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'), labels)

#%% Find dense 0 cluster
densityKern = KernelDensity().fit(X)
s = densityKern.score_samples(X)
plt.figure()
plt.subplot(121)
plt.scatter(df['gate jump'],s)
plt.subplot(122)
plt.scatter(df['yellow jump'],s)

X = X[s<-2.5,:]
#%%
# translate by mean and scale with std

#db = DBSCAN(eps=0.5, min_samples=50).fit(X) # fit centers
db=SpectralClustering(7,gamma=0.2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
try:
    core_samples_mask[db.core_sample_indices_] = True
except:
    pass
labels = db.labels_

encoder = sklearn.preprocessing.LabelEncoder()
encoder.fit(labels)

plt.figure(301); plt.clf(); plt.jet()
plt.scatter(X[:,0],X[:,1],c=labels,cmap=cm.jet)
#dfNoCentre.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), c=labels, cmap=cm.jet, linewidths=0, colorbar=False)

#%%
from nvtools.nvtools import clusterCenters, showTSNE
chars, l2i, i2l = labelMapping(labels)    
ll=chars
cc=clusterCenters(db, Xbase, ll)
cc=datascalerBase.inverse_transform(cc)
pmatlab.plotPoints(cc.T, '.k')
pmatlab.plotLabels(cc.T, (ll) )

if 1:
    plt.figure(303); plt.clf(); plt.jet()
    dfS.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), c=labels, cmap=cm.jet, linewidths=0, colorbar=False)
    cc=clusterCenters(db, Xbase)
    pmatlab.plotLabels(cc.T, ['%d: %s' % (x,y) for (x,y) in zip(range(len(ll)), (ll)) ])
        
pmatlab.tilefigs([300,301, 303])
#plt.figure(400);plt.clf(); plt.jet() ;plt.scatter(X[:,0], X[:,1], c=labels.astype(np.int)+1)

      
#%% tSNE

showTSNE(Xbase, labels=labels, fig=400)

#%% Split into test and train

n=int(data.shape[0]/2)
train_idx = list(range(0, n))
test_idx = list(range(n, data.shape[0]))

#%% Linear regression like model for jump size prediction

X0 = labels.reshape( (-1,1)) # jump label

encoded_X = encoder.transform(X0.flatten())
# convert integers to dummy variables (i.e. one hot encoded)
X = np_utils.to_categorical(encoded_X)

label_idx = encoder.transform(X0.flatten())

def processInput(x):
    encoded_X = encoder.transform(x.flatten())
    X = np_utils.to_categorical(encoded_X)
    return X    

Y = dataS[:,4:] # jump size

model = Sequential(name='linear regressor like')
nhidden=7
model.add(Dense(nhidden,input_dim=X.shape[1], init='uniform', activation='linear'))
model.add(Dense(2,input_dim=nhidden, init='uniform', activation='linear'))
model.compile(loss='mse', optimizer='sgd')
model.summary()

showModel(model)

#model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr=.01))

seed=7
np.random.seed(seed)

if 0:
    estimator = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=5, verbose=1)
    
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))




#%%
X0_train=X0[train_idx,:]
X_train=X[train_idx,:]
Y_train=Y[train_idx,:]

from nvtools.nvtools import Trainer
        
trainer = Trainer(model, X_train, Y_train)
_=trainer.train()
_=trainer.train()
_=trainer.train()
trainer.plotLoss(fig=50)    
#loss=[]
#loss=trainN(model, loss)   
 
#%%

#%% Todo: 
#
# - scatter plot of predicted errors (with colors for each class)
# - add more input to the model
# - restructure
# proper cost function
# - learn probability of next jump (bayes)


#%%
sc=model.evaluate(X, Y)

qq=np.unique(X0); cl=processInput(qq)
ccpred=model.predict(cl)
print(ccpred)

plt.figure(301); plt.clf(); plt.jet()
dfS.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), c=labels, cmap=cm.jet, linewidths=0, colorbar=False)

plt.plot(ccpred[:,0], ccpred[:,1], '.k', linewidth=5, label='predicted centers' )

for i,c in enumerate(ccpred):
    pmatlab.plotLabels(c.reshape( (-1,1) ), '%d: %s' % (i, c) )

# plot centers
for i in [0,1,2,3,4]:
    ii=(X0_train==i).flatten()
    cc=Y_train[ii,:].mean(0).reshape( (-1,1))
    #plt.plot(Y_train[ii,0], Y_train[ii,1], '.', linewidth=5 )
    pmatlab.plotPoints(cc, '.k')
    pmatlab.plotLabels(cc, i)
    
#%% Error
    
errors=dfS[['gate jump','yellow jump']] - model.predict( processInput(labels) )
errors[:]=datascalerBase.inverse_transform(errors)

plt.figure(301); plt.clf(); plt.jet()
errors.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), c=labels, cmap=cm.jet, linewidths=0, colorbar=False)
plt.title('Error in cluster step prediction')

cc=clusterCenters(db, Xbase, labels=ll)
Yavg=cc[label_idx[train_idx], :]
    
# errors
eavg=model.evaluate(X_train, Yavg, verbose=0)
etrain=model.evaluate(X_train, Y_train, verbose=0)
etest=model.evaluate(X[test_idx,:], Y[test_idx,:], verbose=0)
print('error: averages: %.3f, error: train %.3f, test %.3f' % (eavg, etrain, etest))
    
print('TODO: better predictions... (take other variables into account)')
      
    
    
