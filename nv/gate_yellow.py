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
from sklearn.cluster import DBSCAN, Birch
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
import pmatlab

import sklearn.manifold

os.chdir('/home/eendebakpt/svn/qutech/qtt/nv')
import nvtools
from nvtools import labelMapping
from nvtools import showModel

#%%
print('Generating Data')
data = np.load(os.path.join(os.path.expanduser('~'), 'tmp', 'jdata.npy')).T


df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump'])
plt.figure(300); plt.clf()
df.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), linewidths=0)

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

#plt.figure(100); plt.clf(); plt.plot(x,y, '.b'); plt.axis('image')

#%% Learn clusters


X=Xbase
db = DBSCAN(eps=0.2, min_samples=10).fit(X) # fit centers
#db=Birch(threshold=0.15, branching_factor=3, compute_labels=True).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
try:
    core_samples_mask[db.core_sample_indices_] = True
except:
    pass
labels = db.labels_

encoder = sklearn.preprocessing.LabelEncoder ()
encoder.fit(labels)


# Number of clusters in labels, ignoring noise if present.
if 0:
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

#plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)


plt.figure(301); plt.clf(); plt.jet()
df.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), c=labels, cmap=cm.jet, linewidths=0, colorbar=False)



from deeptools import clusterCenters
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
      
      
np.save(os.path.join(os.path.expanduser('~'), 'tmp', 'labels.npy'), labels)

      
#%% tSNE

from deeptools import showTSNE
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

from deeptools import Trainer
        
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
      
    
    
#%% Predict jump type
#    
# Input data: jump labels, other data...   
#

Xtrain=labels[train_idx]
Ytrain=0

#%% Naive Bayes?

# probability given no training data
# probability given previous jump (or previous n-jumps)
#
#

if 0:
    from sklearn import datasets
    iris = datasets.load_iris()
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print("Number of mislabeled points out of a total %d points : %d"
          % (iris.data.shape[0],(iris.target != y_pred).sum()))
    
#%%

if 0:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
#%% Test a LSTM approach to predicting jumps

if 0:
    X=data[4:,np.newaxis,:].T
    
    m = Sequential()
    m.add(LSTM(5, input_dim=X.shape[1], return_sequences=True))
    m.add(LSTM(5, return_sequences=True))
    m.compile(loss='mse', optimizer='adam')
    m.fit(X, X)
    


#%%
# since we are using stateful rnn tsteps can be set to 1
if 0:
    tsteps = 1
    batch_size = 20
    epochs = 250
# number of elements ahead that are used to make the prediction

#%% Data data: only select the large jump
if 0:
    jump_size = np.sqrt(np.sum(data[-2:, :]**2, 0))
    
    datax = data[:, jump_size > 0.25]
    datax = datax[:, :]
    time = datax[0, :]
    
    X_data = datax.T[:-1, np.newaxis, 4:]
    Y_data = datax.T[1:, 4:]
    
    print('Input shape:', X_data.shape)
    
    print('Output shape')
    print(Y_data.shape)
    
    #%%
    print('Creating Model')
    model = Sequential()
    model.add(LSTM(10,
                   batch_input_shape=(batch_size, tsteps, 2),
                   return_sequences=True,
                   stateful=True))
    model.add(LSTM(10,
                   batch_input_shape=(batch_size, tsteps, 2),
                   return_sequences=False,
                   stateful=True))
    model.add(Dense(2))
    model.compile(loss='mse', optimizer='rmsprop')
    
    #%%
    
    epochs=40
    loss = []
    print('Training')
    for i in range(epochs):
        print('Epoch', i, '/', epochs)
        l = model.fit(X_data,
                      Y_data,
                      batch_size=batch_size,
                      verbose=1,
                      nb_epoch=1,
                      shuffle=False)
        model.reset_states()
        loss.append(l.history['loss'][0])
    
    #%%
    print('Predicting')
    predicted_output = model.predict(X_data, batch_size=batch_size)
    
    data_df_a = pd.DataFrame(X_data[:, 0, 0], columns=['jump'])
    data_df_a['time'] = time[:-1]
    data_df_a['next_jump'] = Y_data[:, 0]
    data_df_a['type'] = 'a'
    data_df_a['predicted_jump'] = predicted_output[:, 0]
    
    data_df_b = pd.DataFrame(X_data[:, 0, 1], columns=['jump'])
    data_df_b['time'] = time[:-1]
    data_df_b['next_jump'] = Y_data[:, 1]
    data_df_b['type'] = 'b'
    data_df_b['predicted_jump'] = predicted_output[:, 1]
    
    data_df = pd.concat([data_df_a, data_df_b])
    
    g = sns.FacetGrid(data_df, row='type', aspect=1)
    g.map(plt.scatter, 'next_jump', 'predicted_jump')
    g.set(xlim=(-2, 2), ylim=(-2, 2), )
    
    g = sns.FacetGrid(data_df, row='type', aspect=1)
    g.map(plt.hist, 'next_jump', bins=100)
    
    #%%
    plt.figure(100); plt.clf()
    plt.plot(data_df_a['next_jump'].values, data_df_a['predicted_jump'].values, '.b' )
    pmatlab.plot2Dline([1,-1,0], '--g')
    plt.figure(101); plt.clf()
    plt.plot(data_df_b['next_jump'].values, data_df_b['predicted_jump'].values, '.b' )
    pmatlab.plot2Dline([1,-1,0], '--g')
    
    import pmatlab
    pmatlab.tilefigs([0,100,0,101])