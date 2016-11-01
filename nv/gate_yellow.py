'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''

#%%
import os,sys
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

import pandas as pd
import seaborn as sns
import deeptools

#%%
print('Generating Data')
data = np.load(os.path.join(os.path.expanduser('~'), 'tmp', 'jdata.npy'))
data -= data.mean(1)[:, np.newaxis]
data /= data.std(1)[:, np.newaxis] * 3

xx=data[4:,:].T
x=data[4,:]
y=data[5,:]

#plt.figure(100); plt.clf(); plt.plot(x,y, '.b'); plt.axis('image')

#%%
df=pd.DataFrame(np.vstack( (x,y) ).T, columns=['gate jump', 'yellow jump'])
plt.figure(300); plt.clf()
df.plot(kind='scatter', x='gate jump', y=1, ax=plt.gca(), linewidths=0)

#%% Learn clusters
import sklearn
import sklearn.cluster
from sklearn.cluster import DBSCAN, Birch
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                            random_state=0)
X=xx
#X=np.hstack( (xx, data[3,:].reshape( (-1,1))))

X = StandardScaler().fit_transform(X)
#Compute DBSCAN
db = DBSCAN(eps=0.2, min_samples=10).fit(X)
#db=Birch(threshold=0.15, branching_factor=3, compute_labels=True).fit(X)
#db=sklearn.cluster.AffinityPropagation(damping=.7, max_iter=20, convergence_iter=15).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
try:
    core_samples_mask[db.core_sample_indices_] = True
except:
    pass
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
if 0:
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

#plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)

import matplotlib.cm as cm
import pmatlab

plt.figure(301); plt.clf(); plt.jet()
df.plot(kind='scatter', x='gate jump', y=1, ax=plt.gca(), c=labels, cmap=cm.jet, linewidths=0, colorbar=False)

pmatlab.tilefigs([300,301])
#plt.figure(400);plt.clf(); plt.jet() ;plt.scatter(X[:,0], X[:,1], c=labels.astype(np.int)+1)
      
#%% tSNE
import sklearn

import sklearn.manifold
sklearn.manifold.TSNE(n_components=2)
#, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)[source]

xxx=data[4:,:].T
#xxx=data[1:,:].T

import numpy as np
from sklearn.manifold import TSNE
#X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
qq=model.fit_transform(xxx)

plt.figure(400); plt.clf();
plt.scatter(qq[:,0], qq[:,1], c=labels)
plt.title('t-SNE plot')



#%% Split into test and train

n=int(data.shape[1]/2)

train_idx = list(range(0, n))
test_idx = list(range(n, data.shape[1]))

#%% Linear regression like model for jump size prediction

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X0 = labels.reshape( (-1,1)) # jump label

encoder = LabelEncoder()
encoder.fit(labels)
encoded_X = encoder.transform(X0.flatten())
# convert integers to dummy variables (i.e. one hot encoded)
X = np_utils.to_categorical(encoded_X)

def processInput(x):
    encoded_X = encoder.transform(x.flatten())
    # convert integers to dummy variables (i.e. one hot encoded)
    X = np_utils.to_categorical(encoded_X)
    return X    

Y = data[4:,].T # jump size

model = Sequential()
nhidden=7
model.add(Dense(nhidden,input_dim=X.shape[1], init='uniform', activation='linear'))
#model.add(Embedding(labels.max()+1, nhidden, init='uniform'))
#model.add(Dense(4,input_dim=X.shape[1], init='uniform', activation='relu'))
model.add(Dense(2,input_dim=nhidden, init='uniform', activation='linear'))
#model.compile(loss='mse', optimizer='rmsprop')
model.compile(loss='mse', optimizer='sgd')
import keras

#model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr=.01))

#model.optimizer.lr.get_value()

seed=7
np.random.seed(seed)

if 0:
    estimator = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=5, verbose=1)
    
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#%%
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils import np_utils


import keras as K

class BinaryEmbedding(Dense):
    def build(self, input_shape):
        super(BinaryEmbedding, self).build(input_shape)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return self.activation(K.dot(x, self.W))
        

#%%
if 0:
    max_features=8
    model = Sequential()
    model.add(Embedding(max_features, 128, dropout=0.2))
    model.add(Dense(18, input_dim=1 ))  # try using a GRU instead, for fun
#model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun

#%%
X0_train=X0[train_idx,:]
X_train=X[train_idx,:]
#X_train=X_train-X_train.mean(0)
Y_train=Y[train_idx,:]

#Y_train=np.hstack( (2*X_train, -Y_train[:,0:1] ) )

epochs=40
loss = []
print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    l = model.fit(X_train,
                  Y_train,
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=5,
                  shuffle=False)
    model.reset_states()
    loss.append(l.history['loss'][-1])
    
#%%
def plotLoss(loss):
    plt.figure(10); plt.clf()
    plt.plot(loss, '.b')    
    
plotLoss(loss)
#for j in range(5):
#    model.fit(X_train, Y_train, nb_epoch=30, batch_size=batch_size, verbose=1)

#%% Todo: 
#
# - scatter plot of predicted errors (with colors for each class)
# - add more input to the model
# - restructure
# - learn probability of next jump (bayes)


#%%
sc=model.evaluate(X, Y)


qq=np.unique(X0); cl=processInput(qq)
cc=model.predict(cl)
print(cc)

plt.figure(301); plt.clf(); plt.jet()
df.plot(kind='scatter', x='gate jump', y=1, ax=plt.gca(), c=labels, cmap=cm.jet, linewidths=0, colorbar=False)

plt.plot(cc[:,0], cc[:,1], '.k', linewidth=5 )

for i,c in enumerate(cc):
    pmatlab.plotLabels(c.reshape( (-1,1) ), '%d: %s' % (i, c) )

# plot centers
for i in [0,1,2,3,4]:
    ii=(X0_train==i).flatten()
    cc=Y_train[ii,:].mean(0).reshape( (-1,1))
    #plt.plot(Y_train[ii,0], Y_train[ii,1], '.', linewidth=5 )
    pmatlab.plotPoints(cc, '.k')
    pmatlab.plotLabels(cc, i)
    

#%% Test a LSTM approach to predicting jumps

if 0:
    X=data[4:,np.newaxis,:].T
    
    m = Sequential()
    m.add(LSTM(5, input_dim=X.shape[1], return_sequences=True))
    m.add(LSTM(5, return_sequences=True))
    m.compile(loss='mse', optimizer='adam')
    m.fit(X, X)
    
#%%

from keras.utils.visualize_util import plot
#plot(model, to_file='model.png')


#%%
# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 20
epochs = 250
# number of elements ahead that are used to make the prediction

#%%
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