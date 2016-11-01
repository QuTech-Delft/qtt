'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''

#%%
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

import pandas as pd
import seaborn as sns

#%%
print('Generating Data')
data = np.load('jdata.npy')
data -= data.mean(1)[:, np.newaxis]
data /= data.std(1)[:, np.newaxis] * 3

xx=data[4:,:].T
x=data[4,:]
y=data[5,:]

plt.figure(100); plt.clf()
plt.plot(x,y, '.b')
plt.axis('image')

#%% Learn clusters

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

plt.figure(400);
plt.clf(); plt.jet()
plt.scatter(X[:,0], X[:,1], c=labels)
      
#%% tSNE
import sklearn

import sklearn.manifold
sklearn.manifold.TSNE(n_components=2)
#, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)[source]

import numpy as np
from sklearn.manifold import TSNE
#X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
qq=model.fit_transform(xx)

plt.figure(400); plt.clf();
plt.scatter(qq[:,0], qq[:,1])

#%%
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

data = data[:, jump_size > 0.25]
data = data[:, :-2]
time = data[0, :]

X_data = data.T[:-1, np.newaxis, 4:]
Y_data = data.T[1:, 4:]

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

epochs=400
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