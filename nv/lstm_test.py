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


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 20
epochs = 250
# number of elements ahead that are used to make the prediction

#%%
print('Generating Data')
data = np.load('jdata.npy')
data -= data.mean(1)[:, np.newaxis]
data /= data.std(1)[:, np.newaxis] * 3
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

epochs = 400
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
plt.figure(100)
plt.clf()
plt.plot(data_df_a['next_jump'].values, data_df_a['predicted_jump'].values, '.b')
pmatlab.plot2Dline([1, -1, 0], '--g')
plt.figure(101)
plt.clf()
plt.plot(data_df_b['next_jump'].values, data_df_b['predicted_jump'].values, '.b')
pmatlab.plot2Dline([1, -1, 0], '--g')

import pmatlab
pmatlab.tilefigs([0, 100, 0, 101])
