# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:16:29 2016

@author: eendebakpt
"""

#%% Todo list
#
# * clean up code 
# * Get rid of the `zero cluster`?
# * 2-grams without the zero cluster
# * Auxilirary variable: last big step as input for machine learning
# * Why does the learning not converge?

#%% Load packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys,os
from theano import tensor as T

from nvtools.nvtools import Trainer

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import sklearn
import qcodes
from qtt import pgeometry

labels = np.load(os.path.join("C:/Users/Laurens/Documents/qtech/qdata",'labels.npy'))
#labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))
text=labels
print('corpus length:', len(labels))

#%%

#%% Naive
#
# make histogram

encoder = sklearn.preprocessing.LabelEncoder()
encoder.fit(labels)

from nvtools.nvtools import avg_steps, fmt

chars = sorted(list(set(labels)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))

textX=[char_indices[c] for c in text]

bc=np.bincount(textX)
prob=bc/bc.sum()


print('probability of each of the classes: %s' % fmt(prob))

import keras.backend as K


def avg_steps(y_true, y_pred, verbose=0):
    """ Calculate average number of steps needed for finding the correct cluster """
    A=0.
    for i, g in enumerate(y_true):
       v=y_pred[i]
       idx=np.argsort(v)[::-1]
       j = int((idx==g).nonzero()[0])
       A += j+1

       if verbose:
           print('i %d: gt %d: %d' % (i,g, j))
    if verbose:
       print('A: %.3f' % A)
    A /= len(y_true)
    return A

if 0:
    def avg_steps2(y_true, y_pred):
        A=0.
        for i, g in enumerate(y_true): # loop over the predictions
           v=y_pred[i]
           idx=np.argsort(v)[::-1]
           j = int((idx==g).nonzero()[0])
           A += j+1
        A /= len(y_true)
        return A
    
    from theano import tensor as T
    
    
    def avg_steps3(y_true, y_pred):
        # dummy code!
        return K.in_top_k(y_pred, y_true, k=3)
    
n=len(labels)
lx=encoder.transform(labels)
y_pred=np.tile( prob, (n, 1))
av1=avg_steps(lx, y_pred)    
print('  avg number of steps: %.3f' % av1)

  

#%% 2-grams
alphabet=chars

def two_grams(alphabet, textX, normalize=True):
    gg=np.zeros( ( len(alphabet), len(alphabet) ), dtype=float)
    for i in range(len(textX)-1):
        ix=textX[i]
        iy=textX[i+1]
        gg[iy, ix]+=1

    if normalize:        
        for j in range(gg.shape[1]):
            gg[:,j]=gg[:,j] / gg[:,j].sum()

    return gg
    
gg=two_grams(alphabet, textX)

#gg /= (len(text)-1 )    
plt.figure(100); plt.clf()
plt.imshow(gg, interpolation='nearest')
plt.axis('image')
plt.colorbar()
plt.xlabel('Class label')
plt.ylabel('Next')
    
plt.xticks(range(len(alphabet)), alphabet)    
plt.yticks(range(len(alphabet)), alphabet)    

pgeometry.tilefigs([100])

y_pred2 = np.vstack( (prob, gg[:, lx[:-1]].T ) )
#y_pred2 = np.vstack( (prob, gg[:, lx[1:]].T ) )
y_pred=np.tile( prob, (n, 1))

_=avg_steps(lx, y_pred, verbose=1)    
av2=avg_steps(lx, y_pred2, verbose=1)    
print('  avg number of steps (1-grams): %.5f' % av1)
print('  avg number of steps (2-grams): %.5f' % av2)

#%%  Sequences
chars = sorted(list(set(labels)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 4
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#%% Train LSTM like thing with custom loss function
import keras.backend as K

def top_k_categorical_accuracy(y_true, y_pred, k=3):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))

def top_k_categorical_accuracy_loss(y_true, y_pred, k=3):
    return 10-K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))

def avg_step_loss(y_true, y_pred):
    q1=K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), 1))
    q2=K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), 2))
    q3=K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), 3))
    q4=K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), 4))

    return 5 - (q1+q2+q3+q4)
#    return 10-K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k))

#avg_step_loss= top_k_categorical_accuracy_loss

# build the model: a single LSTM
print('Build model...')
model = Sequential(name='dummy')
model.add(LSTM(10, input_shape=(maxlen,  len(chars))))
#model.add(LSTM(10, input_shape=(maxlen, len(chars))))
#model.add(LSTM(10))
#model.add(LSTM(10, input_shape=(maxlen, 10)))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

import keras
optimizer = RMSprop(lr=0.02)
optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss=avg_step_loss, optimizer=optimizer, metrics=['accuracy', avg_step_loss])
#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', top_k_categorical_accuracy])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
trainer=Trainer(model, X, y)
trainer.train(epochs=10)
trainer.plotLoss(fig=10)


y_pred=model.predict(X)
y_true=y.argmax(axis=1)
av=avg_steps(y_true, y_pred)    
print('  avg number of steps: %.5f' % av)

#%% Distraction

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

pred=y_pred[10]    
sample(pred, 1)

#%% Distraction 2 (2-grams for reduced data)

qq=lx[lx>1]
print(qq)
gg=two_grams(alphabet, qq, normalize=False)

plt.figure(1)
plt.clf()
plt.imshow(gg, interpolation='nearest'); plt.axis('image')
plt.colorbar()
#plt.plot(qq, '.b')

plt.xticks(range(len(alphabet)), alphabet)    
plt.yticks(range(len(alphabet)), alphabet)    

    
#%%
    
import numpy

def create_dataset(dataset, look_back=1):
    # convert a sequence of values into a dataset matrix
	if len(np.array(dataset).shape)==1:
		dataset=np.array(dataset).reshape( (-1,1))
        
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


y_true=y
y_pred=model.predict(X)
k=3
q=K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k) )
print(q.eval())
q=(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k) ).eval()
print(q)


#%%
dataX, dataY = create_dataset(textX, look_back=4)

#processInput(x)
X = np_utils.to_categorical(dataX.flatten())
X=X.reshape( (-1, 4, 6))

y_predD= model.predict(X)

encoder.transform( dataX[2] )

avD=avg_steps(dataY, y_predD)    
print('  avg number of steps: %.3f' % av1)
print('  avg number of steps 2-grams: %.3f' % av2)
print('  avg number of steps char enc: %.3f' % avD)


  
#%%

# Naive LSTM to learn one-char to one-char mapping


# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = chars
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
print('example data:')
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	print ('%s -> %s' % (seq_in,  seq_out  ))

if 0:
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
    # normalize
    X = X / float(len(alphabet))
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
else:
    
    textX=np.array([char_to_int[c] for c in text])


    seq_length = 1
    X=np.array([char_to_int[c] for c in text[:-1]])
    Y=np.array( [char_to_int[c] for c in text[2:]] )


#%% ###########################################################################

#%% Debugging
 
 
dataX, dataY = create_dataset(textX, look_back=4)

X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / float(len(alphabet))
    
y = np_utils.to_categorical(dataY)

 #%%
if 0:    
    # increase data
    X=np.concatenate( (X,X), 0)
    y=np.concatenate( (y,y), 0)

# create and fit the model
model = Sequential()
model.add(LSTM(7, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','categorical_accuracy'])
model.fit(X, y, nb_epoch=5, batch_size=1, verbose=2)

# http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

#model.fit(X, y, nb_epoch=25, batch_size=1, verbose=2, shuffle=False)


# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for pattern in dataX:
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print ('%s -> %s' % (seq_in,  result  ))
    
#%%


for start_index in range(10,211,30):
    sentence = textX[start_index: start_index + maxlen]
    sentenceF = textX[start_index: start_index + maxlen + 1]
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char] = 1. # char_indices[char]

    preds = model.predict(x, verbose=0)[0]

    print('----')
    print('sequence: %s' % sentence)
    print('predicted naive: %s' % fmt(prob) )
    print('predicted:       %s' % fmt(preds) )
    print('true: %s' %  sentenceF )

import keras.backend as K


y_true=y
y_pred = model.predict(X, verbose=0)


acc=K.metrics.categorical_accuracy(y_true, y_pred).eval()
print('accuracy: %.5f' % acc)


acc=K.metrics.categorical_accuracy(y_true, prob).eval()
print('accuracy naive: %.5f' % acc)


prob2=gg[:, dataX].T

acc=K.metrics.categorical_accuracy(y_true, prob2).eval()
print('accuracy 2-grams: %.5f' % acc)

#%%

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()






# scores

        