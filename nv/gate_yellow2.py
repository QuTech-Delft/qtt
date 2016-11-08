# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:16:29 2016

@author: eendebakpt
"""

'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
#from __future__ import print_function

#%% Load packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils


labels=np.load('labels.npy')
text=labels
print('corpus length:', len(labels))

#%%
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

#%%

# build the model: a single LSTM
print('Build model...')
model = Sequential(name='dummy')
model.add(LSTM(10, input_shape=(maxlen,  len(chars))))
#model.add(LSTM(10, input_shape=(maxlen, len(chars))))
model.add(LSTM(10))
#model.add(LSTM(10, input_shape=(maxlen, 10)))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


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
trainer.train()
trainer.plotLoss(fig=10)

#%% Naive
#
# make histogram

textX=[char_indices[c] for c in text]

bc=np.bincount(textX)
prob=bc/bc.sum()

def fmt(x):
    v=['%.2f' % p for p in x]
    return ', '.join(v)

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


#%%
import numpy

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	if len(np.array(dataset).shape)==1:
		dataset=np.array(dataset).reshape( (-1,1))
        
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
 
 
dataX, dataY = create_dataset(textX, look_back=1)

X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / float(len(alphabet))
    
y = np_utils.to_categorical(dataY)

#%% 2-grams
alphabet=set(text)
gg=np.zeros( ( len(alphabet), len(alphabet) ), dtype=float)
for i in range(len(textX)-1):
    ix=textX[i]
    iy=textX[i+1]
    gg[iy, ix]+=1
    
for j in range(gg.shape[1]):
    gg[:,j]=gg[:,j] / gg[:,j].sum()

#gg /= (len(text)-1 )    
plt.figure(100); plt.clf()
plt.imshow(gg, interpolation='nearest')
plt.axis('image')
plt.colorbar()
plt.xlabel('Class label')
plt.ylabel('Next')
    
plt.xticks(range(len(alphabet)), alphabet)    
plt.yticks(range(len(alphabet)), alphabet)    

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

import keras as K


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

        