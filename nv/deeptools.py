from theano import function, config, shared, tensor, sandbox
import numpy
import time
import pygpu

from matplotlib import pyplot as plt
import sklearn
import sklearn.manifold
import numpy as np

import sklearn.manifold
from sklearn.manifold import TSNE


#%%
def testTheano():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 10
    
    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], tensor.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, tensor.Elemwise) and
                  ('Gpu' not in type(x.op).__name__)
                  for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')

#%%

class Trainer:
    def __init__(self,  model, X, Y, loss=None, batch_size=20, verbose=1):
        """ Helper class for training a Keras model """
        if loss is None:
            loss = []
        self.loss=loss
        self.verbose=verbose
        self.batch_size=batch_size
        self.X = X
        self.Y = Y
        self.model = model
        
    def train(self, epochs=20):
        """ Train the model for a specified number of epochs """
        
        if self.verbose:
            print('Training')
        for i in range(epochs):
            if self.verbose:
                print('Trainer: epoch', i, '/', epochs)
            l = self.model.fit(self.X,
                          self.Y,
                          batch_size=self.batch_size,
                          verbose=1,
                          nb_epoch=1,
                          shuffle=False)
            self.model.reset_states()
            self.loss.append(l.history['loss'][-1])
        return self.loss
        
    def plotLoss(self, fig=None):
        """ Plot the loss function during training """
        if fig is not None:
            plt.figure(fig); plt.clf()
        plt.plot(self.loss, '.b') 
        plt.title(self.model.name)
        plt.xlabel('Epoch')

#%%
import keras as K
from keras.layers import Dense, LSTM


class BinaryEmbedding(Dense):
    def build(self, input_shape):
        super(BinaryEmbedding, self).build(input_shape)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return self.activation(K.dot(x, self.W))
        


#%%

def clusterCenters(db, X, labels=None):
    ''' Calculate cluster centres from a sklearn clustering '''
    l=db.labels_
    
    if labels is None:
        ss=set(db.labels_)
    else:
        ss=labels
    #X=db.components_
    
    cc=np.zeros( (len(ss), 2))
    for i, s in enumerate(ss):
        ii=(l==s).flatten()
        cc[ i,:]=X[ii,:].mean(0).reshape( (-1,1)).flatten()
        #plt.plot(Y_train[ii,0], Y_train[ii,1], '.', linewidth=5 )
    return cc


def labelMapping(labels):
    
    chars = sorted(list(set(labels)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    return chars, char_indices, indices_char

def showModel(model, fig=10):
    import keras
    import cv2
    import tempfile
    from IPython.display import SVG
    from keras.utils.visualize_util import model_to_dot
    
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    from keras.utils.visualize_util import plot
    mfile=tempfile.mktemp(suffix='.png')
    plot(model, to_file=mfile)
    
    im=cv2.imread(mfile)
    plt.figure(fig)
    plt.clf()
    plt.imshow(im)
    plt.axis('off')
    plt.title('Model %s' % model.name)


def showTSNE(X, labels=None, fig=400):
    sklearn.manifold.TSNE(n_components=2)
        
    #X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    qq=model.fit_transform(X)
    
    plt.figure(fig); plt.clf();
    if labels is None:
        plt.scatter(qq[:,0], qq[:,1])
    else:
        plt.scatter(qq[:,0], qq[:,1], c=labels)
    plt.title('t-SNE plot')


#%%        
if __name__=='__main__':
    testTheano()
        