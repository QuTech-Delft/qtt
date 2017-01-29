import numpy
import time
try:
    import pygpu
except:
    pass
from theano import function, config, shared, tensor, sandbox

from matplotlib import pyplot as plt
import sklearn
import sklearn.manifold
import numpy as np
import copy

from sklearn.manifold import TSNE

#%%

from contextlib import contextmanager
import sys
import os


@contextmanager
def suppress_stdout():
    """ Suppress output 

    Example:
       >>> with suppress_stdout():
       >>>    print("You cannot see this")

    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def remove_empty_intervals(data, thr=30 * 60):
    """ Remove empty intervals 

    Args:
        data (list): each element is a list with the first element a numpy array with time
        thr (float): time threshold in seconds
    Returns:
        allData (list): list with merged arrays
        debugvar (Anything)
    """
    stitchIndices, stitchTimeDiff = list(range(len(data))), list(range(len(data)))

    for i, d in zip(range(len(data)), data):
        timeDiff = np.diff(d)
        stitchTimeDiff[i] = np.append(np.array([0]), timeDiff[timeDiff > 30 * 60])
        stitchTimeDiff[i] = np.cumsum(stitchTimeDiff[i])
        # find times whenever we were idle for more than thr seconds
        ind1, ind2 = np.where(timeDiff > thr)
        stitchIndices[i] = np.append(ind2[ind1 == 0], np.array([len(d[0]) - 1]))
        stitchIndices[i] = np.append(stitchIndices[i][0], np.diff(stitchIndices[i]))

        # create an array that corrects for the idle times
        subtraction_arr = np.array([0])
        for j, inds, diff in zip(range(len(stitchIndices[i])), stitchIndices[i], stitchTimeDiff[i]):
            subtraction_arr = np.append(subtraction_arr, np.ones(inds) * diff)

        # manipulate the original time series by setting it initially to 0 and adjusting the idle time with the subtraction array
        data[i][0] = np.subtract(data[i][0], subtraction_arr) - data[i][0][0]

    allData = [None] * len(data)  # np.array([]),np.array([]),np.array([]),np.array([])]
    allStitchIndices = []
    for i in range(len(data)):
        if i == 0:
            allData = copy.deepcopy(data[i])
        else:
            for j, d in zip(range(len(data[i])), data[i]):
                if j == 0:
                    addtotime = allData[0][-1]
                else:
                    addtotime = 0
                allData[j] = np.append(allData[j], d + addtotime)

    return allData, data


def extract_data(filename, gate_scaling):
    """ Extract data from a datafile """
    x = np.array([])
    y = np.array([])
    y2 = np.array([])
    y3 = np.array([])
    ii = 0
    with open(filename, "r") as f:

        for line in f.readlines():
            ii = ii + 1
            # print('## line %d' %ii)
            if line[0] in str(list(range(10))):
                srch = "\t"
                xapp, yapp, y2app, y3app = line.split(srch)
                #print('%.1f,%.1f,%.1f,%.1f' % ( float(xapp), float(yapp), float(y2app), float(y3app)) )
                x = np.append(x, float(xapp))    # corresponds to elapsed time
                y = np.append(y, float(yapp))    # yellow frequency
                y2 = np.append(y2, float(y2app))  # gate voltage
                y3 = np.append(y3, float(y3app))  # newfocus frequency --> mostly ignored for the moment
            else:
                pass
                #print('funny line: |%s|' % line)
                #print(' |%s|' % line[0])

    # need to clean data up in case 'wrong value' was recorded. this can happen with the laser freuqencies if the Wavemeter has got no signal
    filter_cond = (-2000 < y2 * gate_scaling) & (12 < y)
    filter_cond = filter_cond & (2000 > y2 * gate_scaling) & (y < 100)

    return [x[filter_cond], y[filter_cond], gate_scaling * y2[filter_cond], y3[filter_cond]]


#%%
def testTheano():
    ''' That the Theano package '''
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

    def __init__(self, model, X, Y, loss=None, batch_size=20, verbose=1):
        """ Helper class for training a Keras model """
        if loss is None:
            loss = []
        self.loss = loss
        self.verbose = verbose
        self.batch_size = batch_size
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
            plt.figure(fig)
            plt.clf()
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
    l = db.labels_

    if labels is None:
        ss = set(db.labels_)
    else:
        ss = labels
    # X=db.components_

    cc = np.zeros((len(ss), 2))
    for i, s in enumerate(ss):
        ii = (l == s).flatten()
        cc[i, :] = X[ii, :].mean(0).reshape((-1, 1)).flatten()
        #plt.plot(Y_train[ii,0], Y_train[ii,1], '.', linewidth=5 )
    return cc


def labelMapping(labels):
    ''' Helper function '''
    chars = sorted(list(set(labels)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    return chars, char_indices, indices_char


def showModel(model, fig=10):
    """ Show keras model """
    import keras
    import cv2
    import tempfile
    from IPython.display import SVG
    from keras.utils.visualize_util import model_to_dot

    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    from keras.utils.visualize_util import plot
    mfile = tempfile.mktemp(suffix='.png')
    plot(model, to_file=mfile)

    im = cv2.imread(mfile)
    plt.figure(fig)
    plt.clf()
    plt.imshow(im)
    plt.axis('off')
    plt.title('Model %s' % model.name)


def showTSNE(X, labels=None, fig=400):
    """ Visualize a dataset using t-SNE """
    sklearn.manifold.TSNE(n_components=2)

    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    qq = model.fit_transform(X)

    plt.figure(fig)
    plt.clf()
    if labels is None:
        plt.scatter(qq[:, 0], qq[:, 1])
    else:
        plt.scatter(qq[:, 0], qq[:, 1], c=labels)
    plt.title('t-SNE plot')

#%%


def fmt(x, ndigits=3):
    """ Format numbers """
    v = [('%%.%df' % ndigits) % p for p in x]
    return ', '.join(v)


def avg_steps(y_true, y_pred, verbose=0):
    """ Calculate average number of steps needed for finding the correct cluster """
    A = 0.
    for i, g in enumerate(y_true):
        v = y_pred[i]
        idx = np.argsort(v)[::-1]
        j = int((idx == g).nonzero()[0])
        A += j + 1

        if verbose:
            print('i %d: gt %d: %d' % (i, g, j))
    if verbose:
        print('A: %.3f' % A)
    A /= len(y_true)
    return A

    
#%% Visualization
import matplotlib.ticker as plticker

def add_attraction_grid(ax, attractmV, attractFreq, zorder=0):
    minorLocator = plticker.MultipleLocator(attractFreq)
    ax.yaxis.set_minor_locator(minorLocator)
    minorLocator = plticker.MultipleLocator(attractmV)
    ax.xaxis.set_minor_locator(minorLocator)
    # Set grid to use minor tick locations. 
    ax.grid(which = 'minor', linestyle='-', color=(.9,.9,.9), zorder=zorder)

    

def nv_plot_callback(plotidx, adata, fig=100, singlefig=True, *args, **kwargs):
    """ Callback function to plot NV centre data """
    verbose = kwargs.get('verbose', 1)
    if verbose:
        print('plotidx = %s' % plotidx)
    plt.figure(fig)
    plt.clf()
    if singlefig:
        plt.subplot(1, 2, 1)
    #dataidx = int(jumpdata[plotidx, 6])
    dataidx = plotidx
    plotSection(adata, list(range(dataidx - 60, dataidx + 100)), jumps=None, si=dataidx)
    if singlefig:
        plt.subplot(1, 2, 2)
    else:
        plt.pause(1e-4)
        plt.figure(fig + 1)
        plt.clf()
    plotSection(adata, list(range(dataidx - 60, dataidx + 100)), jumps=None, mode='freq', si=dataidx)
    plt.pause(1e-4)


def plotSection(allData, idx, jumps=None, mode='gate', si=None):
    """ Helper function to plot a section of data

    Args:
        allData (list of numpy arrays or numpy array): data with time in first element
        idx (array): indices to plot
        jumps (None or boolean array): positions of jumps
        mode (str): 'gate' or 'freq'
        si (index): index of special point
    """

    if isinstance(allData, list):
        pdata = np.array(allData).T
    else:
        pdata=allData
    idx=idx[idx<len(pdata)]
    
    x = pdata[idx,0]
    y = pdata[idx,1]
    y2 = pdata[idx,2]
    v=np.zeros( len(pdata) ).astype(bool); v[idx]=1
    if jumps is not None:
        plot_select = jumps & v  # [:-1]
    else:
        plot_select = []
    ax = plt.gca()
    if mode == 'gate':
        plt.plot(x, y2, 'x-', label='data')
        plt.plot(pdata[plot_select, 0], pdata[plot_select, 2], 'ro', label='jump')
        ax.set_xlabel('elapsed time (s)')
        ax.set_ylabel('Gate voltage (V)')
    else:
        plt.plot(x, y, 'x-', label='data')
        plt.plot(pdata[plot_select, 0], pdata[plot_select, 1], 'ro', label='freq')
        ax.set_xlabel('elapsed time (s)')
        # ax2.set_xlim([0,1000])
        ax.set_ylabel('Yellow frequency (GHz)')
    if si is not None:
        if mode == 'gate':
            plt.plot(pdata[si, 0], pdata[si, 2], '.y', markersize=12, label='special point')
        else:
            plt.plot(pdata[si, 0], pdata[si, 1], '.y', markersize=12, label='special point')

if __name__ == '__main__' and 0:
    plotSection(allData, range(si - offset, si - offset + 100), jumpSelect, mode='gate')

#%%
if __name__ == '__main__' and 0:
    testTheano()
