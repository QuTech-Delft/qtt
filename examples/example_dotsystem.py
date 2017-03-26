#%% Load packages
from imp import reload
import numpy as np
from numpy import linalg as la
import itertools
import matplotlib.pyplot as plt
import time
import pdb

try:
    import graphviz
except:
    pass
try:
    import pmatlab
except:
    pass

#%%


import qtt
reload(qtt)
import qtt.simulation.dotsystem
reload(qtt.simulation.dotsystem)
from qtt.simulation.dotsystem import DotSystem, TripleDot


#%%
class DoubleDot(DotSystem):

    def __init__(self, name='doubledot'):
        super().__init__(name=name)

        self.ndots = 2
        self.makebasis(ndots=self.ndots, maxelectrons=3)
        self.varnames = ['det1', 'det2',
                         'osC1', 'osC2', 'isC1', 'isC2', 'tun1', 'tun2']
        self.varnames += itertools.chain(* [['eps%d%d' % (d + 1, orb + 1) for d in range(self.ndots)] for orb in range(0, self.maxelectrons)])
        self.makevars()
        self.makevarMs()

ds = DoubleDot()
self = ds

ds.osC1 = 37
ds.osC2 = 36
ds.osC3 = 38
ds.isC1 = 3
ds.isC2 = 2.4

ds.eps11 = .1
ds.eps12 = .15
ds.eps21 = .1
ds.eps22 = .15


plt.figure(20)
plt.clf()
ds.visualize(fig=20)
plt.title('Double dot')
pmatlab.tilefigs(20, [2, 2])

#%%


class FourDot(DotSystem):

    def __init__(self, name='doubledot'):
        super().__init__(name=name, ndots=4)

        self.makebasis(ndots=self.ndots, maxelectrons=2)
        self.varnames = ['det%d' % (i + 1) for i in range(self.ndots)]
        self.varnames += ['osC%d' % (i + 1) for i in range(self.ndots)]
        self.varnames += ['isC%d' % (i + 1) for i in range(self.ndots)]
        self.varnames += ['tun%d' % (i + 1) for i in range(self.ndots)]
        self.varnames += itertools.chain(* [['eps%d%d' % (d + 1, orb + 1) for d in range(self.ndots)] for orb in range(0, self.maxelectrons)])
        self.makevars()
        self.makevarMs()

ds = FourDot()
self = ds

for ii in range(ds.ndots):
    setattr(ds, 'osC%d' % (ii + 1), 35)
for ii in range(ds.ndots - 1):
    setattr(ds, 'isC%d' % (ii + 1), 3)

if 0:
    ds.eps11 = .1
    ds.eps12 = .15
    ds.eps21 = .1
    ds.eps22 = .15


plt.figure(20)
plt.clf()
ds.visualize(fig=20)
plt.title('Double dot')
pmatlab.tilefigs(20, [2, 2])


#%%
ds.resetMu(0)
ds.det2 = 4
# ds.isC3=3

paramnames = ['det1', 'det3']
minmax = [[-40, 80, -40, 80], [-40, -40, 80, 80]]
#minmax = 1.5*np.array([[-20,80,-20,80],[-20,-20,80,80]])
npointsx = 60
npointsy = 60
ds.makeparamvalues2D(paramnames, minmax, npointsx, npointsy)

ds.simulatehoneycomb(verbose=1, usediag=True)

plt.figure(10)
plt.clf()
plt.pcolor(ds.vals2D[paramnames[0]], ds.vals2D[paramnames[1]], ds.honeycomb, cmap='Blues')
plt.xlabel('Abcissa gate (mV)')
plt.ylabel('Ordinate gate (mV)')
plt.colorbar()
plt.show()

self = ds

i = 30
j = 30
print(ds.hcgs[j, i])
print(ds.hcgs[0, 0])

pmatlab.tilefigs(10, [2, 2])


#%%


sweepdata = ['P4', -50, 80, 1.5]
stepdata = ['P1', -40, 80, 1.5]
sweepdata = ['P4', -30, 60, 1.]
stepdata = ['P1', -20, 60, 0.75]


from qtt.simulation.dotsystem import GateTransform


def make2Dscan(sweepdata, stepdata):
    ''' Convert sweepdata and stepdata to a range of values to scan '''
    vals2D = {}

    sweepvalues = np.arange(sweepdata[1], sweepdata[2], sweepdata[3])
    stepvalues = np.arange(stepdata[1], stepdata[2], stepdata[3])
    x, y = np.meshgrid(sweepvalues, stepvalues)

    vals2D[sweepdata[0]] = x
    vals2D[stepdata[0]] = y
    return vals2D


vals2D = make2Dscan(sweepdata, stepdata)


#%%

targetnames = ['det%d' % (i + 1) for i in range(4)]
sourcenames = ['P%d' % (i + 1) for i in range(4)]

Vmatrix = qtt.simulation.dotsystem.defaultVmatrix(n=4)


print(Vmatrix)


if 0:
    targetnames = ['det1', 'det4']
    sourcenames = ['P1', 'P4']
    Vmatrix = np.eye(2)
    Vmatrix[1, 0] = .2
    Vmatrix[0, 1] = .2

nn = vals2D['P1'].shape


gate_transform = GateTransform(Vmatrix, sourcenames, targetnames)

out = gate_transform.transformGateScan(vals2D, nn=nn)

self.vals2D = out

plt.figure(30)
plt.clf()
plt.imshow(out['det1'])

ds.resetMu(0)
ds.det2 = 4
# ds.isC3=3

ds.simulatehoneycomb(verbose=1, usediag=False, multiprocess=True)
paramnames = ['P1', 'P4']
plt.figure(10)
plt.clf()
plt.pcolor(vals2D[paramnames[0]], vals2D[paramnames[1]], ds.honeycomb, cmap='Blues')
plt.xlabel('Abcissa gate (mV)')
plt.ylabel('Ordinate gate (mV)')
plt.colorbar()
plt.axis('image')
plt.show()

val = 6 * ds.hcgs[:, :, 0] + 4 * ds.hcgs[:, :, 1] + 2 * ds.hcgs[:, :, 2] + 1.5 * ds.hcgs[:, :, 3]

plt.figure(11)
plt.clf()
plt.pcolor(vals2D[paramnames[0]], vals2D[paramnames[1]], val, cmap='Blues')
plt.xlabel('Abcissa gate (mV)')
plt.ylabel('Ordinate gate (mV)')
plt.colorbar()
plt.axis('image')
plt.title('distance map')
plt.show()

pmatlab.tilefigs([10, 11, 20, 30], [2, 2])

#%% Apply qutechalgorithms....
import qutechalgorithms
from qutechalgorithms import *
import qcodes as qc
import qtt.scans
import pmatlab

im = 10 * val.copy()
im = qutechalgorithms.smoothImage(im)

imc = qutechalgorithms.cleanSensingImage(im, dy='xy', sigma=0.93, removeoutliers=True)

qutechalgorithms.showIm(im, fig=130)
plt.colorbar()
plt.gca().invert_yaxis()
qutechalgorithms.showIm(imc, fig=131)
plt.colorbar()
plt.gca().invert_yaxis()


if 0:

    sigma = None
    sigma = .93
    dy = 'xy'
    verbose = 1
    removeoutliers = True
    # removeoutliers=False
    if sigma is None:
        imx = diffImage(im, dy=dy, size='same')
    else:
        imx = diffImageSmooth(im, dy=dy, sigma=sigma)
    order = 3
    vv = fitBackground(imx, smooth=True, verbose=verbose, fig=400, order=int(order), removeoutliers=removeoutliers)

    pmatlab.tilefigs([130, 131, 400])

    plt.figure(210)
    plt.clf()
    plt.imshow(imx - vv)
    plt.colorbar()

    plt.figure(210)
    plt.clf()
    plt.imshow(np.abs(imx - vv) > 1)
    plt.colorbar()

    plt.figure(210)
    plt.clf()
    plt.imshow(vv)
    plt.colorbar()

#%%

STOP
#%%


def showImage(data, im):
    pl = qc.QtPlot(qtt.scans.getDefaultParameter(data))
    return pl
pl = showImage(data, imc)
# qutechalgorithms.showIm(imc)

#%%
plt.figure(11)
plt.clf()
for ii in range(ds.ndots):
    plt.subplot(2, 2, ii + 1)
    val = ds.hcgs[:, :, ii]
#    plt.pcolor(vals2D[ paramnames[0]],vals2D[ paramnames[1]],val,cmap='Blues')
    plt.imshow(val)
    plt.gca().invert_yaxis()

    plt.xlabel('Abcissa gate (mV)')
    plt.ylabel('Ordinate gate (mV)')
    plt.title('Occupancy in dot %d' % ii)
    plt.colorbar()
plt.show()


#%%

ds.resetMu(0)
ds.det2 = 4
ds.det1 = -20
ds.det3 = -20

ds.makeH()
ds.solveH()
print(ds.OCC)

#%%


plt.figure(11)
plt.clf()
for ii in range(ds.ndots):
    plt.subplot(2, 2, ii + 1)
    val = ds.hcgs[:, :, ii]
    plt.pcolor(ds.vals2D[paramnames[0]], ds.vals2D[paramnames[1]], val, cmap='Blues')
    plt.xlabel('Abcissa gate (mV)')
    plt.ylabel('Ordinate gate (mV)')
    plt.title('Occupancy in dot %d' % ii)
    plt.colorbar()
plt.show()


#%%

def showMmatrix(self, name='det1', fig=10):
    plt.figure(fig)
    plt.clf()
    plt.imshow(getattr(self, 'M' + name), interpolation='nearest')
    plt.title('M' + name)
    plt.grid('on')


#%%

def showM(self, fig=10):
    plt.figure(fig)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.imshow(self.Mdet1, interpolation='nearest')
    plt.title('Mdet1')
    plt.grid('on')

    plt.subplot(2, 2, 2)
    plt.imshow(self.MosC1, interpolation='nearest')
    plt.title('MosC1')
    plt.grid('on')
    plt.subplot(2, 2, 3)
    plt.imshow(self.MisC1, interpolation='nearest')
    plt.title('MisC1')
    plt.grid('on')
    plt.subplot(2, 2, 4)
    plt.imshow(self.Meps11, interpolation='nearest')
    plt.title('Meps11')
    plt.grid('on')

# MosC
# MisC
# Meps

showM(self, fig=10)

#%%
if 0:
    import pydotplus as pd
    graph = pd.Dot(graph_type='graph')

    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    G = nx.DiGraph()

    G = nx.DiGraph()
    G.graph['rankdir'] = 'LR'
    G.graph['dpi'] = 120
    G.add_cycle(range(4))
    G.add_node(0, color='red', style='filled', fillcolor='pink')
    G.add_node(1, shape='square')
    G.add_node(3, style='filled', fillcolor='#00ffff')
    G.add_edge(0, 1, color='red', style='dashed')
    G.add_edge(3, 3, label='a')
    nx.draw(G)
    from nxpd import draw
    draw(G)

    for ii in range(self.ndots):
        G.add_node(ii)

    for ii in range(self.ndots):
        G.add_edge(ii, ii, label='det%d' % ii)

        # G.add_edge('1','2')

    #G.add_node(10, {'name': 'sdfs', 'label': 'hi'})

    plt.figure(10)
    plt.clf()
    plt.axis('off')
    pos = nx.layout.spring_layout(G)
    #nx.draw(G, pos=pos)
    nx.draw_networkx_nodes(G, pos=pos, node_size=4000, node_color="g")
    nx.draw_networkx_edges(G, pos=pos)

    labels = dict([(i, 'node%d' % i) for i in range(3)])
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=16)
    plt.show()


#%% Testing

def testTriple():
    tripledot = TripleDot(name='tripledot')

    tripledot.det1 = 0
    tripledot.det2 = 0
    tripledot.det3 = 0
    tripledot.eps11 = .1
    tripledot.eps12 = .15
    tripledot.eps13 = .2
    tripledot.eps21 = .1
    tripledot.eps22 = .15
    tripledot.eps23 = .2
    tripledot.eps31 = .1
    tripledot.eps32 = .15
    tripledot.eps33 = .2
    tripledot.osC1 = 27
    tripledot.osC2 = 26
    tripledot.osC3 = 28
    tripledot.isC1 = 6
    tripledot.isC2 = 6
    tripledot.isC3 = 2
    tripledot.tun1 = 0.1
    tripledot.tun2 = 0.1

    return tripledot


#%%

if __name__ == "__main__":
    import doctest
    # doctest.testmod()


#%%
tripledot = testTriple()
self = tripledot


paramnames = ['det1', 'det3']
minmax = [[-20, 80, -20, 80], [-20, -20, 80, 80]]
npointsx = 100
npointsy = 100
tripledot.makeparamvalues2D(paramnames, minmax, npointsx, npointsy)

tripledot.simulatehoneycomb(verbose=1)

plt.figure(10)
plt.clf()
plt.pcolor(tripledot.vals2D['det1'], tripledot.vals2D['det3'], tripledot.honeycomb, cmap='Blues')
plt.xlabel('Abcissa gate (mV)')
plt.ylabel('Ordinate gate (mV)')
plt.colorbar()
plt.show()

self = tripledot

#%%
if 0:
    paramnames = ['det1', 'det2', 'det3']
    bottomtop = [[-20, 100], [-20, 100], [-20, 100]]
    rangex = 60
    npointsx = 200
    npointsy = 200
    tripledot.makeparamvalues2Dx(paramnames, bottomtop, rangex, npointsx, npointsy)
    tripledot.simulatehoneycomb()

    fig = plt.figure(10)
    plt.clf()
    fig.set_size_inches(10.5, 10.5)
    plt.pcolor(tripledot.vals2D['det3'] - tripledot.vals2D['det1'], tripledot.vals2D['det2'], tripledot.honeycomb, cmap='Blues')
    plt.xlabel('Detuning L-R (mV)')
    plt.ylabel('Filling L+M+R (mV)')
    plt.colorbar()
    plt.axis('image')
    plt.show()

#%%


def makeH(self):
    #self.H = np.full((self.Nt, self.Nt), 0, dtype=float)
    self.H = np.zeros((self.Nt, self.Nt), dtype=float)
    for name in self.varnames:
        self.H = self.H + getattr(self, 'M' + name) * getattr(self, name)
    self.solved = False
    return self.H

#%timeit makeH(self)

#%%


def isdiagonal(HH):
    return not(np.any(HH - np.diag(np.diagonal(HH))))


self.tun1 = .0
self.makeH()


def solveH2(self, usediag=False):
    if usediag:
        self.energies = self.H.diagonal()
        idx = np.argsort(self.energies)
        self.energies = self.energies[idx]
        self.eigenstates[:] = 0  # =np.zeros( (self.Nt, self.Nt), dtype=float)
        for i, j in enumerate(idx):
            self.eigenstates[j, i] = 1
    else:
        self.energies, self.eigenstates = la.eigh(self.H)
    self.states = self.eigenstates
    self.stateprobs = np.square(np.absolute(self.states))
    self.stateoccs = np.dot(self.stateprobs.T, self.basis)
    self.nstates = np.sum(self.stateoccs, axis=1, dtype=float)
    self.orderstatesbyE()
    self.solved = True
    self.findcurrentoccupancy()
    return self.energies, self.eigenstates

#%timeit a,b=solveH(self)
#%timeit a2,b2=solveH2(self, usediag=True)

#%%
self.makeH()


def showH(self, fig=10):
    plt.figure(fig)
    plt.clf()
    # plt.subplot(2,2,1)
    plt.imshow(self.H, interpolation='nearest')
    plt.title('Hamiltonian')
    plt.grid('on')

showH(self)


#%%
from pmatlab import tprint

try:
    from numba import autojit, prange
except:
    def autojit(original_function):
        """ dummy autojit decorator """
        def dummy_function(*args, **kwargs):
            return original_function(*args, **kwargs)
        return dummy_function
    pass


import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

import copy


def simulate_row(i, ds, npointsy, usediag):
    dsx = copy.deepcopy(ds)
    paramnames = list(dsx.vals2D.keys())
    for j in range(npointsy):
        for name in paramnames:
            setattr(dsx, name, dsx.vals2D[name][i][j])
        dsx.makeH()
        dsx.solveH(usediag=usediag)
        dsx.hcgs[i, j] = dsx.OCC
    return dsx.hcgs[i]


def simulatehoneycomb(self, verbose=1, usediag=False, multiprocess=True):
    '''Loop over the 2D matrix of parameter values defined by makeparamvalues2D, calculate the ground state
    for each point, search for transitions and save in self.honeycomb'''
    t0 = time.time()
    paramnames = list(self.vals2D.keys())
    npointsx = np.shape(self.vals2D[paramnames[0]])[0]
    npointsy = np.shape(self.vals2D[paramnames[0]])[1]
    self.hcgs = np.empty((npointsx, npointsy, self.ndots))

    if multiprocess:
        pool = Pool(processes=4)
        aa = [(i, self, npointsy, usediag) for i in range(npointsx)]
        result = pool.starmap_async(simulate_row, aa)
        out = result.get()
        self.hcgs = np.array(out)
    else:
        for i in range(npointsx):
            if verbose:
                tprint('simulatehoneycomb: %d/%d' % (i, npointsx))

            for j in range(npointsy):
                for name in paramnames:
                    setattr(self, name, self.vals2D[name][i][j])
                self.makeH()
                self.solveH(usediag=usediag)
                self.hcgs[i, j] = self.OCC
    self.honeycomb, self.deloc = self.findtransitions(self.hcgs)

    if verbose:
        print('simulatehoneycomb: %.2f [s]' % (time.time() - t0))

    sys.stdout.flush()

paramnames = ['det1', 'det3']
minmax = [[-40, 80, -40, 80], [-40, -40, 80, 80]]
npointsx = 60
npointsy = 120
ds.makeparamvalues2D(paramnames, minmax, npointsx, npointsy)

#simulatehoneycomb(ds, verbose=1, usediag=True, multiprocess=False)
simulatehoneycomb(ds, verbose=1, usediag=True, multiprocess=True)

plt.figure(10)
plt.clf()
plt.pcolor(ds.vals2D[paramnames[0]], ds.vals2D[paramnames[1]], ds.honeycomb, cmap='Blues')
plt.xlabel('Abcissa gate (mV)')
plt.ylabel('Ordinate gate (mV)')
plt.colorbar()
plt.show()

#%%
import copy
from multiprocessing import Pool
import multiprocessing as mp
from pmatlab import tprint


def worker(args):
    name, que = name
    que.put("%d is done" % name)
    return name

npointsx = 60

usediag = False

#%%

#%%

import multiprocessing


def workerx(name, que):
    # time.sleep(2)
    que.put("%d is done" % name)
    return name


def worker(i, ds, npointsy, usediag, que):
    # time.sleep(2)
    que.put("%d is done" % i)
    return i

if __name__ == '__main__' and 1:
    pool = multiprocessing.Pool(processes=3)
    m = multiprocessing.Manager()
    q = m.Queue()
    ww = []

    args = [i + (q, ) for i in inputs]

    for jj in args:
        workers = pool.apply_async(worker, jj)
        ww.append(workers)

    print(q.qsize())
    time.sleep(0.1)
    print(q.qsize())
    print([w.ready() for w in ww])
    #print( [ w.successful() for w in ww] )


#%%
import multiprocessing
import multiprocessing as mp
import time
import copy
from pmatlab import tprint

usediag = False

from multiprocessing import Pool


def simulate_row(i, ds, npointsy, usediag, q):
    #i, ds, npointsy, usediag, q=args
    # print(i)
    dsx = copy.deepcopy(ds)
    paramnames = list(dsx.vals2D.keys())
    for j in range(npointsy):
        for name in paramnames:
            setattr(dsx, name, dsx.vals2D[name][i][j])
        dsx.makeH()
        dsx.solveH(usediag=usediag)
        dsx.hcgs[i, j] = dsx.OCC
    q.put(dsx.hcgs[i])
    return dsx.hcgs[i]


#%%
inputs = [(i, ds, npointsy, usediag) for i in range(npointsx)]

if __name__ == '__main__' and 1:

    pool = Pool(processes=4)
    m = mp.Manager()
    q = m.Queue()

    t0 = time.time()

    #args=[ (1,q), (2,q)]
    #result = pool.map_async(simulate_row, args)

    print(q.qsize())
    ww = []
    args = [i + (q, ) for i in inputs]     # create with new queue
    for a in args:
        workers1 = pool.apply_async(simulate_row, a)
        #workers1 = pool.apply_async(worker, a)
        # workers1.successful()
        ww.append(workers1)

    [w.ready() for w in ww]

    size = q.qsize()
    tprint('multiprocessing queue: %d/%d' % (size, len(inputs)))

    # monitor loop
    while True:
        size = q.qsize()
        tprint('multiprocessing queue: %d/%d' % (size, len(inputs)))

        if size == len(args):
            break
        time.sleep(0.1)

    tprint('multiprocessing queue: %d/%d' % (q.qsize(), len(inputs)))

    print('dt: %.3f [s]' % (time.time() - t0))

#outputs = q.get()


# results=[]
#job=pool.starmap_async( mymap2,  aa, callback=results.append )

# job.ready()

# vv=out.get()


#%%

from multiprocessing import Pool


def process_image(name, val):
    return name * name + val

pool = Pool(processes=4)              # process per core
pool.map(partial(process_image, val=3), range(5))  # proces data_inputs iterable with pool
