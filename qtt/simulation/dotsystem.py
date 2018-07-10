""" Simulation of a coupled dot system

"""
#%% Load packages
import numpy as np
from numpy import linalg as la
import itertools
import matplotlib.pyplot as plt
import time
import copy
from abc import abstractmethod
from functools import partial
import sys

try:
    import graphviz
except:
    pass

try:
    import multiprocessing as mp
    from multiprocessing import Pool

    _have_mp = True
except:
    _have_mp = False
    pass

import qtt.tools
from qtt import pgeometry 

#%% Helper functions


def showGraph(dot, fig=10):
    """ Show graphviz object in matplotlib window """
    dot.format = 'png'
    outfile = dot.render('dot-dummy', view=False)
    print(outfile)

    im = plt.imread(outfile)
    plt.figure(fig)
    plt.clf()
    plt.imshow(im)
    plt.axis('off')


def static_var(varname, value):
    """ Helper function to create a static variable """
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("time", 0)
def tprint(string, dt=1, output=False):
    """ Print progress of a loop every dt seconds """
    if (time.time() - tprint.time) > dt:
        print(string)
        tprint.time = time.time()
        if output:
            return True
        else:
            return
    else:
        if output:
            return False
        else:
            return


def isdiagonal(HH):
    """ Helper function """
    return not(np.any(HH - np.diag(np.diagonal(HH))))



def simulate_row(i, ds, npointsy, usediag):
    """ Helper function """
    dsx = copy.deepcopy(ds)
    paramnames = list(dsx.vals2D.keys())
    for j in range(npointsy):
        for name in paramnames:
            setattr(dsx, name, dsx.vals2D[name][i][j])
        dsx.makeH()
        dsx.solveH(usediag=usediag)
        dsx.hcgs[i, j] = dsx.OCC
    return dsx.hcgs[i]


#%%
# move into class?
def defaultVmatrix(n):
    """ Helper function 
    >>> m=defaultVmatrix(2)
    """
    Vmatrix = np.eye(n)
    vals = [1, .25, .07, .02, .001, 0, 0]
    for x in range(1, n):
        for i in range(n - x):
            Vmatrix[i, i + x] = vals[x]
            Vmatrix[i + x, i] = vals[x]

    VmatrixF = np.eye(n + 1)
    VmatrixF[0:n, 0:n] = Vmatrix
    return VmatrixF

class GateTransform:
    ''' Class to describe virtual gate transformations '''
    # FIXME: move into other submodule

    def __init__(self, Vmatrix, sourcenames, targetnames):
        self.Vmatrix = np.array(Vmatrix).astype(np.float32)
        self.sourcenames = sourcenames
        self.targetnames = targetnames

    def transformGateScan(self, vals2D, nn=None):
        '''Get a list of parameter names and [c1 c2 c3 c4] 'corner' values
        to generate dictionary self.vals2D[name] = matrix of values'''
        vals2Dout = {}

        zz = np.zeros(nn, dtype=float)
        if isinstance(vals2D, dict):
            xx = [vals2D.get(s, zz) for s in self.sourcenames]
            xx = [x.flatten() for x in xx]
            v = np.vstack(xx).astype(np.float32)
        else:
            xx = vals2D
            v = np.array(xx).astype(np.float32)

        vout = pgeometry.projectiveTransformation(self.Vmatrix, v)

        for j, n in enumerate(self.targetnames):
            vals2Dout[n] = vout[j].reshape(nn).astype(np.float)
        return vals2Dout
#%%

class BaseDotSystem():

    @abstractmethod
    def calculate_ground_state(self, gatevalues):
        pass

    # could be a static method
    def findtransitions(self, occs):
        """ Find transitions in occupancy image """
        transitions = np.full([np.shape(occs)[0], np.shape(occs)[1]], 0, dtype=float)
        delocalizations = np.full([np.shape(occs)[0], np.shape(occs)[1]], 0, dtype=float)

        if 1:
            d1 = np.sum(np.absolute(occs - np.roll(occs, 1, axis=0)), axis=2)
            d2 = np.sum(np.absolute(occs - np.roll(occs, -1, axis=0)), axis=2)
            d3 = np.sum(np.absolute(occs - np.roll(occs, 1, axis=1)), axis=2)
            d4 = np.sum(np.absolute(occs - np.roll(occs, -1, axis=1)), axis=2)
            transitions = d1 + d2 + d3 + d4
            # fix borders
            transitions[0, :] = 0
            transitions[-1, :] = 0
            transitions[:, 0] = 0
            transitions[:, -1] = 0

            occs1 = occs % 1

            for mi in range(occs.shape[2]):
                m1 = np.minimum(occs1[:, :, mi], np.abs(1 - occs1[:, :, mi]))
                delocalizations[1:-1, 1:-1] += m1[1:-1, 1:-1]

        else:
            for i in range(1, np.shape(occs)[0] - 1):
                for j in range(1, np.shape(occs)[1] - 1):
                    diff1 = np.sum(np.absolute(occs[i, j] - occs[i - 1, j - 1]))
                    diff2 = np.sum(np.absolute(occs[i, j] - occs[i - 1, j + 1]))
                    diff3 = np.sum(np.absolute(occs[i, j] - occs[i + 1, j - 1]))
                    diff4 = np.sum(np.absolute(occs[i, j] - occs[i + 1, j + 1]))
                    transitions[i, j] = diff1 + diff2 + diff3 + diff4
                    delocalizations[i, j] = min(occs[i, j, 0] % 1, abs(1 - occs[i, j, 0] % 1)) + min(occs[i, j, 0] % 1, abs(1 - occs[i, j, 0] % 1)) + min(occs[i, j, 0] % 1, abs(1 - occs[i, j, 0] % 1))
        return transitions, delocalizations

    def orderstatesbyN(self):
        """ Order the calculated states by occupation """
        sortinds = np.argsort(self.nstates)
        self.energies = self.energies[sortinds]
        self.states = self.states[sortinds]
        self.stateprobs = self.stateprobs[sortinds]
        self.stateoccs = self.stateoccs[sortinds]
        self.nstates = self.nstates[sortinds]

    def orderstatesbyE(self):
        """ Order the calculated states by energy """
        sortinds = np.argsort(self.energies)
        self.energies = self.energies[sortinds]
        self.states = self.states[sortinds]
        self.stateprobs = self.stateprobs[sortinds]
        self.stateoccs = self.stateoccs[sortinds]
        self.nstates = self.nstates[sortinds]

    def showstates(self, n):
        """ List states of the system with energies """
        print('\nEnergies/states list for %s:' % self.name)
        print('-----------------------------------')
        for i in range(n):
            print(str(i) + '       - energy: ' + str(np.around(self.energies[i], decimals=2)) + ' ,      state: ' + str(np.around(self.stateoccs[i], decimals=2)) + ' ,      Ne = ' + str(self.nstates[i]))
        print(' ')

    def makebasis(self, ndots, maxelectrons=2):
        ''' Define a basis of occupancy states with a specified number of dots and max occupancy '''
        self.maxelectrons = maxelectrons
        self.ndots = ndots

        basis = list(itertools.product(range(maxelectrons + 1), repeat=ndots))
        basis = np.asarray(sorted(basis, key=lambda x: sum(x)))
        self.basis = np.ndarray.astype(basis, int)
        self.nbasis = np.sum(self.basis, axis=1)
        self.Nt = len(self.nbasis)
        self.H = np.zeros((self.Nt, self.Nt), dtype=float)
        self.eigenstates = np.zeros((self.Nt, self.Nt), dtype=float)        
        
        
#%%


class DotSystem(BaseDotSystem):
    ''' Class to simulate a system of interacting quantum dots '''

    def __init__(self, name='dotsystem', ndots=3, **kwargs):
        self.name = name
        self.ndots = ndots
        self.temperature = 0



    def makevars(self):
        for name in self.varnames:
            setattr(self, name, 0)
            # also define that these are float32 numbers!
            setattr(self, 'M' + name, np.full((self.Nt, self.Nt), 0, dtype=int))

    def makevarMs(self, ring=False):
        ''' Create matrices for the interactions 
        
        Args:
            ring (bool): is the dot array in a ring configuration? (e.g. 2x2)
        '''
        m = np.zeros((self.ndots), dtype=int)

        def mkb(i, j):
            mx = m.copy()
            mx[i] = 1
            mx[j] = -1
            return mx

        for i in range(self.Nt):
            for j in range(self.Nt):
                if i == j:
                    for dot in range(1, self.ndots + 1):
                        n = self.basis[i, dot - 1]
                        getattr(self, 'Mdet%i' % dot)[i, i] = [0, -1, -2, -3][n] # chemical potential
                        getattr(self, 'MosC%i' % dot)[i, i] = [0, 0, 1, 3][n] # on site charging energy
                        n2 = self.basis[i, dot % self.ndots]
                        n3 = self.basis[i,(dot + 1) % self.ndots]
                        getattr(self, 'MisC%i' % dot)[i, i] = n*n2 # nearest-neighbour charging energy
                        if hasattr(self, 'isC%i' % (self.ndots + 1)):
                            if i % 2:
                                getattr(self, 'MisC%i' % (self.ndots + 2))[i, i] = n*n3 # next-nearest-neighbour charging energy
                            else:
                                getattr(self, 'MisC%i' % (self.ndots + 1))[i, i] = n*n3 # next-nearest-neighbour charging energy
                        for orb in range(1, n + 1):                                                            
                            var = 'Meps%i%i' % (dot, orb)
                            if hasattr(self, var):
                                getattr(self, var)[i, i] = 1 # orbital energy
                else:
                    statediff = self.basis[i, :] - self.basis[j, :]

                    for p in range(self.ndots - 1):   
                        pn = p + 1
                        if (statediff == mkb(p,pn)).all() or (statediff == mkb(pn,p)).all():
                            getattr(self, 'Mtun%i' % pn)[i, j] = -1 # tunneling term
                        elif ring and ((statediff == mkb(0,self.ndots-1)).all() or (statediff == mkb(self.ndots-1,0)).all()):
                            getattr(self, 'Mtun%i' % self.ndots)[i, j] = -1 # tunneling term at boundary
                        pass

        self.initSparse()

    def initSparse(self):
        ''' Create sparse structures '''
        self.H = np.zeros((self.Nt, self.Nt), dtype=float)
        # self.sH = smtype(self.H)

        for name in self.varnames:
            A = getattr(self, 'M' + name)
            if 0:
                sA = smtype(A)
                setattr(self, 'sM' + name, sA)
                # ri = np.repeat(np.arange(sA.shape[0]),np.diff(sA.indptr))
                setattr(self, 'srM' + name, ri)
                setattr(self, 'scM' + name, sA.indices)
                arr = np.array([ri, sA.indices])
                ind = np.ravel_multi_index(arr, self.H.shape)
            ind = A.flatten().nonzero()[0]
            setattr(self, 'indM' + name, ind)
            setattr(self, 'sparseM' + name, A.flat[ind])

    def makeH(self):
        ''' Create a new Hamiltonian '''
        self.H.fill(0)
        for name in self.varnames:
            val = getattr(self, name)
            if not val == 0:
                self.H += getattr(self, 'M' + name) * val
        self.solved = False
        return self.H

    def makeHsparse(self, verbose=0):
        ''' Create a new Hamiltonian '''
        self.H.fill(0)
        for name in self.varnames:
            if verbose:
                print('set %s: %f' % (name, getattr(self, name)))
            val = float(getattr(self, name))
            if not val == 0:
                a = getattr(self, 'sparseM' + name)
                ind = getattr(self, 'indM' + name)
                self.H.flat[ind] += a * val
                # self.H[ri, ci] +=a.data * val
        self.solved = False
        return self.H

    def solveH(self, usediag=False):
        ''' Solve the system '''
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

    #%% Helper functions
    def getall(self, param):
        ''' Return all stored values for a particular parameter
        
        Args:
            param (str): det, osC, isC or tun
        
        Returns:
            vals (list): values corresponding to the parameter that was queried
        '''
        numvars = 0
        for var in self.varnames:
            if var.startswith(param): numvars += 1
        vals = [getattr(self, param + str(i + 1)) for i in range(numvars)]
        return vals
    
    def setall(self, param, vals):
        ''' Sets all values for a particular parameter
        
        Args:
            param (str): det, osC, isC or tun
            vals (list): values corresponding to the parameter to be set
        '''
        numvars = 0
        for var in self.varnames:
            if var.startswith(param): numvars += 1
        if len(vals) != numvars: raise(Exception('Need same amount of values as '+param))
        for i in range(numvars):
            setattr(self, param + str(i + 1), vals[i])
    
    def simulate_honeycomb(self, paramvalues2D, verbose=1, usediag=False, multiprocess=True):
        self.vals2D = {}
        for i in range(paramvalues2D.shape[0]):
            nm = 'det%d' % (i + 1)
            self.vals2D[nm] = paramvalues2D[i, :, :]
        return self.simulatehoneycomb(verbose=verbose, usediag=usediag, multiprocess=multiprocess)

    def simulatehoneycomb(self, verbose=1, usediag=False, multiprocess=True):
        '''Loop over the 2D matrix of parameter values defined by makeparamvalues2D, calculate the ground state
        for each point, search for transitions and save in self.honeycomb'''
        t0 = time.time()
        paramnames = list(self.vals2D.keys())
        initparamvalues = self.getall('det')
        npointsx = np.shape(self.vals2D[paramnames[0]])[0]
        npointsy = np.shape(self.vals2D[paramnames[0]])[1]
        self.hcgs = np.empty((npointsx, npointsy, self.ndots))

        self.initSparse()

        if multiprocess and _have_mp:
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
                    self.makeHsparse()
                    self.solveH(usediag=usediag)
                    self.hcgs[i, j] = self.OCC
        self.honeycomb, self.deloc = self.findtransitions(self.hcgs)
        self.setall('det', initparamvalues)

        if verbose:
            print('simulatehoneycomb: %.2f [s]' % (time.time() - t0))

        sys.stdout.flush()

    def simulatehoneycomb_original(self, verbose=1, usediag=False):
        '''Loop over the 2D matrix of parameter values defined by makeparamvalues2D, calculate the ground state
        for each point, search for transitions and save in self.honeycomb'''
        t0 = time.time()
        paramnames = list(self.vals2D.keys())
        npointsx = np.shape(self.vals2D[paramnames[0]])[0]
        npointsy = np.shape(self.vals2D[paramnames[0]])[1]
        self.hcgs = np.empty((npointsx, npointsy, self.ndots))
        for i in range(npointsx):
            if verbose:
                tprint('simulatehoneycomb: %d/%d' % (i, npointsx))
            for j in range(npointsy):
                for name in paramnames:
                    exec('self.' + name + ' = self.vals2D[name][' + str(i) + '][' + str(j) + ']')
				
                self.makeH()
                self.solveH(usediag=usediag)
                self.hcgs[i, j] = self.OCC
        self.honeycomb, self.deloc = self.findtransitions(self.hcgs)

        if verbose:
            print('simulatehoneycomb: %.1f [s]' % (time.time() - t0))

    def calculate_energies(self, gatevalues):
        for i, val in enumerate(gatevalues):
            setattr(self, 'det%d' % (i + 1), val)
        self.makeHsparse()
        self.solveH()
        return self.energies

    def calculate_ground_state(self, gatevalues):
        """ Calculate the ground state of the dot system, given a set of gate values. Returns a state array. """
        energies = self.calculate_energies(gatevalues)
        return self.stateoccs[0]  # self.basis[np.argmin(energies)]

    def findcurrentoccupancy(self, exact=True):
        if self.solved == True:
            self.orderstatesbyE()
            if exact:
                # almost exact...
                idx = self.energies == self.energies[0]
                self.OCC = np.around(np.mean(self.stateoccs[idx], axis=0), decimals=2)
            else:
                # first order approximation
                self.OCC = np.around(self.stateoccs[0], decimals=2)
        else:
            self.solveH()
        return self.OCC

    def makeparamvalues1D(self, paramnames, startend, npoints):
        '''Get a list of parameter names and [start end] values
        to generate dictionary self.vals1D[name] = vector of values'''
        self.vals1D = {}
        for i in range(len(paramnames)):
            name = paramnames[i]
            self.vals1D[name] = np.linspace(startend[i][0], startend[i][1], num=npoints)

    def makeparamvalues2D(self, paramnames, cornervals, npointsx, npointsy):
        '''Get a list of parameter names and [c1 c2 c3 c4] 'corner' values
        to generate dictionary self.vals2D[name] = matrix of values'''
        self.vals2D = {}
        for i in range(len(paramnames)):
            name = paramnames[i]
            if len(cornervals[i]) == 2:
                cornervals[i] = np.append(cornervals[i], cornervals[i])
                bottomrow = np.linspace(cornervals[i][0], cornervals[i][1], num=npointsx)
                toprow = np.linspace(cornervals[i][2], cornervals[i][3], num=npointsx)
            bottomrow = np.linspace(cornervals[i][0], cornervals[i][2], num=npointsx)
            toprow = np.linspace(cornervals[i][1], cornervals[i][3], num=npointsx)
            self.vals2D[name] = np.array([np.linspace(i, j, num=npointsy) for i, j in zip(bottomrow, toprow)])

    def makeparamvalues2Dx(self, paramnames, bottomtop, rangex, npointsx, npointsy):
        '''Get a list of parameter names and [bottom top] values
        to generate dictionary self.vals2D[name] = matrix of values
        where the x-direction detunes dots 1,3'''
        self.vals2D = {}
        for i in range(len(paramnames)):
            name = paramnames[i]
            self.vals2D[name] = np.transpose(np.array([np.linspace(bottomtop[i][0], bottomtop[i][1], num=npointsy) for j in range(npointsy)]))
            if name == 'det1':
                self.vals2D[name] = self.vals2D[name] + np.array([np.linspace(rangex / 2, -rangex / 2, num=npointsx) for i in range(npointsx)])
            elif name == 'det3':
                self.vals2D[name] = self.vals2D[name] + np.array([np.linspace(-rangex / 2, rangex / 2, num=npointsx) for i in range(npointsx)])
            else:
                pass

    def resetMu(self, value=0):
        ''' Reset chemical potential '''
        for ii in range(self.ndots):
            setattr(self, 'det%d' % (ii + 1), value)

    def showMmatrix(self, name='det1', fig=10):
        plt.figure(fig)
        plt.clf()
        plt.imshow(getattr(self, 'M' + name), interpolation='nearest')
        plt.title('M' + name)
        plt.grid('on')

    def showvars(self):
        print('\nVariable list for %s:' % self.name)
        print('----------------------------')
        for name in self.varnames:
            print(name + ' = ' + str(eval('self.' + name)))
        print(' ')

    def getHn(self, numberofelectrons):
        inds = np.where(self.nbasis == numberofelectrons)[0]
        return self.H[inds[0]:inds[-1] + 1, inds[0]:inds[-1] + 1]

    def visualize(self, fig=1):
        ''' Create a graphical representation of the system (needs graphviz) '''
        if self.ndots is None:
            print('no number of dots defined...')
            return
        dot = graphviz.Digraph(name=self.name)

        for ii in range(self.ndots):
            # dot.node('%d'% ii)
            dot.node(str(ii), label='dot %d' % ii)
            dot.edge(str(ii), str(ii), label='det%d' % ii)

        showGraph(dot, fig=fig)


@qtt.tools.deprecated
def setDotSystem(ds, gate_transform, gv):
    """ Set dot system values using gate transform """
    tv = gate_transform.transformGateScan(gv)
    for k, val in tv.items():
        setattr(ds, k, val)


@qtt.tools.deprecated
def defaultDotValues(ds):
    for ii in range(ds.ndots):
        setattr(ds, 'osC%d' % (ii + 1), 55)
    for ii in range(ds.ndots - 1):
        setattr(ds, 'isC%d' % (ii + 1), 3)


#%% Example dot systems

class OneDot(DotSystem):

    def __init__(self, name='doubledot', maxelectrons=3):
        """ Simulation of a single quantum dot """
        super().__init__(name=name, ndots=1)
        self.makebasis(ndots=self.ndots, maxelectrons=maxelectrons)
        self.varnames = ['det1', 'osC1', 'isC1']
        self.varnames += itertools.chain(* [['eps%d%d' % (d + 1, orb + 1) for d in range(self.ndots)] for orb in range(0, self.maxelectrons)])
        self.makevars()
        self.makevarMs()
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()


class DoubleDot(DotSystem):

    def __init__(self, name='doubledot'):
        """ Simulation of double-dot system
        
        See: DotSytem
        """

        super().__init__(name=name, ndots=2)
        self.makebasis(ndots=self.ndots, maxelectrons=3)
        self.varnames = ['det1', 'det2',
                         'osC1', 'osC2', 'isC1', 'isC2', 'tun1', 'tun2']
        self.varnames += itertools.chain(* [['eps%d%d' % (d + 1, orb + 1) for d in range(self.ndots)] for orb in range(0, self.maxelectrons)])
        self.makevars()
        self.makevarMs()
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()


class TripleDot(DotSystem):

    def __init__(self, name='tripledot', maxelectrons=3):
        """ Simulation of triple-dot system """
        super().__init__(name=name, ndots=3)
        self.makebasis(ndots=self.ndots, maxelectrons=maxelectrons)
        self.varnames = ['det1', 'det2', 'det3',
                         'eps11', 'eps12', 'eps13', 'eps21', 'eps22', 'eps23', 'eps31', 'eps32', 'eps33',
                         'osC1', 'osC2', 'osC3',
                         'isC1', 'isC2', 'isC3',
                         'tun1', 'tun2']
        self.makevars()
        self.makevarMs()
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()


class FourDot(DotSystem):

    def __init__(self, name='fourdot', use_tunneling=True, use_orbits=False, **kwargs):
        """ Simulation of 4-dot system """
        super().__init__(name=name, ndots=4, **kwargs)

        self.use_tunneling = use_tunneling
        self.use_orbits = use_orbits
        self.makebasis(ndots=self.ndots, maxelectrons=2)
        self.varnames = ['det%d' % (i + 1) for i in range(self.ndots)]
        self.varnames += ['osC%d' % (i + 1) for i in range(self.ndots)]
        self.varnames += ['isC%d' % (i + 1) for i in range(self.ndots)]
        if self.use_tunneling:
            self.varnames += ['tun%d' % (i + 1) for i in range(self.ndots)]
        if self.use_orbits:
            self.varnames += itertools.chain(* [['eps%d%d' % (d + 1, orb + 1) for d in range(self.ndots)] for orb in range(0, self.maxelectrons)])
        self.makevars()
        self.makevarMs()
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()

class TwoXTwo(DotSystem):    
    def __init__(self, name='2x2'):
        super().__init__(name=name, ndots=4)									
        self.makebasis(ndots=self.ndots, maxelectrons=2)
        self.varnames = ['det1','det2','det3','det4',
                         'osC1','osC2','osC3','osC4',
                         'isC1','isC2','isC3','isC4','isC5','isC6',
                         'tun1','tun2','tun3','tun4']
        self.makevars()
        self.makevarMs(ring=True)
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()

def test_twoxtwo():
    m=TwoXTwo()
    m.calculate_energies(np.random.rand(m.ndots))
    m.solveH()
    if __name__=='__main__':
        m.showstates(81)

if __name__=='__main__':
    test_twoxtwo()
