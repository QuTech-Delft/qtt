# -*- coding: utf-8 -*-
""" Classical Quantum Dot Simulator

This class aims to be a generic classical simulator for calculating energy levels and occupancy of quantum dots.
Note: interaction between the dots is treated completely classically (no tunnel coupling) resulting in faster simulations.

User defines:
    - number of dots: ndots
    - number of gates: ngates
    - maximum occupancy per dot: maxelectrons
    - capacitances and cross-capacitances between dots and gates: alphas
    - chemical potential (at zero gate voltage): mu0
    - addition energy: Eadd
    - coulomb repulsion: W

@author: lgnjanssen
"""

import itertools
import os

import numpy as np
import operator as op
import functools

import time

try:
    import multiprocessing
    import multiprocessing as mp
    from multiprocessing import Pool
    _have_mp = True
except:
    _have_mp = False
    pass


from qtt.simulation.dotsystem import tprint, BaseDotSystem


def ncr(n, r):
    """ Calculating number of possible combinations: nCr"""
    r = min(r, n - r)
    if r == 0:
        return 1
    numer = functools.reduce(op.mul, range(n, n - r, -1))
    denom = functools.reduce(op.mul, range(1, r + 1))
    return numer // denom


class ClassicalDotSystem(BaseDotSystem):

    def __init__(self, name='dotsystem', ndots=3, ngates=3, maxelectrons=3, **kwargs):
        self.name = name
        
        self.makebasis(ndots=ndots, maxelectrons=maxelectrons)
        self.ngates = ngates
        
        # initialize characterizing dot variables
        self.varnames = ['mu0', 'Eadd', 'W', 'alpha']
        self.mu0 = np.zeros((ndots,))  # chemical potential at zero gate voltage
        self.Eadd = np.zeros((ndots,))  # addition energy
        self.W = np.zeros((ncr(ndots, 2),))  # coulomb repulsion
        self.alpha = np.zeros((ndots, ngates))  # virtual gate matrix, mapping gates to chemical potentials

        self.makebasis_extra()

    def makebasis_extra(self):
        """ Define a basis of occupancy states """
        # make addition energy basis
        self.add_basis = self.basis.copy()
        self.coulomb_energy = np.zeros((self.basis.shape[0], self.W.size))
        for i in range(self.Nt):
            self.add_basis[i] = (1 / 2 * np.multiply(self.basis[i], self.basis[i] + 1))
            self.coulomb_energy[i] = [np.dot(*v) for v in itertools.combinations(self.basis[i], 2)]

    def calculate_energies(self, gatevalues):
        """ Calculate the energies of all dot states, given a set of gate values. Returns array of energies. """
        energies = np.zeros((self.Nt,))
        tmp1 = -(self.mu0 + np.dot(self.alpha, gatevalues))
        if 0:
            for i in range(self.Nt):
                energy = 0
                energy += np.dot(tmp1, self.basis[i])
                energy += np.dot(self.coulomb_energy[i], self.W)
                energy += np.dot(self.add_basis[i], self.Eadd)
                energies[i] = energy
        else:
            energies += self.basis.dot(tmp1) # chemical potentiol times number of electrons
            energies += self.coulomb_energy.dot(self.W) # coulomb repulsion
            energies += self.add_basis.dot(self.Eadd) # addition energy
        self.energies = energies

        idx = np.argsort(self.energies)
        self.energies = self.energies[idx]
        self.eigenstates[:] = 0  # =np.zeros( (self.Nt, self.Nt), dtype=float)
        for i, j in enumerate(idx):
            self.eigenstates[j, i] = 1
        return energies

    def calculate_ground_state(self, gatevalues):
        """ Calculate the ground state of the dot system, given a set of gate values. Returns a state array. """
        energies = self.calculate_energies(gatevalues)
        return self.basis[np.argmin(energies)]

    def simulate_honeycomb(self, paramvalues2D, verbose=1, usediag=False, multiprocess=True):
        """ Simulating a honeycomb by looping over a 2D array of parameter values (paramvalues2D),
         resulting honeycomb is stored in self.honeycomb """
        t0 = time.time()

        nparams = np.shape(paramvalues2D)[0]
        npointsx = np.shape(paramvalues2D)[1]
        npointsy = np.shape(paramvalues2D)[2]

        if nparams != self.ngates:
            print('simulate_honeycomb: number of parameters (%d) does not equal number of gates (%d)...' % (nparams, self.ngates))
            return

        self.hcgs = np.empty((npointsx, npointsy, self.ndots))

        if multiprocess and _have_mp:
            pool = Pool(processes=os.cpu_count())
            param_iter = [(paramvalues2D[:, i, j]) for i in range(npointsx) for j in range(npointsy)]
            result = pool.map(self.calculate_ground_state, param_iter)
            self.hcgs = np.reshape(np.array(result), (npointsx, npointsy, self.ndots))
            pool.terminate()
        else:
            for i in range(npointsx):
                if verbose:
                    tprint('simulatehoneycomb: %d/%d' % (i, npointsx))
                for j in range(npointsy):
                    self.hcgs[i, j] = self.calculate_ground_state(paramvalues2D[:, i, j])
        self.honeycomb, self.deloc = self.findtransitions(self.hcgs)

        if verbose:
            print('simulatehoneycomb: %.2f [s]' % (time.time() - t0))

    def solve(self):
        self.states = self.eigenstates

        self.stateprobs = np.square(np.absolute(self.states))
        self.stateoccs = np.dot(self.stateprobs.T, self.basis)
        self.nstates = np.sum(self.stateoccs, axis=1, dtype=float)
        self.orderstatesbyE()
        self.findcurrentoccupancy()
        
    def findcurrentoccupancy(self, exact=True):
        if exact:
            # almost exact...
            idx = self.energies == self.energies[0]
            self.OCC = np.around(np.mean(self.stateoccs[idx], axis=0), decimals=2)
        else:
            # first order approximation
            self.OCC = np.around(self.stateoccs[0], decimals=2)
        return self.OCC

class TripleDot(ClassicalDotSystem):

    def __init__(self, name='tripledot', **kwargs):
        super().__init__(name=name, ndots=3, ngates=3, **kwargs)

        self.makebasis(ndots=3)

        vardict = {}

        vardict["mu0_values"] = np.array([-27.0, -20.0, -25.0])  # chemical potential at zero gate voltage
        vardict["Eadd_values"] = np.array([54.0, 52.8, 54.0])  # addition energy
        vardict["W_values"] = 3 * np.array([6.0, 1.0, 5.0])  # coulomb repulsion (!order is important: (1,2), (1,3), (2,3)) (lexicographic ordering)
        vardict["alpha_values"] = np.array([[1.0, 0.25, 0.1],
                                            [0.25, 1.0, 0.25],
                                            [0.1, 0.25, 1.0]])

        for name in self.varnames:
            setattr(self, name, vardict[name + '_values'])

class MultiDot(ClassicalDotSystem):

    def __init__(self, name='multidot', ndots=6, maxelectrons = 3,  **kwargs):
        super().__init__(name=name, ndots=ndots, ngates=ndots, maxelectrons=maxelectrons, **kwargs)

        self.makebasis(ndots=ndots, maxelectrons=maxelectrons)

        vardict = {}


        vardict["mu0_values"] = 10*np.sin(np.arange(ndots))  # chemical potential at zero gate voltage
        vardict["Eadd_values"] = 50+np.sin(2+np.arange(ndots))  # addition energy
        
        
        dotpairs = list(itertools.combinations(range(ndots), 2))
        
        
        coulomb_repulsion = [np.Inf, 18.0, 3.0, 0.05,]+[0]*ndots
        W=np.array([ coulomb_repulsion[p[1]-p[0]] for p in dotpairs] )
        vardict["W_values"] = W # coulomb repulsion (!order is important: (1,2), (1,3), (2,3)) (lexicographic ordering)
        vardict["alpha_values"] = np.eye(self.ndots)

        for name in self.varnames:
            setattr(self, name, vardict[name + '_values'])


class DoubleDot(ClassicalDotSystem):

    def __init__(self, name='doubledot', **kwargs):
        super().__init__(name=name, ndots=2, ngates=2, **kwargs)

        self.makebasis(ndots=2)

        vardict = {}

        vardict["mu0_values"] = np.array([120.0, 100.0])  # chemical potential at zero gate voltage
        vardict["Eadd_values"] = np.array([54.0, 52.8])  # addition energy
        vardict["W_values"] = np.array([6.0])  # coulomb repulsion (!order is important: (1,2), (1,3), (2,3)) (lexicographic ordering)
        vardict["alpha_values"] = np.array([[1.0, 0.25], [0.25, 1.0]])

        for name in self.varnames:
            setattr(self, name, vardict[name + '_values'])


class SquareDot(ClassicalDotSystem):

    def __init__(self, name='squaredot', **kwargs):
        super().__init__(name=name, ndots=4, ngates=4, **kwargs)

        self.makebasis(ndots=4)

        vardict = {}

        vardict["mu0_values"] = np.array([-30.0, -30.0, -30.0, -30.0])
        vardict["Eadd_values"] = np.array([50.0, 50.0, 50.0, 50.0])
        vardict["W_values"] = np.array([5.0, 2.0, 5.0, 5.0, 2.0, 5.0])  # order:(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        vardict["alpha_values"] = np.array([[1.0, 0.25, 0.1, 0.25],
                                            [0.25, 1.0, 0.25, 0.1],
                                            [0.1, 0.25, 1.0, 0.25],
                                            [0.25, 0.1, 0.25, 1.0]])

        for name in self.varnames:
            setattr(self, name, vardict[name + '_values'])


def test_dotsystem():
    m=MultiDot('multidot', 4, maxelectrons=3)
    m.calculate_energies(np.random.rand(m.ndots))
    m.solve()
    self=m
    if __name__=='__main__':
        m.showstates(8)
    
    

if 0:
    test_dotsystem()
    m=MultiDot('multidot', 4, maxelectrons=3)
    m.calculate_energies(np.random.rand(m.ndots))
    m.solve()
    self=m
    