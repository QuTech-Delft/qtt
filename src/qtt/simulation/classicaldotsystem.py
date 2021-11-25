# -*- coding: utf-8 -*-
""" Classical Quantum Dot Simulator

@author: lgnjanssen / eendebakpt / hensgens
"""

import itertools
import os
import numpy as np
import operator as op
import functools
import time
import logging

try:
    import multiprocessing as mp
    from multiprocessing import Pool
    _have_mp = True
except ImportError:
    _have_mp = False
    pass

from qtt.simulation.dotsystem import tprint, BaseDotSystem


def ncr(n, r):
    """ Calculating number of possible combinations: n choose r """
    r = min(r, n - r)
    if r == 0:
        return 1
    numerator = functools.reduce(op.mul, range(n, n - r, -1))
    denominator = functools.reduce(op.mul, range(1, r + 1))
    return numerator // denominator


class ClassicalDotSystem(BaseDotSystem):

    """ Classical Quantum Dot Simulator

    This class aims to be a generic classical simulator for calculating energy levels and occupancy of quantum dots.
    Note: interaction between the dots is treated completely classically (no tunnel coupling) resulting in faster simulations.

    Args:
        name (str): name of the system
        ndots (int): number of dots
        ngates (int): number of voltage gates
        maxelectrons (int): maximum occupancy per dot

    The user should set variables on the object :
        - capacitances and cross-capacitances between dots and gates: alphas
        - chemical potential (at zero gate voltage): mu0
        - addition energy: Eadd
        - coulomb repulsion: W

    """
    def __init__(self, name='classicaldotsystem', ndots=3, ngates=3, maxelectrons=3, **kwargs):
        super().__init__(name=name, ndots=ndots, maxelectrons=maxelectrons)

        logging.info('ClassicalDotSystem: max number of electrons %d' % maxelectrons)
        self.ngates = ngates

        # initialize characterizing dot variables
        self.varnames = ['mu0', 'Eadd', 'W', 'alpha']
        self.mu0 = np.zeros((ndots,))  # chemical potential at zero gate voltage
        self.Eadd = np.zeros((ndots,))  # addition energy
        self.W = np.zeros((ncr(ndots, 2),))  # coulomb repulsion
        self.alpha = np.zeros((ndots, ngates))  # virtual gate matrix, mapping gates to chemical potentials

        self._makebasis_extra()

    def _makebasis_extra(self):
        """ Define a basis of occupancy states

        These addition structures are used for efficient construction of the Hamiltonian
        """
        # make addition energy basis
        self._add_basis = self.basis.copy()
        self._coulomb_energy = np.zeros((self.basis.shape[0], self.W.size))
        for i in range(self.number_of_basis_states):
            self._add_basis[i] = (1 / 2 * np.multiply(self.basis[i], self.basis[i] + 1))
            self._coulomb_energy[i] = [np.dot(*v) for v in itertools.combinations(self.basis[i], 2)]

    def calculate_energies(self, gatevalues):
        """ Calculate the energies of all dot states, given a set of gate values. Returns array of energies. """
        energies = np.zeros((self.number_of_basis_states,))
        tmp1 = -(self.mu0 + np.dot(self.alpha, gatevalues))
        energies += self.basis.dot(tmp1)  # chemical potential times number of electrons
        energies += self._coulomb_energy.dot(self.W)  # coulomb repulsion
        energies += self._add_basis.dot(self.Eadd)  # addition energy
        self.energies = energies

        idx = np.argsort(self.energies)
        self.energies = self.energies[idx]
        self.eigenstates[:] = 0
        for i, j in enumerate(idx):
            self.eigenstates[j, i] = 1
        return energies

    def calculate_ground_state(self, gatevalues):
        """ Calculate the ground state of the dot system, given a set of gate values. Returns a state array. """
        energies = self.calculate_energies(gatevalues)
        return self.basis[np.argmin(energies)]

    def simulate_honeycomb(self, paramvalues2D, verbose=1, usediag=False, multiprocess=True):
        """ Simulating a honeycomb by looping over a 2D array of parameter values (paramvalues2D),
         resulting honeycomb is stored in self.honeycomb

         Args:
             paramvalues2D (array): shape nparams x nx x ny
             verbose (int)
             usediag (bool)
             multiprocess(bool)

         """
        t0 = time.time()

        nparams = np.shape(paramvalues2D)[0]
        npointsx = np.shape(paramvalues2D)[1]
        if len(paramvalues2D.shape) == 3:
            npointsy = np.shape(paramvalues2D)[2]
        else:
            npointsy = 1

        if nparams != self.ngates:
            print('simulate_honeycomb: number of parameters (%d) does not equal number of gates (%d)...' %
                  (nparams, self.ngates))
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
        """ Find electron occupancy

        Args:
            exact (bool): If True then average over all ground states
        """
        if exact:
            # almost exact...
            idx = self.energies == self.energies[0]
            self.OCC = np.around(np.mean(self.stateoccs[idx], axis=0), decimals=2)
        else:
            # first order approximation
            self.OCC = np.around(self.stateoccs[0], decimals=2)
        return self.OCC


class TripleDot(ClassicalDotSystem):

    def __init__(self, name='tripledot', maxelectrons=2, **kwargs):
        """ Classical simulation of triple dot """
        super().__init__(name=name, ndots=3, ngates=3, maxelectrons=maxelectrons, **kwargs)

        logging.info('TripleDot: maxelectrons %d' % maxelectrons)

        vardict = {}

        vardict["mu0_values"] = np.array([-27.0, -20.0, -25.0])  # chemical potential at zero gate voltage
        vardict["Eadd_values"] = np.array([54.0, 52.8, 54.0])  # addition energy
        # coulomb repulsion (!order is important: (1,2), (1,3), (2,3)) (lexicographic ordering)
        vardict["W_values"] = 3 * np.array([6.0, 1.0, 5.0])
        vardict["alpha_values"] = np.array([[1.0, 0.25, 0.1],
                                            [0.25, 1.0, 0.25],
                                            [0.1, 0.25, 1.0]])

        for name in self.varnames:
            setattr(self, name, vardict[name + '_values'])


class MultiDot(ClassicalDotSystem):

    def __init__(self, name='multidot', ndots=6, maxelectrons=3, **kwargs):
        """ Classical simulation of multi dot """
        super().__init__(name=name, ndots=ndots, ngates=ndots, maxelectrons=maxelectrons, **kwargs)

        vardict = {}

        vardict["mu0_values"] = 10 * np.sin(np.arange(ndots))  # chemical potential at zero gate voltage
        vardict["Eadd_values"] = 50 + np.sin(2 + np.arange(ndots))  # addition energy

        dotpairs = list(itertools.combinations(range(ndots), 2))

        coulomb_repulsion = [np.Inf, 18.0, 3.0, 0.05, ] + [0] * ndots
        W = np.array([coulomb_repulsion[p[1] - p[0]] for p in dotpairs])
        # coulomb repulsion (!order is important: (1,2), (1,3), (2,3)) (lexicographic ordering)
        vardict["W_values"] = W
        vardict["alpha_values"] = np.eye(self.ndots)

        for name in self.varnames:
            setattr(self, name, vardict[name + '_values'])


class DoubleDot(ClassicalDotSystem):
    """ Classical simulation of double dot """

    def __init__(self, name='doubledot', maxelectrons=2, **kwargs):
        super().__init__(name=name, ndots=2, ngates=2, maxelectrons=maxelectrons, **kwargs)

        vardict = {}

        vardict["mu0_values"] = np.array([120.0, 100.0])  # chemical potential at zero gate voltage
        vardict["Eadd_values"] = np.array([54.0, 52.8])  # addition energy
        # coulomb repulsion (!order is important: (1,2), (1,3), (2,3)) (lexicographic ordering)
        vardict["W_values"] = np.array([6.0])
        vardict["alpha_values"] = np.array([[1.0, 0.25], [0.25, 1.0]])

        for name in self.varnames:
            setattr(self, name, vardict[name + '_values'])


class SquareDot(ClassicalDotSystem):

    def __init__(self, name='squaredot', maxelectrons=2, **kwargs):
        """ Classical simulation of a 2x2 dot configuration """
        super().__init__(name=name, ndots=4, ngates=4, maxelectrons=maxelectrons, **kwargs)

        vardict = {}

        vardict["mu0_values"] = np.array([-30.0, -30.0, -30.0, -30.0])
        vardict["Eadd_values"] = np.array([50.0, 50.0, 50.0, 50.0])
        # order:(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        vardict["W_values"] = np.array([5.0, 2.0, 5.0, 5.0, 2.0, 5.0])
        vardict["alpha_values"] = np.array([[1.0, 0.25, 0.1, 0.25],
                                            [0.25, 1.0, 0.25, 0.1],
                                            [0.1, 0.25, 1.0, 0.25],
                                            [0.25, 0.1, 0.25, 1.0]])

        for name in self.varnames:
            setattr(self, name, vardict[name + '_values'])
