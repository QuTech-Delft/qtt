# -*- coding: utf-8 -*-
""" Classical Quantum Dot Honeycomb Calculator

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
import numpy as np
import operator as op
import functools


def ncr(n, r):
    """ Calculating number of possible combinations: nCr"""
    r = min(r, n - r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n - r, -1))
    denom = functools.reduce(op.mul, range(1, r + 1))
    return numer // denom


class ClassicalDotSystem:

    def __init__(self, name='dotsystem', ndots=3, ngates=3, maxelectrons=3, **kwargs):
        self.name = name
        self.ndots = ndots
        self.ngates = ngates
        self.maxelectrons = maxelectrons

        # initialize attributes that are set later on
        self.basis = None   # basis of charge states
        self.nbasis = None  # corresponding total occupancy (for each charge state)
        self.Nt = None      # total number of charge states

        # initialize characterizing dot variables
        self.varnames = ['mu0', 'Eadd', 'W', 'alpha']
        self.mu0 = np.zeros((ndots,))               # chemical potential at zero gate voltage
        self.Eadd = np.zeros((ndots,))              # addition energy
        self.W = np.zeros((ncr(2, self.ndots),))    # coulomb repulsion
        self.alpha = np.zeros(ndots, ngates)        # virtual gate matrix, mapping gates to chemical potentials

    def makebasis(self):
        """ Define a basis of occupancy states """
        basis = list(itertools.product(range(self.maxelectrons + 1), repeat=self.ndots))
        basis = np.asarray(sorted(basis, key=lambda x: sum(x)))
        self.basis = np.ndarray.astype(basis, int)
        self.nbasis = np.sum(self.basis, axis=1)
        self.Nt = len(self.nbasis)

    def calculate_energies(self, gatevalues):
        """ Calculate the energies of all dot states, given a set of gate values. Returns array of energies. """
        energies = np.zeros((self.Nt,))
        for i in range(self.Nt):
            energy = 0
            energy += -(self.mu0 + np.dot(self.alpha, gatevalues))*self.basis[i]
            energy += np.dot([np.dot(*v) for v in itertools.combinations(self.basis[i], 2)], self.W)
            energy += np.dot((1/2 * np.multiply(self.basis[i], self.basis[i]+1)), self.Eadd)
            energies[i] = energy
        return energies

    def calculate_ground_state(self, gatevalues):
        """ Calculate the ground state of the dot system, given a set of gate values. Returns a state array. """
        energies = self.calculate_energies(gatevalues)
        return self.basis(np.argmin(energies))


class TripleDot(ClassicalDotSystem):

    def __init__(self, name='tripledot', **kwargs):
        super().__init__(name=name, ndots=3, ngates=3, **kwargs)

        self.makebasis()

        mu0_values = np.array([-27.0, -20.0, -25.0])    # chemical potential at zero gate voltage
        Eadd_values = np.array([54.0, 52.8, 54.0])      # addition energy
        W_values = np.array([6.0, 1.0, 5.0])            # coulomb repulsion (!order is important: (1,2), (1,3), (2,3))
                                                        # (lexicographic ordering)
        alpha_values = np.array([[1.0, 0.25, 0.1],
                                 [0.25, 1.0, 0.25],
                                 [0.1, 0.25, 1.0]])

        for name in self.varnames:
            exec('self.' + name + ' = ' + name + '_values')
