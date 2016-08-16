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

    def makebasis(self):
        """ Define a basis of occupancy states """
        basis = list(itertools.product(range(self.maxelectrons + 1), repeat=self.ndots))
        basis = np.asarray(sorted(basis, key=lambda x: sum(x)))
        self.basis = np.ndarray.astype(basis, int)
        self.nbasis = np.sum(self.basis, axis=1)
        self.Nt = len(self.nbasis)

    def makevars(self, varnames):
        """ Create the variables that characterize the dot system """
        self.varnames = varnames
        for name in self.varnames:
            exec('self.' + name + ' = np.zeros((self.ndots,))')


class TripleDot(ClassicalDotSystem):

    def __init__(self, name='tripledot', **kwargs):
        super().__init__(name=name, ndots=3, ngates=3, **kwargs)

        self.makebasis()

        varnames = ['mu0', 'Eadd', 'W']
        varnames += ['alpha%d' % (i+1) for i in range(self.ndots)]
        self.makevars(varnames)

        mu0_values = np.array([-27.0, -20.0, -25.0])
        Eadd_values = np.array([54.0, 52.8, 54.0])
        W_values = np.array([6.0, 5.0, 1.0])
        alpha1_values = np.array([1.0, 0.25, 0.1])
        alpha2_values = np.array([0.25, 1.0, 0.25])
        alpha3_values = np.array([0.1, 0.25, 1.0])

        for name in self.varnames:
            exec('self.' + name + ' = ' + name + '_values')

        


