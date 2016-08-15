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
        basis = list(itertools.product(range(maxelectrons + 1), repeat=ndots))
        basis = np.asarray(sorted(basis, key=lambda x: sum(x)))
        self.basis = np.ndarray.astype(basis, int)
        self.nbasis = np.sum(self.basis, axis=1)
        self.Nt = len(self.nbasis)

    def makevars(self, varnames):
        self.varnames = varnames
        for name in self.varnames:
            exec('self.' + name + ' = 0')

