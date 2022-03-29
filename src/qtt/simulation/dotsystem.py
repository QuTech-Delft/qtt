""" Simulation of a coupled dot system."""
# %% Load packages
import copy
import itertools
import logging
import sys
import time
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la

try:
    import graphviz
except ImportError:
    pass

try:
    from pathos.multiprocessing import Pool

    _have_mp = True
except ImportError:
    _have_mp = False

import qtt.utilities.tools
from qtt import pgeometry


def showGraph(dot, fig=10):
    """ Show graphviz object in matplotlib window."""
    dot.format = 'png'
    outfile = dot.render('dot-dummy', view=False)
    print(outfile)

    im = plt.imread(outfile)
    plt.figure(fig)
    plt.clf()
    plt.imshow(im)
    plt.axis('off')


def static_var(varname, value):
    """ Helper function to create a static variable."""
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("time", 0)
def tprint(string, dt=1, output=False):
    """ Print progress of a loop every dt seconds."""
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
    """ Return True if matrix is diagonal."""
    return not(np.any(HH - np.diag(np.diagonal(HH))))


def _simulate_row(i, ds, npointsy, usediag):
    """ Helper function."""
    dsx = copy.deepcopy(ds)
    parameter_names = list(dsx.vals2D.keys())
    for j in range(npointsy):
        for name in parameter_names:
            setattr(dsx, name, dsx.vals2D[name][i][j])
        dsx.makeH()
        dsx.solveH(usediag=usediag)
        dsx.hcgs[i, j] = dsx.OCC
    return dsx.hcgs[i]


# %%


class GateTransform:

    def __init__(self, Vmatrix, sourcenames, targetnames):
        """ Class to describe a linear transformation between source and target gates."""
        self.Vmatrix = np.array(Vmatrix).astype(float)
        self.sourcenames = sourcenames
        self.targetnames = targetnames

    def transformGateScan(self, vals2D, nn=None):
        """ Get a list of parameter names and [c1 c2 c3 c4] 'corner' values
        to generate dictionary self.vals2D[name] = matrix of values.

        Args:
            vals2D (dict): keys are the gate names, values are matrices with the gate values.
            nn : TODO.
        Returns:
            dict: tranformed gate values.
        """
        vals2Dout = {}

        zz = np.zeros(nn if nn is not None else (), dtype=float)
        if isinstance(vals2D, dict):
            xx = [vals2D.get(s, zz) for s in self.sourcenames]
            xx = [x.flatten() for x in xx]
            gate_values = np.vstack(xx).astype(float)

        else:
            gate_values = np.array(vals2D).astype(float)

        gate_values_out = pgeometry.projectiveTransformation(self.Vmatrix, gate_values)

        for j, n in enumerate(self.targetnames):
            vals2Dout[n] = gate_values_out[j].reshape(nn).astype(float)
        return vals2Dout


class BaseDotSystem:
    """ Base class for the dot simulation classes.

    Based on the arguments the system calculates the energies of the different
    dot states. Using the energies the ground state, occupancies etc. can be calculated.
    The spin-state of electrons in the dots is ignored.

    The main functionality:

        * Build a Hamiltonian from the number of dots
        * Solve for the eigenvalues and eigenstates of the Hamiltonian
        * Present the results.

    The model used is [reference xxx].

    Attributes:

        number_of_basis_states (int): number of basis states.
        H (array): Hamiltonian of the system.

        energies (array): calculated energy for each state (ordered).
        states (array): eigenstates expressed in the basis states.
        stateprobs (array): TODO.
        stateoccs (array): TODO.
        nstates (array): for each state the number of electrons.

    """

    def __init__(self, name='basedotsystem', ndots=3, maxelectrons=2):
        """
        Args:
            name (str): name of the system.
            ndots (int): number of dots to simulate.
            maxelectrons (int): maximum occupancy per dot.
        """
        self.name = name
        self.ndots = ndots
        self.maxelectrons = maxelectrons

        self.makebasis(self.ndots, self.maxelectrons)

    @abstractmethod
    def calculate_ground_state(self, gatevalues):
        """ Calculate ground state for a set of gate values."""
        pass

    # could be a static method
    def findtransitions(self, occs):
        """ Find transitions in occupancy image."""
        transitions = np.full(
            [np.shape(occs)[0], np.shape(occs)[1]], 0, dtype=float)
        delocalizations = np.full(
            [np.shape(occs)[0], np.shape(occs)[1]], 0, dtype=float)

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

        return transitions, delocalizations

    def orderstatesbyN(self):
        """ Order the calculated states by occupation."""
        sortinds = np.argsort(self.nstates)
        self.energies = self.energies[sortinds]
        self.states = self.states[sortinds]
        self.stateprobs = self.stateprobs[sortinds]
        self.stateoccs = self.stateoccs[sortinds]
        self.nstates = self.nstates[sortinds]

    def orderstatesbyE(self):
        """ Order the calculated states by energy."""
        sortinds = np.argsort(self.energies)
        self.energies = self.energies[sortinds]
        self.states = self.states[sortinds]
        self.stateprobs = self.stateprobs[sortinds]
        self.stateoccs = self.stateoccs[sortinds]
        self.nstates = self.nstates[sortinds]

    def showstates(self, n):
        """ List states of the system with energies."""
        print('\nEnergies/states list for %s:' % self.name)
        print('-----------------------------------')
        for i in range(n):
            print(str(i) + '       - energy: ' + str(np.around(self.energies[i], decimals=2)) + ' ,      state: ' +
                  str(np.around(self.stateoccs[i], decimals=2)) + ' ,      Ne = ' + str(self.nstates[i]))
        print(' ')

    def makebasis(self, ndots, maxelectrons=2):
        """ Define a basis of occupancy states with a specified number of dots and max occupancy.

        The basis consists of vectors of length (ndots) where each entry in the vector indicates
        the number of electrons in a dot. The number of electrons in the total system is specified
        in `nbasis`.

        Args:
            ndots (int): number of dots to simulate.
            maxelectrons (int): maximum occupancy per dot.
        """
        assert(self.ndots == ndots)
        assert(self.maxelectrons == maxelectrons)
        basis = list(itertools.product(range(maxelectrons + 1), repeat=ndots))
        basis = np.asarray(sorted(basis, key=lambda x: sum(x)))
        self.basis = np.ndarray.astype(basis, int)
        self.number_of_electrons = np.sum(self.basis, axis=1)
        self.number_of_basis_states = len(self.number_of_electrons)
        self.H = np.zeros((self.number_of_basis_states, self.number_of_basis_states), dtype=float)
        self.eigenstates = np.zeros((self.number_of_basis_states, self.number_of_basis_states), dtype=float)

# %%


class DotSystem(BaseDotSystem):

    """ Class to simulate a system of interacting quantum dots.

    For a full model see "Quantum simulation of a Fermi-Hubbard model using a semiconductor quantum dot array":
        https://arxiv.org/pdf/1702.07511.pdf

    Args:
        name (str): name of the system.
        ndots (int): number of dots to simulate.
        maxelectrons (int): maximum occupancy per dot.

    Attributes:

    For the chemical potential, on site charging energy and inter site charging energy there are
    variables in the object. The names are given by the functions chemical_potential_name, etc.

    """
    _matrix_prefix = '_M'

    def __init__(self, name='dotsystem', ndots=3, maxelectrons=2, **kwargs):
        super().__init__(name=name, ndots=ndots, maxelectrons=maxelectrons)

        self.temperature = 0

    def get_chemical_potential(self, dot):
        return getattr(self, self.chemical_potential_name(dot))

    def set_chemical_potential(self, dot, value):
        return setattr(self, self.chemical_potential_name(dot), value)

    def get_on_site_charging(self, dot):
        return getattr(self, self.on_site_charging_name(dot))

    def set_on_site_charging(self, dot, value):
        return setattr(self, self.on_site_charging_name(dot), value)

    @staticmethod
    def chemical_potential_name(dot) -> str:
        return 'det%d' % dot

    @staticmethod
    def chemical_potential_matrix(dot):
        return DotSystem._matrix_prefix + DotSystem.chemical_potential_name(dot)

    @staticmethod
    def on_site_charging_name(dot):
        return 'onsiteC%d' % dot

    @staticmethod
    def on_site_charging_matrix(dot):
        return DotSystem._matrix_prefix + DotSystem.on_site_charging_name(dot)

    @staticmethod
    def inter_site_charging_name(dot1, dot2=None):
        """ Return name for nearest - neighbour charging energy."""
        if dot2 is None:
            dot2 = (dot1 + 1)
        return "intersiteC%d%d" % (dot1, dot2)

    @staticmethod
    def inter_site_charging_matrix(dot1, dot2=None):
        return DotSystem._matrix_prefix + DotSystem.inter_site_charging_name(dot1, dot2)

    @staticmethod
    def tunneling_name(dot_left):
        return 'tun%d' % dot_left

    @staticmethod
    def tunneling_matrix(dot_left):
        return DotSystem._matrix_prefix + DotSystem.tunneling_name(dot_left)

    def make_variables(self):
        """ Create value and matrix for a single variable."""
        for name in self.varnames:
            setattr(self, name, 0.)
            # also define that these are float32 numbers!
            setattr(self, self._matrix_prefix + name,
                    np.full((self.number_of_basis_states, self.number_of_basis_states), 0, dtype=int))

    def _make_variable_matrices(self, ring=False):
        """ Create matrices for the interactions.

        These matrices are used to quickly calculate the Hamiltonian.

        Args:
            ring (bool): set to True if the dot array in a ring configuration (e.g. 2x2).
        """
        m = np.zeros(self.ndots, dtype=int)

        def mkb(i, j):
            mx = m.copy()
            mx[i] = 1
            mx[j] = -1
            return mx

        potential = range(0, -(self.maxelectrons + 1), -1)
        on_site_potential = [n * (n - 1) / 2 for n in range(0, self.maxelectrons + 1)]
        for i in range(self.number_of_basis_states):
            for j in range(self.number_of_basis_states):
                if i == j:
                    for dot in range(1, self.ndots + 1):
                        dot_index = dot - 1
                        next_dot_idx = (dot_index + 1) % self.ndots
                        next_dot = next_dot_idx + 1

                        n = self.basis[i, dot_index]
                        getattr(self, self.chemical_potential_matrix(dot))[i, i] = potential[n]
                        getattr(self, self.on_site_charging_matrix(dot))[i, i] = on_site_potential[n]

                        if next_dot_idx > dot_index or ring:
                            n2 = self.basis[i, next_dot_idx]
                            # nearest-neighbour charging energy
                            logging.info('set inter_site_charging for dot %d-%d' % (dot, next_dot))
                            getattr(self, self._matrix_prefix +
                                    self.inter_site_charging_name(dot, next_dot))[i, i] = n * n2

                        second_next_dot_idx = (dot_index + 2) % self.ndots
                        n3 = self.basis[i, second_next_dot_idx]
                        # specific for 2x2 example!!
                        if getattr(self, 'is2x2', False):
                            if i % 2:
                                # next-nearest-neighbour charging energy
                                getattr(self, self.inter_site_charging_matrix(2, 4))[i, i] = n * n3
                            else:
                                # next-nearest-neighbour charging energy
                                getattr(self, self.inter_site_charging_matrix(1, 3))[i, i] = n * n3
                else:
                    statediff = self.basis[i, :] - self.basis[j, :]

                    for p in range(self.ndots - 1):
                        pn = p + 1
                        if (statediff == mkb(p, pn)).all() or (statediff == mkb(pn, p)).all():
                            getattr(self, self.tunneling_matrix(pn))[i, j] = -1  # tunneling term
                        elif ring and ((statediff == mkb(0, self.ndots - 1)).all() or
                                       (statediff == mkb(self.ndots - 1, 0)).all()):
                            getattr(self, self.tunneling_matrix(self.ndots))[i, j] = -1  # tunneling term at boundary
                        pass

        self.initSparse()

    @staticmethod
    def _sparse_matrix_name(variable):
        return '_sparseM%s' % (variable,)

    def initSparse(self):
        """ Create sparse structures.
        Constructing a matrix using sparse elements can be faster than construction of a full matrix,
        especially for larger systems.
        """
        self.H = np.zeros((self.number_of_basis_states, self.number_of_basis_states), dtype=float)

        for name in self.varnames:
            A = getattr(self, self._matrix_prefix + name)
            ind = A.flatten().nonzero()[0]
            setattr(self, 'indM' + name, ind)
            setattr(self, self._sparse_matrix_name(name), A.flat[ind])

    def makeH(self):
        """ Create a new Hamiltonian."""
        self.H.fill(0)
        for name in self.varnames:
            val = getattr(self, name)
            if not val == 0:
                self.H += getattr(self, self._matrix_prefix + name) * val
        self.solved = False
        return self.H

    def makeHsparse(self, verbose=0):
        """ Create a new sparse Hamiltonian."""
        self.H.fill(0)
        for name in self.varnames:
            if verbose:
                print('set %s: %f' % (name, getattr(self, name)))
            val = float(getattr(self, name))
            if not val == 0:
                a = getattr(self, self._sparse_matrix_name(name))
                ind = getattr(self, 'indM' + name)
                self.H.flat[ind] += a * val
        self.solved = False
        return self.H

    def solveH(self, usediag=False):
        """ Solve the system by calculating the eigenvalues and eigenstates of the Hamiltonian.

            Args:
                usediag (bool) : TODO.
        """
        if usediag:
            self.energies = self.H.diagonal()
            idx = np.argsort(self.energies)
            self.energies = self.energies[idx]
            # =np.zeros( (self.Nt, self.Nt), dtype=float)
            self.eigenstates[:] = 0
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

    def getall(self, param):
        """ Return all stored values for a particular parameter.

        Args:
            param (str): start of one of the variable names.

        Returns:
            vals (list): values corresponding to the parameter that was queried.
        """
        numvars = 0
        for var in self.varnames:
            if var.startswith(param):
                numvars += 1
        vals = [getattr(self, param + str(i + 1)) for i in range(numvars)]
        return vals

    def setall(self, param, vals):
        """ Sets all values for a particular parameter.

        Args:
            param (str): start of one of the variable names.
            vals (list): values corresponding to the parameter to be set.
        """
        numvars = 0
        for var in self.varnames:
            if var.startswith(param):
                numvars += 1
        if len(vals) != numvars:
            raise Exception
        for i in range(numvars):
            setattr(self, param + str(i + 1), vals[i])

    def simulate_honeycomb(self, paramvalues2D, verbose=1, usediag=False, multiprocess=True):
        self.vals2D = {}
        for i in range(paramvalues2D.shape[0]):
            nm = self.chemical_potential_name(i + 1)
            self.vals2D[nm] = paramvalues2D[i, :, :]
        return self.simulatehoneycomb(verbose=verbose, usediag=usediag, multiprocess=multiprocess)

    def simulatehoneycomb(self, verbose=1, usediag=False, multiprocess=False):
        """ Loop over the 2D matrix of parameter values defined by makeparamvalues2D, calculate the ground state
            for each point, search for transitions and save in self.honeycomb.
        """
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
            result = pool.starmap_async(_simulate_row, aa)
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
            print('simulatehoneycomb: %.2f [s] (multiprocess %s)' % (time.time() - t0, multiprocess))

        sys.stdout.flush()

    def simulatehoneycomb_original(self, verbose=1, usediag=False):
        """ Loop over the 2D matrix of parameter values defined by makeparamvalues2D, calculate the ground state
            for each point, search for transitions and save in self.honeycomb.

        Args:
             verbose (int): verbosity (0 == silent).
             usediag (bool): TODO.
        """
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
                    exec('self.' + name
                         + ' = self.vals2D[name][' + str(i) + '][' + str(j) + ']')

                self.makeH()
                self.solveH(usediag=usediag)
                self.hcgs[i, j] = self.OCC
        self.honeycomb, self.deloc = self.findtransitions(self.hcgs)

        if verbose:
            print('simulatehoneycomb: %.1f [s]' % (time.time() - t0))

    def calculate_energies(self, gatevalues):
        """ Calculate energies of the different states in the system.

        Args:
             gatevalues (list): values for the chemical potentials in the dots.

        """
        for i, val in enumerate(gatevalues):
            setattr(self, self.chemical_potential_name(i + 1), val)
        self.makeHsparse()
        self.solveH()
        return self.energies

    def calculate_ground_state(self, gatevalues):
        """ Calculate the ground state of the dot system, given a set of gate values.

        Args:
             gatevalues (list): values for the chemical potentials in the dots.

        Returns:
             array: a state array.

        """
        _ = self.calculate_energies(gatevalues)
        return self.stateoccs[0]

    def findcurrentoccupancy(self, exact=True):
        if self.solved:
            self.orderstatesbyE()
            if exact:
                # almost exact...
                idx = self.energies == self.energies[0]
                self.OCC = np.around(
                    np.mean(self.stateoccs[idx], axis=0), decimals=2)
            else:
                # first order approximation
                self.OCC = np.around(self.stateoccs[0], decimals=2)
        else:
            self.solveH()
        return self.OCC

    def makeparamvalues1D(self, paramnames, startend, npoints):
        """ Get a list of parameter names and [start end] values
            to generate dictionary self.vals1D[name] = vector of values.

        """
        self.vals1D = {}
        for i in range(len(paramnames)):
            name = paramnames[i]
            self.vals1D[name] = np.linspace(
                startend[i][0], startend[i][1], num=npoints)

    def makeparamvalues2D(self, paramnames, cornervals, npointsx, npointsy):
        """ Get a list of parameter names and [c1 c2 c3 c4] 'corner' values
            to generate dictionary self.vals2D[name] = matrix of values.

        """
        self.vals2D = {}
        for i in range(len(paramnames)):
            name = paramnames[i]
            if len(cornervals[i]) == 2:
                cornervals[i] = np.append(cornervals[i], cornervals[i])
                bottomrow = np.linspace(
                    cornervals[i][0], cornervals[i][1], num=npointsx)
                toprow = np.linspace(
                    cornervals[i][2], cornervals[i][3], num=npointsx)
            bottomrow = np.linspace(
                cornervals[i][0], cornervals[i][2], num=npointsx)
            toprow = np.linspace(
                cornervals[i][1], cornervals[i][3], num=npointsx)
            self.vals2D[name] = np.array(
                [np.linspace(i, j, num=npointsy) for i, j in zip(bottomrow, toprow)])

    def resetMu(self, value=0):
        """ Reset chemical potential.
            Args:
                value (int) : TODO.
        """
        for ii in range(self.ndots):
            setattr(self, self.chemical_potential_name(ii + 1), value)

    def showMmatrix(self, name=None, fig=10):
        if name is None:
            name = self.chemical_potential_name(1)
        plt.figure(fig)
        plt.clf()
        plt.imshow(getattr(self, 'M' + name), interpolation='nearest')
        plt.title('M' + name)
        plt.grid('on')

    def showvars(self):
        print('\nVariable list for %s:' % self.name)
        print('----------------------------')
        for name in self.varnames:
            print(name + ' = ' + str(getattr(self, 'name')))
        print(' ')

    def visualize(self, fig=1):
        """ Create a graphical representation of the system (needs graphviz).

            Args:
                fig (int): figure number, None is silent.

        """
        if self.ndots is None:
            print('no number of dots defined...')
            return
        dot = graphviz.Digraph(name=self.name)

        for ii in range(self.ndots):
            dot.node(str(ii), label='dot %d' % ii)
            dot.edge(str(ii), str(ii), label=self.chemical_potential_name(ii))

        showGraph(dot, fig=fig)


# %% Example dot systems

class OneDot(DotSystem):

    def __init__(self, name='onedot', maxelectrons=3):
        """ Simulation of a single quantum dot."""
        super().__init__(name=name, ndots=1, maxelectrons=maxelectrons)

        self.varnames = [self.chemical_potential_name(dot + 1) for dot in range(self.ndots)] \
            + [self.on_site_charging_name(i + 1) for i in range(self.ndots)]
        self.make_variables()
        self._make_variable_matrices()
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()


class DoubleDot(DotSystem):

    def __init__(self, name='doubledot', maxelectrons=3):
        """ Simulation of double-dot system.
        See: DotSystem.
        """
        super().__init__(name=name, ndots=2, maxelectrons=maxelectrons)

        self.varnames = [self.chemical_potential_name(dot + 1) for dot in range(self.ndots)] \
            + [self.on_site_charging_name(i + 1) for i in range(self.ndots)] \
            + [self.inter_site_charging_name(i + 1, i + 2) for i in range(self.ndots - 1)] + \
            [self.tunneling_name(dot) for dot in range(1, 2)]
        self.make_variables()
        self._make_variable_matrices()
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()


class TripleDot(DotSystem):

    def __init__(self, name='tripledot', maxelectrons=3):
        """ Simulation of triple-dot system."""
        super().__init__(name=name, ndots=3, maxelectrons=maxelectrons)

        self.varnames = [self.chemical_potential_name(dot + 1) for dot in range(self.ndots)] \
            + [self.on_site_charging_name(i + 1) for i in range(self.ndots)] \
            + [self.inter_site_charging_name(i + 1, i + 2) for i in range(self.ndots - 1)] \
            + [self.tunneling_name(dot) for dot in range(1, 3)]
        self.make_variables()
        self._make_variable_matrices()
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()


class FourDot(DotSystem):

    def __init__(self, name='fourdot', use_tunneling=True, maxelectrons=2, **kwargs):
        """ Simulation of 4-dot system."""
        super().__init__(name=name, ndots=4, maxelectrons=maxelectrons, **kwargs)

        self.use_tunneling = use_tunneling
        self.varnames = [self.chemical_potential_name(i + 1) for i in range(self.ndots)]
        self.varnames += [self.on_site_charging_name(i + 1) for i in range(self.ndots)]
        self.varnames += [self.inter_site_charging_name(i + 1) for i in range(4)]
        if self.use_tunneling:
            self.varnames += [self.tunneling_name(dot + 1) for dot in range(self.ndots)]
        self.make_variables()
        self._make_variable_matrices()
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()


class TwoXTwo(DotSystem):
    def __init__(self, name='2x2', maxelectrons=2):
        self.is2x2 = True
        super().__init__(name=name, ndots=4, maxelectrons=maxelectrons)

        self.varnames = [self.chemical_potential_name(i + 1) for i in range(self.ndots)] + \
                        [self.on_site_charging_name(i + 1) for i in range(self.ndots)] + \
                        [self.inter_site_charging_name(dot1, dot2)
                         for dot1, dot2 in [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4)]] + \
                        [self.tunneling_name(dot + 1) for dot in range(self.ndots)]
        self.make_variables()
        self._make_variable_matrices(ring=True)
        # initial run
        self.makeH()
        self.solveH()
        self.findcurrentoccupancy()
