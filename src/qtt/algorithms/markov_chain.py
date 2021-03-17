# -*- coding: utf-8 -*-
""" Class to generate signals with continous-time Markov chains


@author: pieter.eendebak@gmail.com
"""

# %%
import numpy as np
import random
import scipy.linalg
import itertools


def _solve_least_squares(a, b):
    rcond = None
    solution = np.linalg.lstsq(a, b, rcond=rcond)[0]
    return solution


class ChoiceGenerator:
    """ Class to generate random elements with weighted selection """

    def __init__(self, number_of_states, cum_weights, block_size=5000):
        """ Class to generate random elements with weighted selection

        This is a replacement for random.choices that is efficient when a large number of choices has to be generated.

        Args:
            number_of_states (int): number of choices that has to be generated
            cum_weights (array[float]): cummulative probabilities of the choices
            block_size (int): size of blocks of choices to generate
        """
        if not number_of_states == len(cum_weights):
            raise Exception('specification of cummulative weights does not match number of states')

        self.number_of_states = number_of_states
        self._idx = 0
        self._block_size = block_size
        self.cum_weights = cum_weights
        self._generate_block()

    def _generate_block(self):
        values = np.random.rand(self._block_size, )
        counts, _ = np.histogram(values, [0] + list(self.cum_weights))
        self._block = np.hstack(tuple([choice_idx * np.ones(c, dtype=int) for choice_idx, c in enumerate(counts)]))
        np.random.shuffle(self._block)

    def generate_choice(self):
        """ Generate a choice

        Returns:
            int: integer in the range 0 to the number of states
        """
        self._idx = self._idx + 1
        if self._idx == self._block_size:
            self._idx = 0
            self._generate_block()
        return self._block[self._idx]


class ContinuousTimeMarkovModel:

    def __init__(self, states, holding_parameters, jump_chain):
        """ Class that represents a continous-time Markov chain

        Args:
            states (str[]): list with names for the states
            holding_parameters(float[]): List with the holding parameters. The holding parameters determine the average
                time before the system will make a jump to a new state
            jump_chain (array): The jump chain or transition matrix. This matrix gives the probability for the system
                        to jump from a state to one of the other states. The sum of the probabilities in each column must
                        equal one.

        For an introduction to Markov chains see https://www.probabilitycourse.com/chapter11/11_3_1_introduction.php

        Also see: https://vknight.org/unpeudemath/code/2015/08/01/simulating_continuous_markov_chains.html

        """
        self.states = states
        self.update_model(holding_parameters, jump_chain)

    def update_model(self, holding_parameters, jump_chain):
        """ Update the model of the markov chain

        Args:
            holding_parameters: List with the holding parameters
            jump_chain: The jump chain or transition matrix

        For a detailed description of the parameters see the class documentation.
        """
        self.holding_parameters = np.array(holding_parameters).flatten().reshape((-1, 1))
        self.jump_chain = jump_chain
        self.generator_matrix = self._create_generator_matrix(self.holding_parameters, self.jump_chain)
        self._validity_check()

    def _validity_check(self):
        if len(self.states) != len(self.jump_chain):
            raise AssertionError('States do not equal jump chain!')

        if not np.allclose(np.sum(self.jump_chain, axis=0), 1):
            raise AssertionError('Jump chain matrix should represent probabilities!')

        if np.all(self.holding_parameters <= 0):
            raise AssertionError('Not all holding parameter are bigger than zero!')

    @staticmethod
    def _create_generator_matrix(holding_parameters, jump_chain):
        generator_matrix = np.array(jump_chain, copy=True)
        for ii in range(generator_matrix.shape[0]):
            generator_matrix[:, ii] = holding_parameters[ii] * jump_chain[:, ii]
        for ii in range(generator_matrix.shape[0]):
            generator_matrix[ii, ii] = -holding_parameters[ii]
        return generator_matrix

    def number_of_states(self):
        """ Return the number of states in the model """
        return len(self.states)

    def transition_matrix(self, delta_time):
        """ Return the transition matrix for a specified amount of time """
        transition_matrix = scipy.linalg.expm(delta_time * self.generator_matrix)
        return transition_matrix

    def __repr__(self):
        return "%s(id=0x%x, states=%s, generator=%s)" % (self.__class__.__name__,
                                                         id(self), self.states, self.generator_matrix)

    def stationary_distribution_direct(self):
        """ Return the stationary distribution of the model

        The calculation method is taken from:
        https://www.probabilitycourse.com/chapter11/11_3_2_stationary_and_limiting_distributions.php, Theorem 11.3

        """
        pi_tilde = self.stationary_distribution_discrete(self.jump_chain)
        norm = np.sum((pi_tilde / self.holding_parameters))
        stationary_distribution = (pi_tilde / self.holding_parameters) / norm
        return stationary_distribution

    def stationary_distribution(self):
        """ Return the stationary distribution of the model

        The calculation method is taken from:
        https://vknight.org/unpeudemath/code/2015/08/01/simulating_continuous_markov_chains.html
        """
        Q = self.generator_matrix
        n = Q.shape[0]
        A = np.vstack((Q, np.ones((1, n))))
        B = np.zeros((n + 1, 1))
        B[-1] = 1
        stationary_distribution = _solve_least_squares(A, B)
        return stationary_distribution

    @staticmethod
    def stationary_distribution_discrete(jump_chain):
        """ Return the stationary distrubution for a Markov chain """

        n = jump_chain.shape[0]
        A = np.vstack((jump_chain - np.eye(n), np.ones((1, n))))
        B = np.zeros((n + 1, 1))
        B[-1] = 1
        pi = _solve_least_squares(A, B)
        return pi

    def generate_sequence(self, length, delta_time, initial_state=None):
        """ Generate a random sequence with the model

        Args:
            length (int): number of elements in the sequence
            delta_time (float): time step to be used. This is equal to one over the samplerate.
            initial_state (None or int or list): This parameter determines how the first element of the generated
                sequence is chosen. If an int, then use that state is initial state. If None then take
                a random state weighted by the stationary distribution. If the initial_state is a list, then the list
                is interpreted as a probability distribution and the first element is samples from all possible states
                according to the distribution specified.
        Returns:
            array : generated sequence

        """
        number_of_states = self.number_of_states()
        if initial_state is None:
            initial_state = self.stationary_distribution()
            initial_state = random.choices(range(number_of_states), weights=initial_state, k=1)[0]
        elif isinstance(initial_state, (list, np.ndarray, tuple)):
            initial_state = random.choices(range(number_of_states), weights=initial_state, k=1)[0]

        P = self.transition_matrix(delta_time)

        sequence = np.zeros(length, dtype=int)
        sequence[0] = initial_state

        # pre-calculate cummulative weights
        generators = [None] * number_of_states
        for jj in range(number_of_states):
            cum_weights = list(itertools.accumulate(P[:, jj]))
            generators[jj] = ChoiceGenerator(number_of_states, cum_weights)

        for i in range(1, sequence.size):
            sequence[i] = generators[sequence[i - 1]].generate_choice()
        return sequence

    def generate_sequences(self, length, delta_time=1, initial_state=None, number_of_sequences=1):
        """ Generate multiple random sequences with the model

        Args:
            length (int): number of elements in the sequence
            delta_time (float): time step to be used
            initial_state (None or int or list): This parameter determines how the first element of the generated
                sequences are chosen. The parameter is passed to the :func:`generate_sequence` method.
            number_of_sequences (int): Specified the number of sequences to generate
        Returns:
            array : generated sequences
        """
        if initial_state is None:
            initial_state = self.stationary_distribution()
        sequences = np.zeros((number_of_sequences, length), dtype=int)
        for n in range(number_of_sequences):
            sequences[n] = self.generate_sequence(length, delta_time, initial_state)
        return sequences


def generate_traces(markov_model, std_gaussian_noise=1, state_mapping=None, *args, **kwargs):
    """ Generate traces for a continuous-time Markov model with added noise

    Args:
        markov_model (ContinuousTimeMarkovModel): model to use for generation of traces
        std_gaussian_noise (float): standard deviation of Gaussian noise to add to the output signal
        state_mapping (None or array): If not None, replace each state in the generated trace by the corresponding element in the array
        *args, **kwargs: passed to the `generate_sequences` function of the model

    The traces are generated by the generate_sequences method from the model.
    """

    traces = np.array(markov_model.generate_sequences(*args, **kwargs))

    if state_mapping is not None:
        traces = np.array(state_mapping)[traces]
    if std_gaussian_noise != 0:
        traces = traces + np.random.normal(0, std_gaussian_noise, traces.size).reshape(traces.shape)

    return traces
