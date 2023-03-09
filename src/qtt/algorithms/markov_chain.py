""" Class to generate signals with continous-time Markov chains


@author: pieter.eendebak@gmail.com
"""

import itertools
import random
from dataclasses import dataclass
from typing import List, Optional, Union, cast

import numpy as np
import scipy.linalg

IntArray = np.typing.NDArray[np.int_]


def _solve_least_squares(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rcond = None
    solution = np.linalg.lstsq(a, b, rcond=rcond)[0]
    return solution


@dataclass
class ChoiceGenerator:
    """ Class to generate random elements with weighted selection

    This is a replacement for `random.choices` that is efficient when a large number of choices has to be generated.

    number_of_states: number of choices that has to be generated
    cum_weights (array[float]): cumulative probabilities of the choices
    block_size: size of blocks of choices to generate
    """

    number_of_states: int
    cum_weights: np.ndarray
    block_size: int = 5000

    def __post_init__(self) -> None:
        if not self.number_of_states == len(self.cum_weights):
            raise Exception(
                f'specification of cumulative weights (len {len(self.cum_weights)})'
                + ' does not match number of states {self.number_of_states}')
        self.rng = np.random.Generator(np.random.SFC64())
        self._idx = 0
        self._block: List[int] = cast(List[int], self._generate_block().tolist())

    def _generate_block(self, size: Optional[int] = None) -> IntArray:
        if size is None:
            size = self.block_size
        else:
            self.block_size = size
        weights = np.concatenate(([self.cum_weights[0]], np.diff(self.cum_weights)))  # type: ignore
        counts = np.random.multinomial(size, weights)
        block: IntArray = np.concatenate(tuple(choice_idx * np.ones(c, dtype=int) for choice_idx,
                                                                                      c in enumerate(counts)))
        self.rng.shuffle(block)
        return block

    def generate_choice(self) -> int:
        """ Generate a single choice

        Returns:
            Integer in the range 0 to the number of states
        """
        self._idx = self._idx + 1
        if self._idx == self.block_size:
            self._idx = 0
            self._block = cast(List[int], self._generate_block().tolist())
        return self._block[self._idx]

    def generate_choices(self, size: int) -> np.ndarray:
        """ Generate a specified number of choice

        Returns:
            Array with elements in the range 0 to the number of states
        """
        data = self._generate_block(size)
        return data


class ContinuousTimeMarkovModel:

    def __init__(self, states: List[str], holding_parameters: Union[List[float], np.ndarray], jump_chain: np.ndarray):
        """ Class that represents a continuous-time Markov chain

        Args:
            states: list with names for the states
            holding_parameters: List with the holding parameters. The holding parameters determine the average
                time before the system will make a jump to a new state
            jump_chain: The jump chain or transition matrix. This matrix gives the probability for the system
                        to jump from a state to one of the other states. The sum of the probabilities in each
                        column must equal one.

        For an introduction to Markov chains see https://www.probabilitycourse.com/chapter11/11_3_1_introduction.php

        Also see: https://vknight.org/unpeudemath/code/2015/08/01/simulating_continuous_markov_chains.html

        """
        self.states = states
        self.update_model(np.asarray(holding_parameters), jump_chain)

    def update_model(self, holding_parameters: np.ndarray, jump_chain: np.ndarray):
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
    def _create_generator_matrix(holding_parameters: np.ndarray, jump_chain: np.ndarray) -> np.ndarray:
        generator_matrix = np.array(jump_chain, copy=True)
        for ii in range(generator_matrix.shape[0]):
            generator_matrix[:, ii] = holding_parameters[ii] * jump_chain[:, ii]
        for ii in range(generator_matrix.shape[0]):
            generator_matrix[ii, ii] = -holding_parameters[ii]
        return generator_matrix

    def number_of_states(self) -> int:
        """ Return the number of states in the model """
        return len(self.states)

    def transition_matrix(self, delta_time: float) -> np.ndarray:
        """ Return the transition matrix for a specified amount of time """
        transition_matrix = scipy.linalg.expm(delta_time * self.generator_matrix)
        return transition_matrix

    def __repr__(self):
        return "%s(id=0x%x, states=%s, generator=%s)" % (self.__class__.__name__,
                                                         id(self), self.states, self.generator_matrix)

    def stationary_distribution_direct(self) -> np.ndarray:
        """ Return the stationary distribution of the model

        The calculation method is taken from:
        https://www.probabilitycourse.com/chapter11/11_3_2_stationary_and_limiting_distributions.php, Theorem 11.3

        """
        pi_tilde = self.stationary_distribution_discrete(self.jump_chain)
        norm = np.sum(pi_tilde / self.holding_parameters)
        stationary_distribution = (pi_tilde / self.holding_parameters) / norm
        return stationary_distribution

    def stationary_distribution(self) -> np.ndarray:
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
    def stationary_distribution_discrete(jump_chain) -> np.ndarray:
        """ Return the stationary distrubution for a Markov chain """

        n = jump_chain.shape[0]
        A = np.vstack((jump_chain - np.eye(n), np.ones((1, n))))
        B = np.zeros((n + 1, 1))
        B[-1] = 1
        pi = _solve_least_squares(A, B)
        return pi

    def generate_sequence(self, length: int, delta_time: float,
                          initial_state: Union[None, int, np.ndarray] = None,
                          generators: Optional[List[ChoiceGenerator]] = None) -> np.ndarray:
        """ Generate a random sequence with the model

        Args:
            length: number of elements in the sequence
            delta_time: time step to be used. This is equal to one over the samplerate.
            initial_state: This parameter determines how the first element of the generated
                sequence is chosen. If an int, then use that state is initial state. If None then take
                a random state weighted by the stationary distribution. If the initial_state is a list, then the list
                is interpreted as a probability distribution and the first element is sampled from all possible states
                according to the distribution specified.
            generators: Optional list of generators to use
        Returns:
            Array with generated sequence

        """
        number_of_states = self.number_of_states()
        if initial_state is None:
            initial_state = self.stationary_distribution().flatten().tolist()
            initial_state = random.choices(range(number_of_states), weights=initial_state, k=1)[0]  # type: ignore
        elif isinstance(initial_state, (list, np.ndarray, tuple)):
            initial_state = np.asarray(initial_state).flatten().tolist()
            initial_state = random.choices(range(number_of_states), weights=initial_state, k=1)[0]  # type: ignore

        if generators is None:
            generators = self._create_generators(delta_time)

        sequence = np.zeros(length, dtype=int)
        sequence[0] = value = initial_state
        for i in range(1, sequence.size):
            # value points to the genererator of the previous element in the sequence, and is then updated directly
            sequence[i] = value = generators[value].generate_choice()
        return sequence

    def _create_generators(self, delta_time: float):
        number_of_states = self.number_of_states()
        P = self.transition_matrix(delta_time)
        generators: List[Optional[ChoiceGenerator]] = [None] * number_of_states
        # pre-calculate cumulative weights
        for jj in range(number_of_states):
            cum_weights = np.array(list(itertools.accumulate(P[:, jj])))
            generators[jj] = ChoiceGenerator(number_of_states, cum_weights)
        return generators

    def generate_sequences(self, length: int, delta_time: float = 1, initial_state: Union[None, int, np.ndarray] = None,
                           number_of_sequences: int = 1) -> np.ndarray:
        """ Generate multiple random sequences with the model

        Args:
            length: number of elements in the sequence
            delta_time: time step to be used
            initial_state: This parameter determines how the first element of the generated
                sequences are chosen. The parameter is passed to the :func:`generate_sequence` method.
            number_of_sequences : Specified the number of sequences to generate
        Returns:
            Array with generated sequences
        """
        if initial_state is None:
            initial_state = self.stationary_distribution()
        sequences = np.zeros((number_of_sequences, length), dtype=int)
        generators = self._create_generators(delta_time)
        for n in range(number_of_sequences):
            sequences[n] = self.generate_sequence(length, delta_time, initial_state, generators=generators)
        return sequences


def generate_traces(markov_model: ContinuousTimeMarkovModel, std_gaussian_noise: float = 1,
                    state_mapping: Optional[np.ndarray] = None, *args, **kwargs):
    """ Generate traces for a continuous-time Markov model with added noise

    Args:
        markov_model: model to use for generation of traces
        std_gaussian_noise: standard deviation of Gaussian noise to add to the output signal
        state_mapping: If not None, replace each state in the generated trace by the corresponding element in the array
        *args, **kwargs: passed to the `generate_sequences` function of the model

    The traces are generated by the `generate_sequences` method from the model.
    """

    traces = np.array(markov_model.generate_sequences(*args, **kwargs))

    if state_mapping is not None:
        traces = np.array(state_mapping)[traces]
    if std_gaussian_noise != 0:
        traces = traces + np.random.normal(0, std_gaussian_noise, traces.size).reshape(traces.shape)

    return traces
