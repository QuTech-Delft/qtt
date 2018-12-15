# -*- coding: utf-8 -*-
""" Class to generate signals with continous-time Markov chains


@author: pieter.eendebak@gmail.com
"""

#%%
import numpy as np
import random
import scipy.linalg

class ContinuousTimeMarkovModel:

    def __init__(self, states, holding_parameters, jump_chain):
        """ Class that represents a continous-time Markov chain
        
        Args:
            states (str[]): list with names for the states
            holding_parameters(float[]): list with the holding parameters
            jump_chain (array): The jump chain or transition matrix
        
        For an introduction to Markov chains see https://www.probabilitycourse.com/chapter11/11_3_1_introduction.php

        Also see: https://vknight.org/unpeudemath/code/2015/08/01/simulating_continuous_markov_chains.html
        
        """
        self.states = states
        self.jump_chain = jump_chain
        self.holding_parameters = np.array(holding_parameters).flatten().reshape( (-1,1))
        self.generator_matrix = self._create_generator_matrix(self.holding_parameters, self.jump_chain)
        self._validity_check()

    def _validity_check(self):
        assert(len(self.states)==len(self.jump_chain))
        if not (np.all(np.sum(self.jump_chain, axis=1)==1)):
            raise Exception('jump chain matrix should represent probabilities')
        
        assert(np.all(self.holding_parameters>0))
        
    @staticmethod
    def _create_generator_matrix(holding_parameters, jump_chain):
        G = np.array(jump_chain, copy=True)
        for ii in range(G.shape[0]):
            G[:,ii]=holding_parameters[ii]*jump_chain[:,ii]
        for ii in range(G.shape[0]):
            G[ii,ii]=-holding_parameters[ii]
        return G
    
    def number_states(self):
        """ Return the number of status in the model """
        return len(self.states)

    def transition_matrix(self, delta_time):
        """ Return the transition matrix for a specified amount of time """
        P = scipy.linalg.expm(delta_time*self.generator_matrix)
        return P
        
    def __repr__(self):
        return "%s(id=0x%x, states=%s, generator=%s)" % (self.__class__.__name__,
            id(self), self.states, self.generator_matrix)

    def stationary_distribution_direct(self):
        """ Return the stationary distrubution of the model
        
        From https://www.probabilitycourse.com/chapter11/11_3_2_stationary_and_limiting_distributions.php, Theorem 11.3
        """
        pi_tilde = self.stationary_distribution_discrete(self.jump_chain)
        norm = np.sum((pi_tilde / self.holding_parameters))
        pi = (pi_tilde / self.holding_parameters)/ norm
        return pi
    
    def stationary_distribution(self):
        """ Return the stationary distrubution of the model
        
        From https://vknight.org/unpeudemath/code/2015/08/01/simulating_continuous_markov_chains.html
        """
        Q=self.generator_matrix 
        n=Q.shape[0]
        A=np.vstack( (Q, np.ones( (1,n) ) ) )
        B=np.zeros( (n+1,1) ); B[-1]=1
        pi = np.linalg.lstsq(A,B, rcond=None)[0]
        return pi
    
    @staticmethod
    def stationary_distribution_discrete(jump_chain):
        """ Return the stationary distrubution for a Markov chain """

        n=jump_chain.shape[0]
        A=np.vstack( (jump_chain-np.eye(n), np.ones( (1,n) ) ) )
        B=np.zeros( (n+1,1) ); B[-1]=1
        pi = np.linalg.lstsq(A,B, rcond=None)[0]
        return pi

    def generate_sequence(self, length, delta_time, initial_state = None):
        """ Generate a random sequence with the model
        
        Args:
            length (int): number of elements in the sequence
            delta_time (float): time step to be used
            initial_state (None or int or list): If an int, then use that state is initial state. If None then take a random state weighted by the stationary distribution        
        Returns:
            array : generated sequence

        """
        n=self.number_states()
        if initial_state is None:
            initial_state = self.stationary_distribution()
            initial_state = random.choices(range(n), weights=initial_state, k=1)[0]
        elif isinstance(initial_state, (list, np.ndarray, tuple)):
            initial_state = random.choices(range(n), weights=initial_state, k=1)[0]

        P = self.transition_matrix(delta_time)

        sequence=np.zeros(length, dtype=int)
        sequence[0] = initial_state
        for i in range(1, sequence.size):
            p = P[:, sequence[i-1]]
            sequence[i]=random.choices(range(n), weights=p, k=1)[0]
        return sequence
    
    def generate_sequences(self, length, delta_time, initial_state = None, number_of_sequences=1):
        """ Generate multiple random sequences with the model
        
        Args:
            length (int): number of elements in the sequence
            delta_time (float): time step to be used
            initial_state (None or int or list): If an int, then use that state is initial state. If None then take a random state weighted by the stationary distribution        
            number_of_sequences (int):
        Returns:
            array : generated sequences
        """
        if initial_state is None:
            initial_state = self.stationary_distribution()
        sequences=np.zeros( (number_of_sequences, length))
        for n in range(number_of_sequences):
            sequences[n]=self.generate_sequence(length, delta_time, initial_state)
        return sequences
    

import unittest

class TestMarkovChain(unittest.TestCase):

    def setUp(self):
        self.rts_model = ContinuousTimeMarkovModel(['up', 'down'], [.002,.001], np.array([[0.,1],[1,0]]) )        
    
    def test_repr(self):
        print(self.rts_model.__repr__())

    def test_generate_sequence(self):
        length=30
        sequence = self.rts_model.generate_sequence(length, delta_time=1)
        self.assertEqual(len(sequence), length)

    def test_generate_sequences(self):
        length=30
        number_of_sequences= 5
        sequences = self.rts_model.generate_sequences(length, delta_time=1, number_of_sequences = number_of_sequences)
        self.assertEqual(sequences.shape, (number_of_sequences, length) )

    def test_stationary_distribution(self):
        jump_chain=self.rts_model.jump_chain
        generator_matrix = self.rts_model.generator_matrix
        expected_generator = np.array([[-0.002,  0.001],       [ 0.002, -0.001]])
        np.testing.assert_array_almost_equal(generator_matrix, expected_generator)

        pi_tilde = ContinuousTimeMarkovModel.stationary_distribution_discrete(self.rts_model.jump_chain)
        expected_stationary_step=np.array([[0.5], [0.5]])
        np.testing.assert_array_equal(pi_tilde, expected_stationary_step)
        pi=self.rts_model.stationary_distribution()
        expected=np.array([[1./3], [2./3]])
        np.testing.assert_array_almost_equal(pi, expected)
        np.testing.assert_array_almost_equal(pi, self.rts_model.stationary_distribution_direct())
        
if __name__=='__main__':
    unittest.main()
    
