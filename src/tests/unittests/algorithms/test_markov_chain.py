# -*- coding: utf-8 -*-
""" Classes to test generate signals with continuous-time Markov chains."""

import unittest
import numbers
import numpy as np
from qtt.algorithms.markov_chain import ChoiceGenerator, ContinuousTimeMarkovModel, generate_traces


class TestChoiceGenerator(unittest.TestCase):

    def test_ChoiceGenerator(self):
        generator = ChoiceGenerator(3, [0, .5, 1.])
        self.assertIsInstance(generator.generate_choice(), numbers.Integral)

        choices = [generator.generate_choice() for _ in range(1000)]
        self.assertNotIn(0, choices)

    def test_ChoiceGenerator_invalid_input(self):
        with self.assertRaises(Exception):
            ChoiceGenerator(2, cum_weights=[.3, .6, 1.])


class TestMarkovChain(unittest.TestCase):

    def setUp(self):
        self.rts_model = ContinuousTimeMarkovModel(['up', 'down'], [.002, .001], np.array([[0., 1], [1, 0]]))

    def test_repr(self, verbose=0):
        if verbose > 0:
            print(self.rts_model.__repr__())

    def test_number_of_states(self):
        self.assertEqual(self.rts_model.number_of_states(), 2)

    def test_update_model(self):
        holding_parameters=np.array([[1],[1.]])
        jump_chain=np.array([[0., 1], [1, 0]])
        self.rts_model.update_model(holding_parameters, jump_chain)
        np.testing.assert_array_almost_equal(self.rts_model.jump_chain, jump_chain)
        np.testing.assert_array_almost_equal(self.rts_model.holding_parameters, holding_parameters)
        np.testing.assert_array_almost_equal(self.rts_model.generator_matrix, np.array([[-1.,  1.],[ 1., -1.]]))

    def test_generate_sequence(self):
        length = 30
        sequence = self.rts_model.generate_sequence(length, delta_time=1)
        self.assertEqual(len(sequence), length)

    def test_generate_sequences(self):
        length = 30
        number_of_sequences = 5
        sequences = self.rts_model.generate_sequences(length, delta_time=1, number_of_sequences=number_of_sequences)
        self.assertEqual(sequences.shape, (number_of_sequences, length))

    def test_stationary_distribution(self):
        jump_chain = self.rts_model.jump_chain
        self.assertIsInstance(jump_chain, np.ndarray)
        generator_matrix = self.rts_model.generator_matrix
        expected_generator = np.array([[-0.002, 0.001], [0.002, -0.001]])
        np.testing.assert_array_almost_equal(generator_matrix, expected_generator)

        pi_tilde = ContinuousTimeMarkovModel.stationary_distribution_discrete(self.rts_model.jump_chain)
        expected_stationary_step = np.array([[0.5], [0.5]])
        np.testing.assert_array_equal(pi_tilde, expected_stationary_step)
        pi = self.rts_model.stationary_distribution()
        expected = np.array([[1. / 3], [2. / 3]])
        np.testing.assert_array_almost_equal(pi, expected)
        np.testing.assert_array_almost_equal(pi, self.rts_model.stationary_distribution_direct())

    def test_elzerman_model(self):
        model_unit = 1e-6
        generator_matrix = np.array([[-10., 2000., 2000.], [0., -12000., 0.], [10., 10000., -2000.]])
        holding_parameters = -np.diag(generator_matrix).reshape((-1, 1))
        self.assertTrue(np.all(holding_parameters > 0))
        jump_chain = (1. / holding_parameters.T) * generator_matrix
        jump_chain[np.diag_indices(generator_matrix.shape[0])] = 0

        elzerman_model = ContinuousTimeMarkovModel(
            ['spin-down', 'spin-up', 'empty'], holding_parameters * model_unit, jump_chain)
        pi = elzerman_model.stationary_distribution()
        np.testing.assert_array_almost_equal(pi, np.array([[0.995025], [0.], [0.004975]]))
        np.testing.assert_array_almost_equal(model_unit * generator_matrix, elzerman_model.generator_matrix)

    def test_generate_traces(self):
        x = generate_traces(self.rts_model, std_gaussian_noise=0, state_mapping=[0, 2], length=200, delta_time=1)
        self.assertTrue(np.all([value in [0, 2] for value in np.unique(x)]))
