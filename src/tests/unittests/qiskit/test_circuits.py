import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit

from qtt.qiskit.circuits import integrate_circuit

class TestCircuits(unittest.TestCase):

    def setUp(self):
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.h(0)
        qc.ry(np.pi/2, 0)
        qc.sdg(0)
        qc.measure(0, 0)
        self.circuit = qc

    def test_integrate_circuit(self):
        new_qc = integrate_circuit(self.circuit, target_qubits=[0], number_of_qubits=5)
        self.assertEqual(new_qc.width(), 5*self.circuit.width())
        self.assertEqual(new_qc.depth(), self.circuit.depth())
        for new, old in zip(new_qc, self.circuit):
            self.assertEqual(new[0].name, old[0].name)
            self.assertEqual(new[0].params, old[0].params)

        new_qc = integrate_circuit(
            self.circuit, target_qubits=[2], target_control_bits=[1], number_of_qubits=5, add_measurements=False)
        self.assertEqual(new_qc.width(), 5*self.circuit.width())
        self.assertEqual(new_qc.depth(), self.circuit.depth())
        for new, old in zip(new_qc, self.circuit):
            self.assertEqual(new[0].name, old[0].name)
            self.assertEqual(new[0].params, old[0].params)

