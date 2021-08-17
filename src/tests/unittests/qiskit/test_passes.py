import unittest

import numpy as np
import qiskit.quantum_info as qi
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CZGate, HGate, RXGate, RYGate, RZGate

from qtt.qiskit.passes import (DecomposeCX, DecomposeU,
                               RemoveDiagonalGatesAfterInput,
                               RemoveSmallRotations)


class TestQiskitPasses(unittest.TestCase):

    def test_DecomposeCX(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(1)

        qcd = DecomposeCX()(qc)

        self.check_identical(qc, qcd)
        self.assertEqual(len(qcd), 4)
        self.assertIsInstance(list(qcd)[0][0], RYGate)
        self.assertIsInstance(list(qcd)[1][0], CZGate)
        self.assertIsInstance(list(qcd)[2][0], RYGate)
        self.assertIsInstance(list(qcd)[3][0], HGate)

    def check_identical(self, qc, qcd):
        I = qc.compose(qcd.inverse())
        op = qi.Operator(I)

        U = op.data/np.sqrt(complex(np.linalg.det(op.data)))
        U *= U[0, 0]
        np.testing.assert_almost_equal(U, np.eye(2**qc.num_qubits))

    def test_RemoveDiagonalGatesAfterInput(self):
        c = QuantumCircuit(2)
        c.rz(.1, 0)
        c.cz(0, 1)

        remove_diagonal_gates_pass = RemoveDiagonalGatesAfterInput()
        c = remove_diagonal_gates_pass(c)

        c = QuantumCircuit(2, 2)
        c.measure(0, 0)
        c.reset(0)
        c.cz(0, 1)
        c = remove_diagonal_gates_pass(c)
        assert len(c) == 2

    def test_RemoveSmallRotations(self):

        identity_circuit = QuantumCircuit(2)
        c = QuantumCircuit(2)
        c.rz(0, 0)
        c.rx(0, 1)
        c.ry(0, 1)

        remove_small_rotations = RemoveSmallRotations()
        qc = remove_small_rotations(c)
        self.assertEqual(len(qc), 0)
        self.check_identical(qc, identity_circuit)

    def test_RemoveSmallRotations_epsilon(self):
        remove_small_rotations = RemoveSmallRotations()
        c = QuantumCircuit(1)
        c.rz(1e-10, 0)
        qc = remove_small_rotations(c)
        self.check_identical(qc, c)

        remove_small_rotations = RemoveSmallRotations(epsilon=1e-6)
        qc = remove_small_rotations(c)
        self.check_identical(qc, QuantumCircuit(1))

    def test_RemoveSmallRotations_modulo(self):
        c = QuantumCircuit(1)
        c.ry(np.pi*2, 0)

        remove_small_rotations = RemoveSmallRotations()
        qc = remove_small_rotations(c)
        self.check_identical(qc, c)

        remove_small_rotations = RemoveSmallRotations(modulo2pi=True)
        qc = remove_small_rotations(c)
        self.check_identical(qc, QuantumCircuit(1))

    def test_DecomposeU(self, draw=False):
        decomposeU = DecomposeU()

        qc = QuantumCircuit(1)
        qc.p(np.pi, 0)
        qcd = decomposeU(qc)
        self.check_identical(qc, qcd)

        qc = QuantumCircuit(1)
        qc.u(np.pi/2, 0.1, 0., 0)
        qcd = decomposeU(qc)
        self.check_identical(qc, qcd)
        self.assertEqual(len(qcd), 3)
        self.assertIsInstance(list(qcd)[1][0], RXGate)

        qc = QuantumCircuit(1)
        qc.u(0.1, 0.2, 0.3, 0)
        qcd = decomposeU(qc)
        self.check_identical(qc, qcd)
        self.assertIsInstance(list(qcd)[0][0], RZGate)
        self.assertIsInstance(list(qcd)[1][0], RXGate)
        self.assertIsInstance(list(qcd)[2][0], RZGate)
        self.assertIsInstance(list(qcd)[3][0], RXGate)
        self.assertIsInstance(list(qcd)[4][0], RZGate)


if __name__ == '__main__':
    import qiskit
    unittest.main()

    qc = QuantumCircuit(1)
    qc.p(np.pi, 0)
    dag = qiskit.converters.circuit_to_dag(qc)
