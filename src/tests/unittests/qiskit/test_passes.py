import unittest

import numpy as np
import qiskit.quantum_info as qi
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CZGate, HGate, RXGate, RYGate, RZGate

from qtt.qiskit.passes import (DecomposeCX, DecomposeU, DelayPass,
                               LinearTopologyParallelPass,
                               RemoveDiagonalGatesAfterInput,
                               RemoveSmallRotations, SequentialPass)


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

    def test_DelayPass(self, draw=False):
        delay_pass = DelayPass(gate_durations={'x': 80, 'h': 40})

        qc = QuantumCircuit(2)
        qc.h(0)
        qcd = delay_pass(qc)
        node = list(qcd)[-1][0]
        self.assertEqual(node.name, 'delay')
        self.assertEqual(node.params, [40])

        delay_pass = DelayPass(gate_durations={'x': 80, 'h': 40}, delay_quantum=40)
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qcd = delay_pass(qc)

        qc0 = QuantumCircuit(3)
        qc0.h(0)
        qc0.delay(40, 0)
        qc0.x(1)
        qc0.delay(40, 2)
        qc0.delay(40, 2)
        qcd = delay_pass(qc)

        self.assertEqual(qc0, qcd)
        self.check_identical(qc0, qcd)

    def test_SequentialPass(self):
        qc = QuantumCircuit(2)
        qc.p(np.pi, 0)
        qc.x(1)

        qc_target = QuantumCircuit(2)
        qc_target.p(np.pi, 0)
        qc_target.barrier()
        qc_target.x(1)
        qc_target.barrier()

        qc_transpiled = SequentialPass()(qc)
        self.check_identical(qc_transpiled, qc_target)

    def test_LinearTopologyParallelPass(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.x(2)

        qc_target = QuantumCircuit(3)
        qc_target.h(0)
        qc_target.x(2)
        qc_target.barrier()
        qc_target.x(1)
        qc_target.barrier()

        qc_transpiled = LinearTopologyParallelPass()(qc)
        self.check_identical(qc_transpiled, qc_target)


if __name__ == '__main__':
    import qiskit

    from qtt.utilities.tools import logging_context
    unittest.main()

    qc = QuantumCircuit(1)
    qc.p(np.pi, 0)
    dag = qiskit.converters.circuit_to_dag(qc)
