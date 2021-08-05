from typing import List, Sequence, Union

import numpy as np
import qiskit
from qiskit.circuit import Reset
from qiskit.circuit.library import (CRXGate, CRYGate, CRZGate, CZGate,
                                    PhaseGate, RXGate, RYGate, RZGate, U1Gate,
                                    U2Gate, U3Gate, UGate)
from qiskit.circuit.library.standard_gates import (CU1Gate, RZZGate, SdgGate,
                                                   SGate, TdgGate, TGate,
                                                   ZGate)
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager


class RemoveSmallRotations(TransformationPass):
    """Return a circuit with small rotation gates removed."""

    def __init__(self, epsilon: float = 0, modulo2pi=False):
        """Remove all small rotations from a circuit
        Args:
            epsilon: Threshold for rotation angle to be removed
            modulo2pi: If True, then rotations multiples of 2pi are removed as well
        """
        super().__init__()

        self.epsilon = epsilon
        self._empty_dag1 = qiskit.converters.circuit_to_dag(QuantumCircuit(1))
        self._empty_dag2 = qiskit.converters.circuit_to_dag(QuantumCircuit(2))
        self.mod2pi = modulo2pi

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on `dag`.
        Args:
            dag: input dag.
        Returns:
            Output dag with small rotations removed
        """
        def modulo_2pi(x):
            x = float(x)
            return np.mod(x + np.pi, 2 * np.pi) - np.pi
        for node in dag.op_nodes():
            if isinstance(node.op, (PhaseGate, RXGate, RYGate, RZGate)):
                phi = float(node.op.params[0])
                if self.mod2pi:
                    phi = modulo_2pi(phi)
                if np.abs(phi) <= self.epsilon:
                    dag.substitute_node_with_dag(node, self._empty_dag1)
            elif isinstance(node.op, (CRXGate, CRYGate, CRZGate)):
                phi = float(node.op.params[0])
                if self.mod2pi:
                    phi = modulo_2pi(phi)
                if np.abs(phi) <= self.epsilon:
                    dag.substitute_node_with_dag(node, self._empty_dag2)
        return dag


class RemoveDiagonalGatesAfterInput(TransformationPass):
    """Remove diagonal gates (including diagonal 2Q gates) at the start of a circuit.

    Transpiler pass to remove diagonal gates (like RZ, T, Z, etc) at the start of a circuit.
    Including diagonal 2Q gates. Nodes after a reset are also included.
    """

    def run(self, dag):
        """Run the RemoveDiagonalGatesBeforeMeasure pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        diagonal_1q_gates = (RZGate, ZGate, TGate, SGate, TdgGate, SdgGate, U1Gate)
        diagonal_2q_gates = (CZGate, CRZGate, CU1Gate, RZZGate)

        nodes_to_remove = set()
        for input_node in (dag.input_map.values()):
            try:
                successor = next(dag.quantum_successors(input_node))
            except StopIteration:
                continue

            if successor.type == "op" and isinstance(successor.op, diagonal_1q_gates):
                nodes_to_remove.add(successor)

            def valid_predecessor(s):
                """ Return True of node is valid predecessor for removal """
                if s.type == 'in':
                    return True
                if s.type == "op" and isinstance(s.op, Reset):
                    return True
                return False
            if successor.type == "op" and isinstance(successor.op, diagonal_2q_gates):
                predecessors = dag.quantum_predecessors(successor)
                if all(valid_predecessor(s) for s in predecessors):
                    nodes_to_remove.add(successor)

        for node_to_remove in nodes_to_remove:
            dag.remove_op_node(node_to_remove)

        return dag


class DecomposeU(TransformationPass):
    """ Decompose U gates into elementary rotations Rx, Ry, Rz

    The U gates are decomposed using McKay decomposition.
    """

    def __init__(self, verbose=0):
        """
        Args:
        """
        super().__init__()
        self._subdags = []
        self.verbose = verbose
        self.initial_layout = None

    def ugate_replacement_circuit(self, ugate):
        qc = QuantumCircuit(1)
        if isinstance(ugate, (U3Gate, UGate)):
            theta, phi, lam = ugate.params

            if theta == np.pi/2:
                # a u2 gate
                qc.rz(lam - np.pi / 2, 0)
                qc.rx(np.pi / 2, 0)
                qc.rz(phi + np.pi / 2, 0)
            else:
                # from https://arxiv.org/pdf/1707.03429.pdf
                qc.rz(lam, 0)
                qc.rx(np.pi / 2, 0)
                qc.rz(theta + np.pi, 0)
                qc.rx(np.pi / 2, 0)
                qc.rz(phi + np.pi, 0)

        elif isinstance(ugate, U2Gate):
            phi, lam = ugate.params
            qc.rz(lam - np.pi / 2, 0)
            qc.rx(np.pi / 2, 0)
            qc.rz(phi + np.pi / 2, 0)
        elif isinstance(ugate, (U1Gate, PhaseGate)):
            lam, = ugate.params
            qc.rz(lam, 0)
        else:
            raise Exception(f'unknown gate type {ugate}')
        return qc

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Decompose pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            output dag where ``CX`` was expanded.
        """
        # Walk through the DAG and expand each node if required
        for node in dag.op_nodes():
            if isinstance(node.op, (PhaseGate, U1Gate, U2Gate, U3Gate, UGate)):
                subdag = circuit_to_dag(self.ugate_replacement_circuit(node.op))
                dag.substitute_node_with_dag(node, subdag)
        return dag


class DecomposeCX(TransformationPass):
    """ Decompose CX into CZ and single qubit rotations
    """

    def __init__(self, mode: str = 'ry'):
        """
        Args:
        """
        super().__init__()
        self._subdags: List = []
        self.initial_layout = None
        self.gate = qiskit.circuit.library.CXGate

        self.decomposition = QuantumCircuit(2)
        if mode == 'ry':
            self.decomposition.ry(-np.pi / 2, 1)
            self.decomposition.cz(0, 1)
            self.decomposition.ry(np.pi / 2, 1)
        else:
            self.decomposition.h(1)
            self.decomposition.cz(0, 1)
            self.decomposition.h(1)

        self._dag = circuit_to_dag(self.decomposition)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Decompose pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            output dag where ``CX`` was expanded.
        """
        # Walk through the DAG and expand each non-basis node
        for node in dag.op_nodes(self.gate):
            dag.substitute_node_with_dag(node, self._dag)
        return dag
