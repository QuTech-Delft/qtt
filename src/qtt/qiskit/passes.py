import logging
from typing import Dict, List, Optional

import numpy as np
import qiskit
from qiskit.circuit import Barrier, Delay, Reset
from qiskit.circuit.library import (CRXGate, CRYGate, CRZGate, CZGate, PhaseGate, RXGate, RYGate, RZGate, U1Gate,
                                    U2Gate, U3Gate, UGate)
from qiskit.circuit.library.standard_gates import CU1Gate, RZZGate, SdgGate, SGate, TdgGate, TGate, ZGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGInNode, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass

logger = logging.getLogger(__name__)


class RemoveGateByName(TransformationPass):
    """Return a circuit with all gates with specified name removed.

    This transformation is not semantics preserving.
    """

    def __init__(self, gate_name: str, *args, **kwargs):
        """Remove all gates with specified name from a DAG
        Args:
            gate_name: Name of the gate to be removed from a DAG
        """
        super().__init__(*args, **kwargs)
        self._gate_name = gate_name

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the RemoveGateByName pass on `dag`."""

        dag.remove_all_ops_named(self._gate_name)

        return dag

    def __repr__(self) -> str:
        name = self.__class__.__module__ + '.' + self.__class__.__name__
        return f"<{name} at 0x{id(self):x}: gate {self._gate_name}"


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
                if node.op.is_parameterized():
                    # for parameterized gates we do not optimize
                    pass
                else:
                    phi = float(node.op.params[0])
                    if self.mod2pi:
                        phi = modulo_2pi(phi)
                    if np.abs(phi) <= self.epsilon:
                        dag.substitute_node_with_dag(node, self._empty_dag1)
            elif isinstance(node.op, (CRXGate, CRYGate, CRZGate)):
                if node.op.is_parameterized():
                    # for parameterized gates we do not optimize
                    pass
                else:
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

            if isinstance(successor, DAGOpNode) and isinstance(successor.op, diagonal_1q_gates):
                nodes_to_remove.add(successor)

            def valid_predecessor(s):
                """ Return True of node is valid predecessor for removal """
                if isinstance(s, DAGInNode):
                    return True
                if isinstance(s, DAGOpNode) and isinstance(s.op, Reset):
                    return True
                return False
            if isinstance(successor, DAGOpNode) and isinstance(successor.op, diagonal_2q_gates):
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
            dag: input DAG.

        Returns:
            Output DAG where ``U`` gates have been decomposed.
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


class SequentialPass(TransformationPass):
    """Adds barriers between gates to make the circuit sequential."""

    def run(self, dag):
        new_dag = DAGCircuit()

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        for node in dag.op_nodes():
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            logger.info('SequentialPass: adding node {node.name}')
            if node.name in ['barrier', 'measure']:
                continue
            new_dag.apply_operation_back(Barrier(new_dag.num_qubits()), list(new_dag.qubits), [])

        return new_dag


class LinearTopologyParallelPass(TransformationPass):
    """Adds barriers to enforce a linear topology

    The barrier are placed between gates such that no two qubit gates are executed
    at the same time and only single qubit gates on non-neighboring qubits can
    be executed in parallel. It assumes a linear topology."""

    def run(self, dag):
        new_dag = DAGCircuit()

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        for ii, layer in enumerate(dag.layers()):
            gates_1q = []
            gates_2q = []
            other_gates = []
            for node in layer['graph'].op_nodes():
                if len(node.qargs) == 2:
                    gates_2q.append(node)
                elif len(node.qargs) == 1:
                    gates_1q.append(node)
                else:
                    logging.info(f'layer {ii}: other type of node {node}')
                    other_gates.append(node)

            even = []
            odd = []
            for node in gates_1q:
                if node.qargs[0].index % 2 == 0:
                    even.append(node)
                else:
                    odd.append(node)
            logging.info(
                f'layer {ii}: 2q gates {len(gates_2q)}, even {len(even)} odd {len(odd)}, other {len(other_gates)}')

            if len(even) > 0:
                for node in even:
                    new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                if not isinstance(node.op, Barrier):
                    new_dag.apply_operation_back(Barrier(new_dag.num_qubits()), list(new_dag.qubits), [])

            if len(odd) > 0:
                for node in odd:
                    new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                if not isinstance(node.op, Barrier):
                    new_dag.apply_operation_back(Barrier(new_dag.num_qubits()), list(new_dag.qubits), [])

            for node in gates_2q:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                if not isinstance(node.op, Barrier):
                    new_dag.apply_operation_back(Barrier(new_dag.num_qubits()), list(new_dag.qubits), [])

            for node in other_gates:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

                if not isinstance(node.op, Barrier):
                    new_dag.apply_operation_back(Barrier(new_dag.num_qubits()), list(new_dag.qubits), [])

        return new_dag


class DelayPass(TransformationPass):
    """Adds delay gates when the qubits are idle.
    For every layer of the circuit it finds the gate that
    lasts the longest and applies appropriate delays on the
    other qubits.
    """

    def __init__(self, gate_durations: Dict[str, float], delay_quantum: Optional[float] = None):
        """
        Args:
            gate_durations: Gate durations in the units of dt
        """
        super().__init__()
        self.gate_durations = gate_durations
        self.delay_quantum = delay_quantum

    def add_delay_to_dag(self, duration, dag, qargs, cargs):
        if self.delay_quantum:
            number_of_delays = int(duration/self.delay_quantum)
            for ii in range(number_of_delays):
                dag.apply_operation_back(Delay(self.delay_quantum), qargs, cargs)
        else:
            dag.apply_operation_back(Delay(duration), qargs, cargs)

    @staticmethod
    def _determine_delay_target_qubits(dag, layer):
        """ Determine qubits in specified layer which require a delay gate """
        partition = layer['partition']
        lst = list(dag.qubits)
        for el in partition:
            for q in el:
                if q in lst:
                    lst.remove(q)
        return lst

    def run(self, dag):
        new_dag = DAGCircuit()

        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        for layer_idx, layer in enumerate(dag.layers()):
            max_duration = 0
            durations = {}
            for node in layer['graph'].op_nodes():
                if node.name in self.gate_durations:
                    max_duration = max(max_duration, self.gate_durations[node.name])
                    for q in node.qargs:
                        durations[q] = self.gate_durations[node.name]
                else:
                    logger.info('layer {layer_idx}, could not find duration for node {node.name}')
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

            partition = layer['partition']
            if len(partition) == 0:
                continue

            lst = DelayPass._determine_delay_target_qubits(dag, layer)
            logger.info(f'layer: {layer_idx}: lst {lst}, durations {durations}')
            for el in lst:
                logger.info(f'apply_operation_back: {[el]}')
                self.add_delay_to_dag(max_duration, new_dag, [el], [])
            for q in durations:
                if max_duration - durations[q] > 0:
                    self.add_delay_to_dag(max_duration - durations[q], new_dag, [q], [])

        return new_dag
