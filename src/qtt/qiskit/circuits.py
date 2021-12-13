from typing import List, Optional

from qiskit.circuit.quantumcircuit import QuantumCircuit


def integrate_circuit(
    qc: QuantumCircuit,
    target_qubits: List[int],
    target_control_bits: Optional[List[int]] = None,
    number_of_qubits: int = 5,
    add_measurements: bool = True,
) -> QuantumCircuit:
    """ Integrate circuit at specfied qubits in a larger qubit system

    This can be used for example to integrate a single-qubit experiment in
    a 5-qubit circuit to be executed on a 5-qubit device.

    Args:
        qc: QuantumCircuit to be integrated
        target_qubits: List of qubits to map the circuit on
        target_control_bits: If None, then use the registers allocated to the target qubits
        number_of_qubits: Number of qubits in the target system
        add_measurements: If True, then add a measure statement for all the new qubits

    Returns:
        Integrated circuit
    """
    if qc.num_qubits > number_of_qubits:
        raise Exception(
            f'number of qubits {qc.num_qubits} in the specified circuit cannot exceed specfied number of qubits {number_of_qubits} in the target system')

    output_qc = QuantumCircuit(number_of_qubits, number_of_qubits)

    qbits = [output_qc.qregs[0][i] for i in target_qubits]
    if target_control_bits is None:
        cbits = [output_qc.cregs[0][i] for i in target_qubits]
    else:
        cbits = target_control_bits
    output_qc = output_qc.compose(qc, qubits=qbits, clbits=cbits)

    if add_measurements:
        for qubit_index in range(number_of_qubits):
            if qubit_index not in target_qubits:
                output_qc.measure(qubit_index, qubit_index)
            else:
                continue
    return output_qc
