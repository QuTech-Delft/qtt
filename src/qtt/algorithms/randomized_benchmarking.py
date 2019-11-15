import warnings
from typing import List, Union, Tuple, Dict
import os
try:
    os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'
    import pygsti
    from pygsti.extras import rb
    from pygsti.extras.rb.results import RBSummaryDataset, RBResults
    from qtt.utilities import pygsti_utils
except ImportError:
    import qtt.exceptions
    warnings.warn('Warning: pygsti not found. Randomized benchmarking not available',
                  qtt.exceptions.MissingOptionalPackageWarning)
    pygsti = None
    rb = None
    RBResults = None
    RBSummaryDataset = None
    pygsti_utils = None


class CliffordRandomizedBenchmarkingSingleQubit:

    def __init__(self, gate_names: List[str] = None):
        """Construct object for randomized benchmarking for given set of supported gates.
        Args:
            gate_names : list of supported gate names
                Default: ['X', 'X90', 'Y', 'Y90']
        """
        if gate_names is None:
            gate_names = ['X', 'X90', 'Y', 'Y90']

        gst_gate_names = pygsti_utils.to_gst_gate_list(gate_names)

        self._qubit_labels = ['Q0']
        self._chip_spec = pygsti.obj.ProcessorSpec(1, gst_gate_names, availability={}, qubit_labels=self._qubit_labels)

    def generate_circuits(self, lengths: Union[List[int], Tuple[int]], num_seq: int) -> Dict:
        """Generate sequences of gates for a full set of Clifford randomized benchmarking experirments.

        Args:
            lengths: Clifford lengths of the sequences to generate
            num_seq: number of different sequences to generate for each length

        Returns:
            a dictionary containing: key = clifford length, value = list of random circuits of that length.
        """

        exp_dict = rb.sample.clifford_rb_experiment(self._chip_spec, lengths, num_seq,
                                                    subsetQs=self._qubit_labels, randomizeout=False)
        circuits = exp_dict['circuits']
        circuits_dict = {m: [circuits[m, i] for i in range(num_seq)] for m in lengths}
        return circuits_dict

    def generate_measurement_sequences(self,  lengths: Union[list, tuple], num_seq: int):
        """Generate qtt measurement sequences for randomized benchmarking."""
        circuits_dict = self.generate_circuits(lengths, num_seq)
        meas_dict = {}
        for m, circuits in circuits_dict.items():
            meas_dict[m] = [pygsti_utils.from_gst_circuit(c) for c in circuits]
        return meas_dict

    def simulate_circuits_probabilities(self, circuits_dict: dict) -> dict:
        model = pygsti.construction.build_standard_localnoise_model(nQubits=len(self._chip_spec.qubit_labels),
                                                                    gate_names=self._chip_spec.root_gate_names,
                                                                    qubit_labels=self._chip_spec.qubit_labels)
        sim_out = {m: [c.simulate(model) for c in cs] for (m, cs) in circuits_dict.items()}
        return sim_out

    def simulate_summary_data(self, lengths: Union[list, tuple], num_seq: int, repetitions: int,
                              error_rate: float) -> RBSummaryDataset:
        # This creates this error model in the format needed by the simulator
        gate_error_rate_dict = {q: error_rate for q in self._chip_spec.qubit_labels}
        error_model = rb.simulate.create_locally_gate_independent_pauli_error_model(self._chip_spec,
                                                                                    gate_error_rate_dict,
                                                                                    ptype='uniform')
        return rb.simulate.rb_with_pauli_errors(self._chip_spec, error_model, lengths, num_seq, repetitions,
                                                rbtype='CRB', verbosity=1)

    def analyse_summary_data(self, data: RBSummaryDataset) -> RBResults:
        return rb.analysis.std_practice_analysis(data, rtype='AGI')
