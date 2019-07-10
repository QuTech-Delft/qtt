try:
    import pygsti
    from pygsti.extras import rb
    from pygsti.extras.rb.results import RBSummaryDataset, RBResults
except ImportError:
    print('Warning: pygsti not found. randomized benchmarking not available')
    pygsti = None
    rb = None
    RBResults = None
    RBSummaryDataset = None


# utilty functions for pyGSTi package
# TODO: move to utilities
class GSTUtils:

    # gate naming remapping from qtt naming to pyGSTi naming
    _to_gst_gate_map = {
        'X':  'Gxpi2',
        'X2': 'Gxpi',
        'Y':  'Gypi2',
        'Y2': 'Gypi',
        'I':  'Gi',
        'CX': 'Gcnot',
        'CZ': 'Gcphase'
    }

    # reverse gate naming remapping from qtt naming to pyGSTi naming
    _from_gst_gate_map = dict(zip(_to_gst_gate_map.values(), _to_gst_gate_map.keys()))

    @staticmethod
    def from_gst_gate(gst_gate_name: str) -> str:
        try:
            return GSTUtils._from_gst_gate_map[gst_gate_name]
        except KeyError:
            raise RuntimeError('pyGSTi gate ' + gst_gate_name + ' not supported')

    @staticmethod
    def from_gst_gate_list(gst_gate_name_list: list) -> list:
        try:
            return [GSTUtils._from_gst_gate_map[g] for g in gst_gate_name_list]
        except KeyError:
            raise RuntimeError('pyGSTi gate ' + g + ' not supported')

    @staticmethod
    def to_gst_gate(qtt_gate_name: str) -> str:
        try:
            return GSTUtils._to_gst_gate_map[qtt_gate_name]
        except KeyError:
            raise RuntimeError('qtt gate ' + qtt_gate_name + ' not supported')

    @staticmethod
    def to_gst_gate_list(qtt_gate_name_list : list) -> list:
        try:
            return [GSTUtils._to_gst_gate_map[g] for g in qtt_gate_name_list]
        except KeyError:
            raise RuntimeError('qtt gate ' + g + ' not supported')


class RandomizedBenchmarkingSingleQubit:

    def __init__(self, gate_names: list = None):
        """Construct object for randomized benchmarking on specified qubit chip.
        Args:
            gate_names : list of supported single qubit gate names
                Default: ['X', 'X2', 'Y', 'Y2']
        """
        # set argument defaults
        if gate_names is None:
            # set default here rather than as default argument value to avoid that object instances change the default
            gate_names = ['X', 'X2', 'Y', 'Y2']

        gst_gate_names = GSTUtils.to_gst_gate_list(gate_names)

        self.qubit_labels = ['Q0']
        self.chip_spec = pygsti.obj.ProcessorSpec(1, gst_gate_names, availability={}, qubit_labels=self.qubit_labels)

    def generate_clifford_circuits(self, lengths : (list,tuple), num_seq : int) -> dict:
        """Generate sequences of gates for a full set of randomized benchmarking experirments.

        Args:
            lengths: sequence with clifford lengths of the sequences to generate
            num_seq: number of different sequences to generate for each length

        Returns:
            a dictionary containing: key = clifford length, value = list of random circuits of that length.
        """

        exp_dict = rb.sample.clifford_rb_experiment(self.chip_spec, lengths, num_seq,
                                                    subsetQs=self.qubit_labels, randomizeout=False)
        circuits = exp_dict['circuits']
        circuits_dict = {m: [circuits[m, i] for i in range(num_seq)] for m in lengths}
        return circuits_dict

    def simulate_circuits_probabilities(self, circuits_dict: dict) -> dict:
        model = pygsti.construction.build_standard_localnoise_model(nQubits=len(self.chip_spec.qubit_labels),
                                                                    gate_names=self.chip_spec.root_gate_names,
                                                                    qubit_labels=self.chip_spec.qubit_labels)
        sim_out = {m: [c.simulate(model) for c in cs] for (m, cs) in circuits_dict}
        return sim_out

    def simulate_summary_data(self, lengths: (list,tuple), num_seq: int, repetitions: int,
                              error_rate: float) -> RBSummaryDataset:
        # This creates this error model in the format needed for the simulator
        gate_error_rate_dict = {q: error_rate for q in self.chip_spec.qubit_labels}
        error_model = rb.simulate.create_locally_gate_independent_pauli_error_model(self.chip_spec,
                                                                                    gate_error_rate_dict,
                                                                                    ptype='uniform')
        return rb.simulate.rb_with_pauli_errors(self.chip_spec, error_model, lengths, num_seq, repetitions,
                                                rbtype='CRB', verbosity=1)

    def analyse_summary_data(self, data: RBSummaryDataset) -> RBResults:
        return rb.analysis.std_practice_analysis(data, rtype='AGI')
