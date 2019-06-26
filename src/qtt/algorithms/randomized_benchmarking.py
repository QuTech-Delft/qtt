
import numpy as np
import pygsti
from pygsti.construction import std1Q_XYI
from pygsti.extras import rb

# pyGSTi utilty functions
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
    def from_gst_gate_list(gst_gate_name_list: str) -> str:
        try:
            return [GSTUtils._from_gst_gate_map[g] for g in gst_gate_name_list]
        except KeyError:
            raise RuntimeError('pyGSTi gate ' + gst_gate_name + ' not supported')


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
        Arguments:
        gates : list of supported single qubit gate names
        Default: ['X', 'X2', 'Y', 'Y2']
        """
        # set argument defaults
        if gate_names is None:
            # set default here rather than as default argument value to avoid that object instances change the default
            gate_names = ['X', 'X2', 'Y', 'Y2']

        gst_gate_names = GSTUtils.to_gst_gate_list(gate_names)

        self.qubit_labels = ['Q0']
        self.chip_spec = pygsti.obj.ProcessorSpec(1, gst_gate_names, availability={}, qubit_labels=self.qubit_labels)


    def generate_clifford_circuits(self, lengths : list, repetitions : int) -> dict:
        """Generate sequences of gates for a full set of randomized benchmarking experirments.

        Arguments:

        lengths: array with clifford lengths of the sequences to generate

        repetitions: number of different sequences to generate for each length

        returns: a dictionary containing: key = clifford length, value = list of circuits at that length.
        """

        exp_dict = rb.sample.clifford_rb_experiment(self.chip_spec, lengths, repetitions,
                                                    subsetQs=self.qubit_labels, randomizeout=False)
        circuits = exp_dict['circuits']
        circuits_dict = dict([(m, [circuits[m, i] for i in range(repetitions)]) for m in lengths])

        return circuits_dict

