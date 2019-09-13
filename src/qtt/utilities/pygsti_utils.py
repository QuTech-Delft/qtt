__doc__ = """utility functions for pyGSTi package"""
from typing import List, Union
import warnings
import os
try:
    os.environ['PYGSTI_BACKCOMPAT_WARNING'] = '0'
    from pygsti.objects import Circuit
    from pygsti.baseobjs.label import LabelTup
except ImportError:
    import qtt.exceptions
    warnings.warn('Warning: pygsti not found. pygsti_utils not available',
                  qtt.exceptions.MissingOptionalPackageWarning)
    Circuit = None
    LabelTub = None

__to_gst_gate_map = {
    # qtt gate : pygsty gate
    'X90': 'Gxpi2',
    'X': 'Gxpi',
    'Y90': 'Gypi2',
    'Y': 'Gypi',
    'I': 'Gi',
    'CX': 'Gcnot',
    'CZ': 'Gcphase'
}

__from_gst_gate_map = dict(zip(__to_gst_gate_map.values(), __to_gst_gate_map.keys()))


def from_gst_gate(gst_gate: Union[str, LabelTup]) -> str:
    if isinstance(gst_gate, LabelTup):
        gst_gate = gst_gate.name
    try:
        return __from_gst_gate_map[gst_gate]
    except KeyError:
        raise RuntimeError('pyGSTi gate ' + gst_gate + ' not supported')


def from_gst_gate_list(gst_gate_name_list: List[str]) -> List[str]:
    return [from_gst_gate(g) for g in gst_gate_name_list]


def from_gst_circuit(gst_circuit: Circuit) -> List[str]:
    return from_gst_gate_list(gst_circuit.tup)


def to_gst_gate(qtt_gate_name: str) -> str:
    try:
        return __to_gst_gate_map[qtt_gate_name]
    except KeyError:
        raise RuntimeError('qtt gate ' + qtt_gate_name + ' not supported')


def to_gst_gate_list(qtt_gate_name_list: List[str]) -> List[str]:
    try:
        return [to_gst_gate(g) for g in qtt_gate_name_list]
    except RuntimeError:
        raise
