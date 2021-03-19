import unittest
import pickle
import json
import tempfile

import qcodes
import qcodes.data.io
from qcodes.data.data_set import DataSet
import qtt.simulation.virtual_dot_array
from qtt.structures import onedot_t, MultiParameter, CombiParameter


class TestStructures(unittest.TestCase):

    def setUp(self):
        DataSet.default_io = qcodes.data.io.DiskIO(tempfile.mkdtemp(prefix='qtt-unittests'))

    @staticmethod
    def test_sensingdot_t(fig=1, verbose=0):
        station = qtt.simulation.virtual_dot_array.initialize(verbose=verbose)
        sensing_dot = qtt.structures.sensingdot_t(
            ['SD1a', 'SD1b', 'SD1c'], station=station, minstrument='keithley1.amplitude')
        sensing_dot.verbose = verbose
        if fig is not None:
            _ = sensing_dot.autoTune(step=-8, fig=fig)
        qtt.simulation.virtual_dot_array.close(verbose=verbose)
