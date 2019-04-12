import unittest
from unittest import TestCase

import qcodes
import qtt.data
import qtt.measurements.scans
from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer
from qtt.instrument_drivers.simulation_instruments import SimulationAWG
import qtt.simulation.virtual_dot_array
from qtt.measurements.videomode import VideoMode

# %%


class TestVideomode(TestCase):

    def test_videomode_getdataset(self):
        station = qtt.simulation.virtual_dot_array.initialize(reinit=True, verbose=0)

        digitizer = SimulationDigitizer(
            qtt.measurements.scans.instrumentName('sdigitizer'), model=station.model)
        station.add_component(digitizer)

        station.awg = SimulationAWG(qtt.measurements.scans.instrumentName('vawg'))
        station.add_component(station.awg)

        sweepparams = {'gates_horz': {'P1': 1}, 'gates_vert': {'P2': 1}}
        minstrument = (digitizer.name, [0])

        vm = VideoMode(station, sweepparams, sweepranges=[120] * 2,
                       minstrument=minstrument, resolution=[12] * 2, Naverage=2)
        vm.stop()
        vm.updatebg()
        data = vm.get_dataset()
        vm.close()

        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], qcodes.DataSet)
        self.assertEqual(data[0].measured.shape, (12, 12))

        vm = VideoMode(station, ['P1', 'P2'], sweepranges=[20] * 2,
                       minstrument=minstrument, resolution=[32] * 2, Naverage=2)
        vm.stop()
        vm.updatebg()
        data = vm.get_dataset()
        vm.close()

        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], qcodes.DataSet)
        self.assertEqual(data[0].measured.shape, (32, 32))

        for name, instrument in station.components.items():
            instrument.close()

        qtt.simulation.virtual_dot_array.close(verbose=0)


if __name__ == '__main__':
    unittest.main()
