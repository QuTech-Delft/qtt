import unittest
from unittest import TestCase

import qtt.data
import qtt.measurements.scans
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qtt.instrument_drivers.gates import VirtualDAC

# %%


class TestVirtualDAC(TestCase):

    def setUp(self):
        gate_map = {
            'T': (0, 15), 'P1': (0, 3), 'P2': (0, 4),
            'L': (0, 5), 'D1': (0, 6), 'R': (0, 7)}

        self.ivvi = VirtualIVVI(qtt.measurements.scans.instrumentName('ivvi'), model=None)
        self.gates = VirtualDAC(qtt.measurements.scans.instrumentName('gates'),
                                instruments=[self.ivvi], gate_map=gate_map)

    def tearDown(self):
        self.gates.close()
        self.ivvi.close()

    def test_videomode_getdataset(self):
        from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer
        from qtt.instrument_drivers.simulation_instruments import SimulationAWG
        import qtt.simulation.virtual_dot_array
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
        data=vm.get_dataset()
        vm.close()
    
        self.assertIsInstance(data, list)
    
        for name, instrument in station.components.items():
            instrument.close()
    
        qtt.simulation.virtual_dot_array.close(verbose=0)

if __name__ == '__main__':
    unittest.main()
