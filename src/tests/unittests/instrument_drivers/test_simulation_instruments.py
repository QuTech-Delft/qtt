import unittest
import numpy as np
from qtt.instrument_drivers.simulation_instruments import SimulationAWG, SimulationDigitizer
import qtt.simulation.virtual_dot_array
import matplotlib.pyplot as plt


class TestSimulationInstruments(unittest.TestCase):

    def test_simulated_digitizer(self, fig=None, verbose=0):
        station = qtt.simulation.virtual_dot_array.initialize(reinit=True, nr_dots=3, maxelectrons=2, verbose=verbose)

        station.model.sdnoise = .05

        station.gates.B0(-300)
        station.gates.B3(-300)
        awg = SimulationAWG(qtt.measurements.scans.instrumentName('test_simulation_awg'))
        waveform, _ = awg.sweep_gate('B3', 400, 1e-3)

        digitizer = SimulationDigitizer(qtt.measurements.scans.instrumentName('test_digitizer'), model=station.model)
        r = digitizer.measuresegment(waveform, channels=[1])

        self.assertTrue(isinstance(r[0], np.ndarray))

        if fig:
            plt.figure(fig)
            plt.clf()
            plt.plot(r[0], label='signal from simulation digitizer')
            plt.close('all')

        awg.close()
        digitizer.close()

        qtt.simulation.virtual_dot_array.close(verbose=verbose)
