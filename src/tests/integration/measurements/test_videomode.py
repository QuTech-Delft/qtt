import unittest
import unittest.mock as mock
import io

import qcodes
import qtt.data
import qtt.measurements.scans
from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer
from qtt.instrument_drivers.simulation_instruments import SimulationAWG
import qtt.simulation.virtual_dot_array
from qtt.measurements.videomode import VideoMode


class TestVideoMode(unittest.TestCase):

    def test_video_mode_update_position(self):
        with mock.patch('sys.stdout', new_callable=io.StringIO):
            station = qtt.simulation.virtual_dot_array.initialize(reinit=True, verbose=0)

            digitizer = SimulationDigitizer(
                qtt.measurements.scans.instrumentName('sdigitizer'), model=station.model)
            station.add_component(digitizer)
            minstrument = (digitizer.name, [0])

            station.awg = SimulationAWG(qtt.measurements.scans.instrumentName('vawg'))
            station.add_component(station.awg)

            vm = VideoMode(station, 'P1', sweepranges=[10.],
                           minstrument=minstrument, resolution=[12], Naverage=2)
            new_position = 2
            vm._update_position((new_position, 0))
            self.assertEqual(station.gates.P1(), new_position)

    def test_VideoMode_all_instances(self):
        self.assertIsInstance(VideoMode.all_instances(), list)
        station = qtt.simulation.virtual_dot_array.initialize()
        dummy_processor = DummyVideoModeProcessor(station)
        videomode = VideoMode(station, dorun=False, nplots=1, videomode_processor=dummy_processor)
        self.assertIn(videomode, VideoMode.all_instances())

        VideoMode.stop_all_instances()
    
    def test_video_1d(self):
        with mock.patch('sys.stdout', new_callable=io.StringIO):
            station = qtt.simulation.virtual_dot_array.initialize(reinit=True, verbose=0)

            digitizer = SimulationDigitizer(
                qtt.measurements.scans.instrumentName('sdigitizer'), model=station.model)
            station.add_component(digitizer)
            station.awg = SimulationAWG(qtt.measurements.scans.instrumentName('vawg'))
            station.add_component(station.awg)

            minstrument = (digitizer.name, [0])
            videomode = VideoMode(station, 'P1', sweepranges=[120],
                                  minstrument=minstrument, Naverage=2)
            self.assertEqual(videomode.videomode_processor.scan_dimension(), 1)

    def test_video_2d(self):
        with mock.patch('sys.stdout', new_callable=io.StringIO):
            station = qtt.simulation.virtual_dot_array.initialize(reinit=True, verbose=0)

            digitizer = SimulationDigitizer(
                qtt.measurements.scans.instrumentName('sdigitizer'), model=station.model)
            station.add_component(digitizer)
            station.awg = SimulationAWG(qtt.measurements.scans.instrumentName('vawg'))
            station.add_component(station.awg)

            sweepparams = ['P1', 'P2']
            minstrument = (digitizer.name, [0])
            videomode = VideoMode(station, sweepparams, sweepranges=[120] * 2,
                                  minstrument=minstrument, resolution=[12] * 2, Naverage=2)
            self.assertEqual(videomode.videomode_processor.scan_dimension(), 2)

    def test_video_mode_get_data_set(self, verbose=0):
        with mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            station = qtt.simulation.virtual_dot_array.initialize(reinit=True, verbose=verbose)

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

            for _, instrument in station.components.items():
                instrument.close()

            qtt.simulation.virtual_dot_array.close(verbose=verbose)

            std_output = mock_stdout.getvalue()
            print(std_output)
            self.assertIn('VideoMode: start readout', std_output)
