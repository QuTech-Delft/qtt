# Currently still required (because HDAWG8 PR has not been merged):
# 1. Clone the QCoDeS repository: https://github.com/qutech-sd/Qcodes.git
# 2. Checkout branch: feature/DEM-607/Update-generate-csv-sequence-program

from qcodes import Instrument, ManualParameter
from qcodes_contrib_drivers.drivers.ZurichInstruments.ZIHDAWG8 import ZIHDAWG8
from qcodes.instrument_drivers.ZI.ZIUHFLI import ZIUHFLI
from qcodes.station import Station
from qcodes.utils.validators import Numbers

from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg
from qtt.measurements.videomode import VideoMode


# Initialize the settings of the hardware

class HardwareSettings(Instrument):

    def __init__(self, name='settings'):
        super().__init__(name)
        awg_gates = {'P1': (0, 4)}
        awg_markers = {'m4i_mk': (0, 4, 0)}
        self.awg_map = {**awg_gates, **awg_markers}

        for awg_gate in self.awg_map:
            parameter_name = 'awg_to_{}'.format(awg_gate)
            parameter_label = '{} (factor)'.format(parameter_name)
            self.add_parameter(parameter_name, parameter_class=ManualParameter,
                               initial_value=1000, label=parameter_label, vals=Numbers(1, 1000))

settings = HardwareSettings()


# Initialize the arbitrary waveform generator

awg = ZIHDAWG8(name='HDAWG8', device_id='DEV8049')

grouping_1x8 = 2
awg.set_channel_grouping(grouping_1x8)

output1_marker1 = 4
awg.triggers_out_4_source(output1_marker1)

output2_marker1 = 6
awg.triggers_out_5_source(output2_marker1)

sampling_rate_293KHz = 13
awg.awgs_0_time(sampling_rate_293KHz)

# Initialize the lock-in amplifier

lockin = ZIUHFLI(name='ZIUHFLI', device_ID='DEV2338')

lockin.scope_trig_enable('ON')

lockin.scope_trig_signal('Trig Input 1')

lockin.scope_samplingrate('7.03 MHz')


# Initialize the virtual AWG

virtual_awg = VirtualAwg([awg], settings)

marker_delay = 5.49e-4
virtual_awg.digitizer_marker_delay(marker_delay)

marker_uptime = 1e-4
virtual_awg.digitizer_marker_uptime(marker_uptime)

# Create the station

station = Station(awg, lockin, virtual_awg)
station.gates = None

# Create and start the video mode

measure_channel = 1
video_mode = VideoMode(station, sweepparams='P1', sweepranges=100,
                       minstrument=[lockin, measure_channel], dorun=False)

#%% Execute the run or press the run button

video_mode.run()

#%%
awg.disable_channel(4)

