import sys
sys.path.append("C:\\Program Files (x86)\\Keysight\\SD1\\Libraries\\Python")
import keysightSD1

import matplotlib.pyplot as plt
import numpy as np

from qcodes import Instrument, ManualParameter
from qcodes.utils.validators import Numbers
from qcodes.instrument_drivers.Spectrum.M4i import M4i
import qcodes.instrument_drivers.Keysight.M3201A as AWG


from qtt.gui.dataviewer import DataViewer
from qtt.measurements.videomode import VideoMode
from qtt.measurements.scans import measuresegment as measure_segment
from qtt.instrument_drivers.virtualAwg.hvi.KeysightM3601A import KeysightM3601A
from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer

#%%

class Hardware(Instrument):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.awg_map = {'B0': (0, 1), 'P1': (0, 2), 'B1': (0, 3), 'P2': (0, 4),
                        'B2': (1, 1), 'P3': (1, 2), 'B3': (1, 3), 'P4': (1, 4),
                        'B4': (2, 1), 'P5': (2, 2), 'B5': (2, 3), 'G1': (2, 4),
                        'I': (3, 1), 'Q': (3, 2), 'm4i_mk': (3, 3, 0)}

        for gate in self.awg_map.keys():
            p = 'awg_to_%s' % gate
            self.add_parameter(p, parameter_class=ManualParameter,
                               initial_value=1000,
                               label='{} (factor)'.format(p),
                               vals=Numbers(1, 1000))

def update_awg_settings(virtual_awg, prescaler, amplitude):
    for awg_number in range(len(virtual_awg.awgs)):
        virtual_awg.update_setting(awg_number, 'prescaler', prescaler)
        virtual_awg.update_setting(awg_number, 'amplitude', amplitude)

def plot_data_1d(digitizer_data, label_x_axis='', label_y_axis='', fig=100):
    plot_window = plt.figure(num=fig)
    plot_window.canvas.manager.window.activateWindow()
    plot_window.canvas.manager.window.raise_()
    plt.clf()
    plt.xlabel(label_x_axis)
    plt.ylabel(label_y_axis)
    plt.plot(digitizer_data.flatten(),'.b')
    plt.show()

def plot_data_2d(digitizer_data, label_x_axis='', label_y_axis='', label_colorbar='', fig=100):
    plot_window = plt.figure(num=fig)
    plot_window.canvas.manager.window.activateWindow()
    plot_window.canvas.manager.window.raise_()
    plt.figure(num=fig)
    plt.clf()
    im = plt.imshow(digitizer_data[0])
    cbar = plt.colorbar(im)
    plt.xlabel(label_x_axis)
    plt.ylabel(label_y_axis)
    cbar.ax.set_ylabel(label_colorbar)
    plt.show()


#%%
    
## DIGITIZER ##
digitizer = M4i(name='digitizer')

sample_rate_in_Hz = 1e7
digitizer.sample_rate(sample_rate_in_Hz)

timeout_in_ms = 10 * 1e3
digitizer.timeout(timeout_in_ms)

millivolt_range = 2000
digitizer.initialize_channels(mV_range=millivolt_range)

import pyspcm
external_clock_mode = pyspcm.SPC_CM_EXTREFCLOCK
digitizer.clock_mode(external_clock_mode)

reference_clock_10mHz = int(1e7)
digitizer.reference_clock(reference_clock_10mHz)

for ii in range(4):
    getattr(digitizer, 'termination_%d' %ii)(0)

# %%

## VIRUAL AWG ##

awg_2 = AWG.Keysight_M3201A("AWG_slot2", 1, 2)
awg_3 = AWG.Keysight_M3201A("AWG_slot3", 1, 3)
awg_4 = AWG.Keysight_M3201A("AWG_slot4", 1, 4)
awg_5 = AWG.Keysight_M3201A("AWG_slot5", 1, 5)

hardware = Hardware('hardware')
virtual_awg = KeysightM3601A([awg_2, awg_3, awg_4, awg_5], hardware)

#%%

uptime_in_seconds = 1e-7
marker_delay_in_sec = 2e-7
virtual_awg.update_digitizer_marker_settings(uptime_in_seconds, marker_delay_in_sec)

prescaler = 10
amplitude = 1.5
update_awg_settings(virtual_awg, prescaler, amplitude)

#%%

output_gate = 'B2'
mV_sweep_range = 50
sec_period = 1.0e-4
sweep_data = virtual_awg.sweep_gates({output_gate: 1}, mV_sweep_range, sec_period)

virtual_awg.enable_outputs([output_gate])
virtual_awg.run()

readout_channels = [0]
number_of_averages = 100
data = measure_segment(sweep_data, number_of_averages, digitizer, readout_channels)

virtual_awg.stop()
virtual_awg.disable_outputs([output_gate])

plot_data_1d(data, 'Digitizer Data Points [a.u.]', 'Amplitude [V]')

#%%


output_gates=['B0', 'B2']

gate_voltages = {'B0': [50, 0, 75, 0, 100, 0], 'B2': [0, 50, 0, 75, 0, 100]}
waiting_times = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])

sweep_data = virtual_awg.pulse_gates(gate_voltages, waiting_times)

virtual_awg.enable_outputs(output_gates)
virtual_awg.run()

readout_channels = [0, 3]
number_of_averages = 100
data = measure_segment(sweep_data, number_of_averages, digitizer, readout_channels, process=False)

virtual_awg.stop()
virtual_awg.disable_outputs(output_gates)

def plot_data_1d_x(digitizer_data, label_x_axis='', label_y_axis='', fig=100):
    plot_window = plt.figure(num=fig)
    plot_window.canvas.manager.window.activateWindow()
    plot_window.canvas.manager.window.raise_()
    plt.clf()
    plt.xlabel(label_x_axis)
    plt.ylabel(label_y_axis)
    for ii in range(digitizer_data.shape[0]):
        plt.plot(digitizer_data[ii,:],'.', label='%d' % ii)
    plt.show()
    plt.legend()

plot_data_1d_x(data, 'Digitizer Data Points [a.u.]', 'Amplitude [V]')

#%%

from qtt.simulation import virtual_dot_array

nr_dots = 3
station = virtual_dot_array.initialize(reinit=True, nr_dots=nr_dots, maxelectrons=2)
_ = station.add_component(virtual_awg)
_ = station.add_component(digitizer)


#%%

from qtt.measurements.scans import scan1Dfast, scanjob_t

scanjob = scanjob_t({'minstrument': [0],'Naverage':100,'wait_time_start_scan':2,
                     'sweepdata': {'param': 'B2','start': 100,'end': 250, 'step': 1,'wait_time': 0.28}})
dataset_measurement_fast = scan1Dfast(station, scanjob)


logviewer = DataViewer()
logviewer.show()


#%%


vm = VideoMode(station, 'B0', 50, minstrument=(digitizer.name,[3]), resolution = 50, diff_dir=[None, 'g'])
vm.stopreadout()
vm.updatebg()


#%%

vm = VideoMode(station, [{'B0': 1}, {'B2': 1}], [40, 40], minstrument=(digitizer.name, [0, 3]), resolution = [96, 96])
vm.stopreadout()
vm.updatebg()
