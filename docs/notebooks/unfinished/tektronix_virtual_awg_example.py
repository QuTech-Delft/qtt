import matplotlib.pyplot as plt
import numpy as np

from qcodes import Instrument, ManualParameter
from qcodes.utils.validators import Numbers
from qcodes.instrument_drivers.Spectrum.M4i import M4i
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014

from qtt.gui.dataviewer import DataViewer
from qtt.measurements.videomode import VideoMode
from qtt.measurements.scans import measuresegment as measure_segment
from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg
from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer


# ch1 awg ==> C1 ==> ch1 m4i
# ch2 awg ==> C1 ==> ch3 m4i

#%%

class Hardware(Instrument):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.awg_map = {'C1': (0, 1), 'C2': (0, 2), 'C3': (0, 3), 'C4' : (0, 4),
                         'm4i_mk': (0, 4, 1)}

        for gate in self.awg_map.keys():
            p = 'awg_to_%s' % gate
            self.add_parameter(p, parameter_class=ManualParameter,
                               initial_value=1000,
                               label='{} (factor)'.format(p),
                               vals=Numbers(1, 1000))


def update_awg_settings(virtual_awg, amplitude, sample_rate):
    for awg_number in range(len(virtual_awg.awgs)):
        virtual_awg.awgs[awg_number].update_sampling_rate(sample_rate)
        virtual_awg.update_setting(awg_number, 'amplitudes', amplitude)


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

address =  'TCPIP0::192.168.137.13::INSTR'
awg_1 = Tektronix_AWG5014(name='awg1', address=address)

hardware = Hardware('hardware')

import qtt.instrument_drivers.virtual_awg
 
if 1:
    virtual_awg = VirtualAwg([awg_1], hardware) # new style
else:
    virtual_awg = qtt.instrument_drivers.virtual_awg.virtual_awg('awg', instruments=[awg_1], awg_map=hardware.awg_map, hardware=hardware)
    #station.add_component(virtual_awg)
    
#%%

uptime_in_seconds = 1e-7
marker_delay_in_sec = 3e-7
virtual_awg.update_digitizer_marker_settings(uptime_in_seconds, marker_delay_in_sec)

amplitude = 4
sample_rate = 1e7
update_awg_settings(virtual_awg, amplitude, sample_rate)


#%%

output_gate = 'C1'
mV_sweep_range = 50
sec_period = 1.0e-4
sweep_data = virtual_awg.sweep_gates({output_gate: 1}, mV_sweep_range, sec_period)

virtual_awg.enable_outputs([output_gate])
virtual_awg.run()

readout_channels = [3]
number_of_averages = 100
data = measure_segment(sweep_data, number_of_averages, digitizer, readout_channels)

virtual_awg.stop()
virtual_awg.disable_outputs([output_gate])

plot_data_1d(data, 'Digitizer Data Points [a.u.]', 'Amplitude [V]')

#%%


output_gates=['C1', 'C2']

gate_voltages = {'C1': [50, 0, 75, 0, 100, 0], 'C2': [0, 50, 0, 75, 0, 100]}
waiting_times = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])

sweep_data = virtual_awg.pulse_gates(gate_voltages, waiting_times)

virtual_awg.enable_outputs(output_gates)
virtual_awg.run()

readout_channels = [1, 3]
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

from qtt.instrument_drivers.gates import VirtualDAC
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qcodes.station import Station

ivvi = VirtualIVVI(name='ivvi', model=None)
gates = VirtualDAC('gates', [ivvi], {'C1': (0, 1), 'C2': (0, 2)})

station = Station(ivvi, gates, virtual_awg, awg_1, digitizer)


#%%

from qtt.measurements.scans import scan1Dfast, scanjob_t

scanjob = scanjob_t({'minstrument': [1],'Naverage':100,'wait_time_start_scan':2,
                     'sweepdata': {'param': 'C2','start': 100,'end': 250, 'step': 1,'wait_time': 0.28}})
dataset_measurement_fast = scan1Dfast(station, scanjob)


logviewer = DataViewer()
logviewer.show()


#%%

from qtt.measurements.scans import scan2Dturbo

gate1 = 'C1'
gate2 = 'C2'

scanjob = scanjob_t()

scanjob['sweepdata'] = {'param': gate1, 'range': 100}
scanjob['stepdata'] = {'param': gate2, 'range': 100}

scanjob['minstrumenthandle'] = (digitizer.name)
scanjob['minstrument'] = [3]

scanjob['wait_time_startscan'] = 2
scanjob['Naverage'] = 200

alldata, _, _ = scan2Dturbo(station, scanjob)


#%%

vm = VideoMode(station, 'C1', 50, minstrument=(digitizer.name,[3]), resolution = 50, diff_dir=[None, 'g'])
vm.stopreadout()
vm.updatebg()


#%%

virtual_awg.delay_FPGA=-3e-6

vm = VideoMode(station, ['C1','C2'], [40, 40], minstrument=(digitizer.name, [1, 3]), resolution = [96, 96])
vm.stopreadout()

vm.updatebg()