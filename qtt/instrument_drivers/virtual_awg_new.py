import numpy as np

from qcodes.utils.validators import Numbers
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import ManualParameter

from qctoolkit.pulses import SequencePT, TablePT, FunctionPT
from qctoolkit.pulses.plotting import plot, render
from qctoolkit.pulses.sequencing import Sequencer


# -----------------------------------------------------------------------------
# qc-toolkit template pulses... separate file?


def pulsewave_template(name: str='pulse'):
    return TablePT({name: [(0, 'amplitude'), ('width', 0), ('holdtime', 0)]})


def sawtooth_template(name: str='sawtooth'):
    return TablePT({name: [(0,0), ('period/4', 'amplitude', 'linear'),
                           ('period*3/4', '-amplitude', 'linear'), 
                           ('period',0, 'linear')]})


def wait_template(name: str='wait'):
    return TablePT({name: [(0, 0), ('holdtime', 0)]})


def to_array(sequence: SequencePT, sample_rate:int):
    sequencer = Sequencer()
    sequencer.push(sequence)
    build = sequencer.build()
    if not sequencer.has_finished():
        raise ValueError
    return render(build, sample_rate)[1]

# -----------------------------------------------------------------------------
# qc-toolkit stuff... separate file?


def make_sawtooth(vpp, period, width, reps=1):
    values = {'period':period, 'amplitude':vpp/2}
    data = (sawtooth_template(), values)
    return SequencePT(*(data,)*reps)


def make_pulses(voltages, waittimes, filter_cutoff=None, mvrange=None):
    sequence = []
    return sequence


def sequence_to_waveform(sequence: SequencePT, sample_rate:int):
    '''generates waveform from sequence'''
    sequencer = Sequencer()
    sequencer.push(sequence)
    build = sequencer.build()
    if not sequencer.has_finished():
        raise ValueError
    voltages = render(build, sample_rate)[1]
    return voltages[next(iter(voltages))] # ugly!!!


def plot_waveform(waveform, sample_rate:int):
    '''plots the waveform array.'''
    from matplotlib import pyplot as plt
    sample_count = len(waveform)
    total_time = sample_count / sample_rate
    times = np.linspace(0, total_time, num=sample_count)
    plt.plot(times, waveform)


def plot_sequence(sequence, sample_rate):
    '''plots the qc-toolkit sequence.'''
    plot(sequence, sample_rate)


'''
s = make_sawtooth(1, 1, 1, 2)
plot(s, sample_rate=1e4)
d = sequence_to_waveform(s, sample_rate=1e4)
'''

# -----------------------------------------------------------------------------
# AWG stuff...


class Hardware(InstrumentBase):

    def __init__(self, name: str, awgs: list, gates: dict, plungers: dict,
                 **kwargs):
        super().__init__(self, name, **kwargs)
        self.__vrange = Numbers(0, 400)
        self._plungers = plungers
        self._gates = gates
        self._awgs = awgs
        self.__set_parameters()

    def __set_parameters(self):
        for gate in self.gates:
            gate_name = 'awg_to_{0}'.format(gate)
            gate_label = '{0} (factor)'.format(gate_name)
            self.add_parameter(gate_name, parameter_class=ManualParameter,
                               initial_value=0, label=gate_label,
                               vals=self.__vrange)

# -----------------------------------------------------------------------------

class VirtualAwgBase():

    clock_speed = 1e8
    awg_delay = 0
    channels = 4
    markers = 2

    def set_sequence_mode(self):
        raise NotImplementedError

    def delete_all_waveforms(self):
        raise NotImplementedError

    def prepare_run(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

# -----------------------------------------------------------------------------


class virtual_awg(InstrumentBase):

    def __init__(self, name, hardware: Hardware, verbose=1, **kwargs):
        super().__init__(name, **kwargs)
        pass

    #def get_idn(self): #  InstumentBase not needed anymore?
    #    '''Overrule default because the VISA command does not work.'''
    #    return {}

    def set_awg_number(self):
        pass

    def awg_gate(self, gate):
        pass

    def reset_awgs(self):
        pass

    def stop_awgs(self, verbose=0):
        ''' Stops all AWGs and turns of all channels.'''
        pass

    def sweep_init(self, waveforms, period=1e-3, delete=True, samp_freq=None):
        '''Returns sweep_info dict.'''
        pass

    def sweep_run(self, sweep_info):
        '''Activate AWGs with channels for the sweeps.'''
        pass

    def sweep_process(self, data, waveform, Naverage=1, direction='forwards', start_offset=1):
        '''Process the returned data using shape of the sawtooth send with the AWG.'''
        pass

    # FIXME: make use if sweep_gate_virt
    def sweep_gate(self, gate, sweeprange, period, width=.95, wave_name=None, delete=True):
        pass

    def sweep_gate_virt(self, gate_comb, sweeprange, period, width=.95, delete=True):
        pass

    # should be replaced by adding waveforms in the qc_toolkit framework?
    @qtt.tools.deprecated
    def sweepandpulse_gate(self, sweepdata, pulsedata, wave_name=None, delete=True):
        pass

    # FIXME: make use of sweep_2D_virt
    # JP: I think FPGA exceptions should not be handled by awg
    #        if resolution[0] * resolution[1] > self.maxdatapts:
    #            raise Exception('resolution is set higher than FPGA memory allows')
    def sweep_2D(self, samp_freq, sweepgates, sweepranges, resolution, width=.95, comp=None, delete=True):
        pass

    def sweep_2D_virt(self, samp_freq, gates_horz, gates_vert, sweepranges, resolution, width=.95, delete=True):
        pass

    def sweep_2D_process(self, data, waveform, diff_dir=None):
        pass

    def pulse_gates(self, gate_voltages, waittimes, filtercutoff=None, delete=True):
        pass

    # FIXME: keep?
    def set_amplitude(self, amplitude):
        pass

    def check_amplitude(self, gate, mvrange):
        pass

    def __check_frequency_waveform(self, period, width):
        pass

# -----------------------------------------------------------------------------
