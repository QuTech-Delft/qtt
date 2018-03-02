import unittest

from qcodes.utils.validators import Numbers
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import ManualParameter
from qtt.instrument_drivers.virtual_awg_new import *

# -----------------------------------------------------------------------------


class Parameters(InstrumentBase):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        awg_gates = ['X1', 'X2', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
        for gate in awg_gates:
            p = 'awg_to_%s' % gate
            self.add_parameter(p, parameter_class=ManualParameter,
                               initial_value=0,
                               label='{} (factor)'.format(p),
                               vals=Numbers(0, 400))

        # coaxes to gates: (awg_nr, channel_nr)
        # markers to instruments: (awg_nr, channel_nr, marker_nr)
        self.awg_map = {'X2': (0, 1), 'P7': (0, 2), 'P6': (0, 3), 'P5': (0, 4),
                        'P2': (1, 1), 'X1': (1, 2), 'P3': (1, 3), 'P4': (1, 4),
                        'dig_mk': (0, 4, 1), 'awg_mk': (0, 4, 2)}

        filterboxes = [1, 2, 3]
        for box_number in filterboxes:
            p = 'filterbox_{}'.format(box_number)
            self.add_parameter(p, parameter_class=ManualParameter,
                               initial_value=0,
                               label='{} (frequency)'.format(p),
                               unit='Hz',
                               vals=Numbers(0, 500e3))

        md = 'marker_delay'
        self.add_parameter(md, parameter_class=ManualParameter,
                           initial_value=0,
                           label='{} (time)'.format(md),
                           unit='s',
                           vals=Numbers(0, 1))

        cs = 'clock_speed'
        self.add_parameter(cs, parameter_class=ManualParameter,
                           initial_value=1e8,
                           label='{} (sample rate)'.format(cs),
                           unit='samples per second',
                           vals=Numbers(0, 1e10))

        ca = 'channel_amplitudes'
        self.add_parameter(ca, parameter_class=ManualParameter,
                           initial_value=4.0,
                           label='{} (amplitude)'.format(ca),
                           unit='Volt',
                           vals=Numbers(0.02, 4.5))


# -----------------------------------------------------------------------------


class Test_VirtualAwg(unittest.TestCase):

    def setUp(self):
        sname = 'test'
        awgA = Tektronix_AWG5014(name='awgA', server_name=sname)
        awgB = Tektronix_AWG5014(name='awgB', server_name=sname)
        digitizer = M4i(name='digitizer', server_name=sname)
        self.intstruments = [awgA, awgB, digitizer]
        self.parameters = Parameters('test')

    def test_create_hardware(self):
        instruments = self.intstruments
        parameters = self.parameters
        hardware = VirtualAwg('test', instruments, parameters)

# -----------------------------------------------------------------------------