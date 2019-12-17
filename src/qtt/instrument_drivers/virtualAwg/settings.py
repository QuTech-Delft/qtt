from qcodes import Instrument, ManualParameter
from qcodes.utils.validators import Numbers


class SettingsInstrument(Instrument):
    """ Instrument that holds settings for the Virtual AWG """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self._awg_map = {}
        self._awg_gates = {}
        self._awg_markers = {}

    @property
    def awg_map(self):
        return self._awg_map

    @awg_map.setter
    def awg_map(self, value):
        self._awg_map = value

    @property
    def awg_gates(self):
        return self._awg_gates

    @awg_gates.setter
    def awg_gates(self, value):
        self._awg_gates = value

    @property
    def awg_markers(self):
        return self._awg_markers

    @awg_markers.setter
    def awg_markers(self, value):
        self._awg_markers = value

    def create_map(self):
        """ Adds default parameters based on known gates and markers """

        self._awg_map = {**self._awg_gates, **self._awg_markers}

        for awg_gate in self._awg_map:
            parameter_name = 'awg_to_{}'.format(awg_gate)
            parameter_label = '{} (factor)'.format(parameter_name)
            self.add_parameter(parameter_name, parameter_class=ManualParameter,
                               initial_value=1000, label=parameter_label, vals=Numbers(1, 1000))

    def write_raw(self, cmd: str) -> None:
        raise NotImplementedError()

    def ask_raw(self, cmd: str) -> str:
        raise NotImplementedError()
