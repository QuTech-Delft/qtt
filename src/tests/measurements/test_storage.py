# %%
import qtt.simulation.virtual_dot_array
import tempfile
from qtt.instrument_drivers.virtual_gates import virtual_gates
from qtt.measurements.storage import save_state, load_state
from unittest import TestCase
# %%


class TestStorage(TestCase):

    def test_storage(self):
        station = qtt.simulation.virtual_dot_array.initialize(reinit=True, nr_dots=2, maxelectrons=2, verbose=0)
        v_gates = virtual_gates('virtual_gates_load_save_state', station.gates, {'vP1': {'P1': 1, 'P2': .1},
                                                                                 'vP2': {'P1': .2, 'P2': 1.}})

        before_crosscap_map = v_gates.get_crosscap_map()
        before_crosscap_map_inv = v_gates.get_crosscap_map_inv()

        tmp_file = tempfile.mktemp()
        tag = save_state(station, virtual_gates=v_gates, statefile=tmp_file)
        _, virtual_gates_loaded = load_state(station=station, tag=tag, verbose=1, statefile=tmp_file)

        after_crosscap_map = virtual_gates_loaded.get_crosscap_map()
        after_crosscap_map_inv = virtual_gates_loaded.get_crosscap_map_inv()

        self.assertDictEqual(before_crosscap_map, after_crosscap_map)
        self.assertDictEqual(before_crosscap_map_inv, after_crosscap_map_inv)

        virtual_gates_loaded.close()
        v_gates.close()
