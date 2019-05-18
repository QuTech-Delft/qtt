import unittest
import numpy as np
from collections import OrderedDict
from qtt.instrument_drivers.virtual_gates import virtual_gates, extend_virtual_gates, update_cc_matrix


class TestVirtualGates(unittest.TestCase):

    def test_virtual_gates(self, verbose=0):
        """ Test for virtual gates object """
        from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
        from qtt.measurements.scans import instrumentName
        import pickle

        gates = VirtualIVVI(name=instrumentName('testivvi'),
                            model=None, gates=['P1', 'P2', 'P3', 'P4'])

        crosscap_map = OrderedDict((
            ('VP1', OrderedDict((('P1', 1), ('P2', 0.6), ('P3', 0)))),
            ('VP2', OrderedDict((('P1', 0.3), ('P2', 1), ('P3', 0.3)))),
            ('VP3', OrderedDict((('P1', 0), ('P2', 0), ('P3', 1))))
        ))
        vgates = virtual_gates(instrumentName('testvgates'), gates, crosscap_map)

        vp1 = vgates.VP1()
        if verbose:
            print('before set: VP1 {}'.format(vp1))
        vgates.VP1.set(10)
        vp1 = vgates.VP1()
        if verbose:
            print('after set: VP1 {}'.format(vp1))
        vgates.VP1.set(10)
        vp1 = vgates.VP1()
        if verbose:
            print('after second set: VP1 {}'.format(vp1))

        vgates_matrix = vgates.convert_map_to_matrix(crosscap_map)
        _ = vgates.convert_matrix_to_map(vgates_matrix)

        vgates.multi_set({'VP1': 10, 'VP2': 20, 'VP3': 30})
        all_values = vgates.allvalues()
        self.assertTrue(isinstance(all_values, dict))

        crosscap_matrix = vgates.get_crosscap_matrix()
        self.assertEqual(1.0, crosscap_matrix[0][0])
        self.assertEqual(0.6, crosscap_matrix[0][1])

        vgates.set_distances(1.0 / np.arange(1, 5))
        _ = vgates.to_dictionary()
        pickled_virtual_gates = pickle.dumps(vgates)
        pickle.loads(pickled_virtual_gates)

        v_gates = vgates.vgates() + ['vP4']
        p_gates = vgates.pgates() + ['P4']
        extended_vgates = extend_virtual_gates(v_gates, p_gates, vgates, name='vgates')
        if verbose:
            extended_vgates.print_matrix()

        extended_vgates.close()

        newvg, _, _ = update_cc_matrix(vgates, update_cc=np.eye(3), verbose=0)
        newvg.close()

        update_matrix = 0.1 * np.random.rand(3, 3)
        np.fill_diagonal(update_matrix, 1)

        # test normalization of virtual gate matrix
        extended_vgates, _, _ = update_cc_matrix(vgates, update_cc=update_matrix, verbose=0)
        np.testing.assert_almost_equal(extended_vgates.get_crosscap_matrix(),
                                       update_matrix.dot(vgates.get_crosscap_matrix()))

        # test normalization of virtual gate matrix
        serialized_matrix = extended_vgates.get_crosscap_matrix()
        extended_vgates.normalize_matrix()
        crosscap_matrix = extended_vgates.get_crosscap_matrix()
        for row in range(serialized_matrix.shape[0]):
            np.testing.assert_almost_equal(serialized_matrix[row] / serialized_matrix[row][row], crosscap_matrix[row])
        cc_matrix_diagonal = crosscap_matrix.diagonal()
        np.testing.assert_almost_equal(cc_matrix_diagonal, 1.)

        vgates.close()
        extended_vgates.close()
        gates.close()

    def test_virtual_gates_serialization(self):
        """ Test for virtual gates object """
        import qtt.instrument_drivers.virtual_instruments
        gates = qtt.instrument_drivers.virtual_instruments.VirtualIVVI(
            name=qtt.measurements.scans.instrumentName('ivvi_dummy_serialization_test'), model=None,
            gates=['P1', 'P2', 'P3', 'P4'])

        crosscap_map = OrderedDict((
            ('VP1', OrderedDict((('P1', 1), ('P2', 0.6), ('P3', 0)))),
            ('VP2', OrderedDict((('P1', 0.3), ('P2', 1), ('P3', 0.3)))),
            ('VP3', OrderedDict((('P1', 0), ('P2', 0), ('P3', 1))))
        ))
        virts = virtual_gates(qtt.measurements.scans.instrumentName('testvgates'), gates, crosscap_map)
        vgdict = virts.to_dictionary()

        vx = virtual_gates.from_dictionary(vgdict, gates, name=qtt.measurements.scans.instrumentName('vgdummy'))

        np.testing.assert_almost_equal(vx.get_crosscap_matrix_inv(), virts.get_crosscap_matrix_inv())
        self.assertTrue(vx.pgates() == ['P%d' % i for i in range(1, 4)])

        vx.close()
        gates.close()
        virts.close()
