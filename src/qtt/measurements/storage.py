# %%
import logging
import os
import warnings
from datetime import datetime

import h5py
import qcodes

import qtt.instrument_drivers.virtual_gates
from qtt.utilities.tools import rdeprecated

try:
    import hickle
except ImportError:
    import qtt.exceptions
    warnings.warn('could not import hickle, not all functionality available',
                  qtt.exceptions.MissingOptionalPackageWarning)

# %%


@rdeprecated(txt='Method will be removed in a future version', expire='Jun 1 2022')
def list_states(verbose=1):
    """ List available states of the system

    Args:
        verbose (int)

    Returns:
        states (list): List of string tags

    See also:
        load_state
    """
    statefile = qcodes.config.get('statefile', None)
    if statefile is None:
        statefile = os.path.join(os.path.expanduser('~'), 'qtt_statefile.hdf5')
    if not os.path.exists(statefile):
        return []
    with h5py.File(statefile, 'r') as h5group:
        tags = list(h5group.keys())
    if verbose:
        print('states on system from file %s: ' % (statefile, ), end='')
        print(', '.join([str(x) for x in tags]))
    return tags

# %%


def _get_statefile(statefile=None):
    if statefile is None:
        statefile = qcodes.config.get('statefile', None)
    if statefile is None:
        statefile = os.path.join(os.path.expanduser('~'), 'qtt_statefile.hdf5')
    return statefile


@rdeprecated(txt='Method will be removed in a future version', expire='Jun 1 2022')
def load_state(tag=None, station=None, verbose=1, statefile=None):
    """ Load state of the system from disk

    Args:
        tag (str): identifier of state to load
        station (None or qcodes station): If defined apply the gatevalues loaded from disk
        verbose (int): verbosity level
        statefile (str): file with the state of the system

    Returns:
        state (dict): Dictionary with state of the system
        virtual_gates (None or object): reconstructed virtual gates

    """
    statefile = _get_statefile(statefile)
    if verbose:
        print('load_state %s: reading from file %s' % (tag, statefile))
    obj = {}
    with h5py.File(statefile, 'r') as h5group:
        if tag not in h5group:
            raise Exception('tag %s not in file' % tag)
        obj = hickle.load(h5group, path=tag)
    if station is not None:
        try:
            gv = obj.get('gatevalues')
            if gv is not None:
                if verbose >= 1:
                    print('load_state %s: resetting gate values' % tag)
                station.gates.resetgates(gv, gv, verbose=verbose >= 2)
        except Exception as ex:
            logging.exception(ex)
    virtual_gates = obj.get('virtual_gates', None)
    if virtual_gates is not None and station is not None:
        if verbose:
            print('load_state %s: creating virtual gate matrix' % tag)

        virtual_gates = qtt.instrument_drivers.virtual_gates.VirtualGates.from_dictionary(
            virtual_gates, station.gates)
    return obj, virtual_gates


@rdeprecated(txt='Method will be removed in a future version', expire='Jun 1 2022')
def save_state(station, tag=None, overwrite=False, virtual_gates=None, data=None, verbose=1, statefile=None):
    """ Save current state of the system to disk

    Args:
        station (qcodes station)
        tag (str or None)
        overwrite (bool): If True overwrite existing data, otherwise raise error
        virtual_gates (None or virtual_gates): virtual gates object to store
        data (None or object): optional extra data
        verbose (int): verbosity level
        statefile (str): file with the state of the system

    Example:
        save_state(station, tag='tripledot1')

    The data is written to an HDF5 file. The default location is the user
    home directory with name qtt_statefile.hdf5.

    To install hickle: pip install git+https://github.com/telegraphic/hickle.git@dev
    """
    statefile = _get_statefile(statefile)

    snapshot = station.snapshot()
    gv = station.gates.allvalues()

    date_string = f"{datetime.now():%Y%m%d-%H%M%S}"

    if verbose >= 2:
        print(date_string)
    if tag is None:
        tag = date_string

    obj = {'gatevalues': gv, 'snapshot': snapshot,
           'datestring': date_string, 'data': data}

    if virtual_gates is not None:
        obj['virtual_gates'] = virtual_gates.to_dictionary()
        # remove the redundant information that can't be serialized
        obj['virtual_gates'].pop('_crosscap_map', None)
        obj['virtual_gates'].pop('_crosscap_map_inv', None)

    if verbose:
        print('save_state: writing to file %s, tag %s' % (statefile, tag))
    with h5py.File(statefile, 'a') as h5group:
        # implementation using dev branch of hickle
        if tag in h5group:
            if overwrite:
                del h5group[tag]
            else:
                raise Exception(
                    'tag %s already exists in state file %s' % (tag, statefile))
        hickle.dump(obj, h5group, path=tag)
    return tag
