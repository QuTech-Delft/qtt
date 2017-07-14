#%%
import hickle
from datetime import datetime
import qcodes
import os
import h5py
import logging

#%%


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
        statefile = os.path.join(os.path.expanduser('~'), 'statefile.hdf5')
    tags = []
    with h5py.File(statefile, 'r') as h5group:
        tags = list(h5group.keys())
    if verbose:
        print('states on system from file %s: ' % (statefile, ), end='')
        print(', '.join([str(x) for x in tags]))
    return tags

#%%


def load_state(tag=None, station=None, verbose=1):
    """ Load state of the system from disk

    Args:
        tag (str)
        station (None or qcodes station): If defined apply the gatevalues loaded from disk
        verbose (int)

    Returns:
        state (dict): Dictionary with state of the system

    """
    statefile = qcodes.config.get('statefile', None)
    if statefile is None:
        statefile = os.path.join(os.path.expanduser('~'), 'statefile.hdf5')
    if verbose:
        print('load_state: reading from file %s, tag %s' % (statefile, tag))
    obj = {}
    with h5py.File(statefile, 'r') as h5group:
        if not tag in h5group:
            raise Exception('tag %s not in file' % tag)
        obj = hickle.load(h5group, path=tag)
    if station is not None:
        try:
            gv = obj.get('gatevalues')
            if gv is not None:
                if verbose >= 1:
                    print('load_state: resetting gate values')
                station.gates.resetgates(gv, gv, verbose=verbose >= 2)
        except Exception as ex:
            logging.exception(ex)
    return obj


def save_state(station, tag=None, overwrite=False, verbose=1):
    """ Save current state of the system to disk

    Args:
        station (qcodes station)
        tag (str or None)
        overwrite (bool)
        verbose (int)

    Example:
        >>> save_state(station, tag='tripledot1')    

    To install hickle: pip install git+https://github.com/telegraphic/hickle.git@dev
    """
    statefile = qcodes.config.get('statefile', None)
    if statefile is None:
        statefile = os.path.join(os.path.expanduser('~'), 'statefile.hdf5')

    snapshot = station.snapshot()
    gv = station.gates.allvalues()

    datestring = "{:%Y%m%d-%H%M%S}".format(datetime.now());

    if verbose >= 2:
        print(datestring)
    if tag is None:
        tag = datestring

    obj = {'gatevalues': gv, 'snapshot': snapshot, 'datestring': datestring}

    if verbose:
        print('save_state: writing to file %s, tag %s' % (statefile, tag))
    with h5py.File(statefile, 'a') as h5group:
        if 0:
            # implementation with qcodes formatter
            f = qcodes.data.hdf5_format.HDF5Format()
            metadata_group = h5group.create_group('metadata')
            f.write_dict_to_hdf5(obj, metadata_group)

        else:
            # implementation using dev branch of hickle
            if (tag in h5group):
                if overwrite:
                    del h5group[tag]
                else:
                    raise Exception(
                        'tag %s already exists in state file %s' % (tag, statefile))
            hickle.dump(obj, h5group, path=tag)
