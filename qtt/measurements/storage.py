#%%
from datetime import datetime
import qcodes
import os
import h5py
import logging
import warnings
import pandas as pd
import numpy as np

try:
    import hickle
except:
    import qtt.exceptions
    warnings.warn('could not import hickle, not all functionality available',
                  qtt.exceptions.MissingOptionalPackageWarning)

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
        statefile = os.path.join(os.path.expanduser('~'), 'qtt_statefile.hdf5')
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
        statefile = os.path.join(os.path.expanduser('~'), 'qtt_statefile.hdf5')
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


def save_state(station, tag=None, overwrite=False, data=None, verbose=1):
    """ Save current state of the system to disk

    Args:
        station (qcodes station)
        tag (str or None)
        overwrite (bool): If True overwrite existing data, otherwise raise error
        data (None or object): optional extra data
        verbose (int)

    Example:
        >>> save_state(station, tag='tripledot1')    

    The data is written to an HDF5 file. The default location is the user
    home directory with name qtt_statefile.hdf5.
    
    To install hickle: pip install git+https://github.com/telegraphic/hickle.git@dev
    """
    statefile = qcodes.config.get('statefile', None)
    if statefile is None:
        statefile = os.path.join(os.path.expanduser('~'), 'qtt_statefile.hdf5')

    snapshot = station.snapshot()
    gv = station.gates.allvalues()

    datestring = "{:%Y%m%d-%H%M%S}".format(datetime.now());

    if verbose >= 2:
        print(datestring)
    if tag is None:
        tag = datestring

    obj = {'gatevalues': gv, 'snapshot': snapshot,
           'datestring': datestring, 'data': data}

    if verbose:
        print('save_state: writing to file %s, tag %s' % (statefile, tag))
    with h5py.File(statefile, 'a') as h5group:
            # implementation using dev branch of hickle
            if (tag in h5group):
                if overwrite:
                    del h5group[tag]
                else:
                    raise Exception(
                        'tag %s already exists in state file %s' % (tag, statefile))
            hickle.dump(obj, h5group, path=tag)

#%% Logging and monitoring functions


def retrieve_logdata(filename, tag='metadata'):
    """ Retrieve logged data from a HDFStore

    Args:
        filename (str or HDFStore)
        tag (str): key of table append to

    Returns:
        df (pandas dataframe)

    """
    if isinstance(filename, str):
        store = pd.HDFStore(filename)
    else:
        store = filename
    d = store.select(tag)
    d.reset_index(inplace=True, drop=True)
    return d


def store_logdata(datadict, filename, tag='metadata'):
    """ Log data to a HDFStore

    Args:
        datadict (dict): dictionary with names and values or functions
        filename (str or HDFStore)
        tag (str): key of table append to

    """

    if isinstance(filename, str):
        store = pd.HDFStore(filename)
        closestore = True
    else:
        store = filename
        closestore = False
    names = sorted(datadict.keys())

    if tag in store:
        for n in names:
            if not n in store[tag]:
                warnings.warn('column %s does not exist in store' % n)
                names.remove(n)

    def ev(p):
        if callable(p):
            return p()
        else:
            return p

    data = [ev(datadict[n]) for n in names]
    df = pd.DataFrame([data], columns=names)

    store.append(tag, df)
    
    if closestore:
        # to prevent memory leaks in loops
        store.close()
        import gc
        gc.collect()

    return data