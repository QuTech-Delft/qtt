""" Functionality to store instrument data in a configuration file """
import configparser
import json
import numbers

import numpy as np

import qtt.utilities.tools
from qtt.utilities.tools import rdeprecated


@rdeprecated(txt='This method will be removed in a future release', expire='Sep 1 2021')
def save_instrument_configparser(instr, ifile, verbose=1):
    """ Save instrument configuration to configparser structure

    Args:
        instr (Instrument): instrument to apply settings to
        ifile (str): configuration file
    """
    jdict = configparser.ConfigParser()
    jdict.read(ifile)
    if not instr.name in jdict:
        jdict.add_section(instr.name)
    for pname, p in instr.parameters.items():
        if not hasattr(p, 'set'):
            continue
        if verbose:
            print('%s: store %s' % (instr.name, pname))
        val = p.get()
        dd = '%s/%s' % (instr.name, pname)
        jdict[instr.name][pname] = str(val)

    with open(ifile, 'w') as fid:
        jdict.write(fid)


@rdeprecated(txt='This method will be removed in a future release', expire='Sep 1 2021')
def load_instrument_configparser(instr, ifile, verbose=1):
    """ Load instrument configuration from configparser structure

    Args:
        instr (Instrument): instrument to apply settings to
        ifile (str): configuration file
    """
    jdict = configparser.ConfigParser()
    jdict.read(ifile)
    for pname, p in instr.parameters.items():
        if not hasattr(p, 'set'):
            continue
        if verbose:
            print('%s: load %s' % (instr.name, pname))
        try:
            val = jdict[instr.name][pname]
            v = p.get()
            if isinstance(v, numbers.Number):
                p.set(float(val))
            else:
                p.set(val)
        except:
            if verbose:
                print('%s: load %s: no entry?' % (instr.name, pname))
            pass
    return jdict


load_instrument = load_instrument_configparser
save_instrument = save_instrument_configparser

# %% Testing

if __name__ == '__main__':
    import os

    from stationV2.tools import V2hardware
    v2hardware = V2hardware(name='v2hardware', server_name=None)

    datadir = '/tmp/qdata/'
    ifile = os.path.join(datadir, 'instrument_settings.txt')

    save_instrument(v2hardware, ifile)
    load_instrument(v2hardware, ifile)
