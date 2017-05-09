""" Functionality to store instrument data in a configuration file """
import numpy as np
import json
import numbers


def save_instrument_json(instr, ifile, verbose=1):
    jdict = dict()
    for pname, p in instr.parameters.items():
        if not p.has_set:
            continue
        if verbose:
            print('%s: store %s' % (instr.name, pname))
        val = p.get()
        dd = '%s/%s' % (instr.name, pname)
        jdict[dd] = val

    json.dump(jdict, open(ifile, 'w'), sort_keys=True, indent=4)


def load_instrument_json(instr, ifile, verbose=1):
    with open(ifile, 'r') as fid:
        jdict = json.load(fid)
    for pname, p in instr.parameters.items():
        if not p.has_set:
            continue
        if verbose:
            print('%s: load %s' % (instr.name, pname))
        dd = '%s/%s' % (instr.name, pname)
        val = jdict[dd]
        p.set(val)
    return jdict

import configparser


def save_instrument_configparser(instr, ifile, verbose=1):
    jdict = configparser.ConfigParser()
    jdict.read(ifile)
    if not instr.name in jdict:
        jdict.add_section(instr.name)
    for pname, p in instr.parameters.items():
        if not p.has_set:
            continue
        if verbose:
            print('%s: store %s' % (instr.name, pname))
        val = p.get()
        dd = '%s/%s' % (instr.name, pname)
        jdict[instr.name][pname] = str(val)

    with open(ifile, 'w') as fid:
        jdict.write(fid)


def load_instrument_configparser(instr, ifile, verbose=1):
    jdict = configparser.ConfigParser()
    jdict.read(ifile)
    for pname, p in instr.parameters.items():
        if not p.has_set:
            continue
        if verbose:
            print('%s: load %s' % (instr.name, pname))
        # dd='%s/%s' % (instr.name, pname )
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

#%% Testing

if __name__ == '__main__':
    import os
    from stationV2.tools import V2hardware
    v2hardware = V2hardware(name='v2hardware', server_name=None)

    datadir = '/tmp/qdata/'
    ifile = os.path.join(datadir, 'instrument_settings.txt')

    save_instrument(v2hardware, ifile)
    load_instrument(v2hardware, ifile)

    import configparser
    cc = configparser.ConfigParser()
