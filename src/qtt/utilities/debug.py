# %% Tools for debugging code

import os
import functools
import tempfile
import time
import datetime

# %% Debugging


def dumpstring(txt, tag='dump', showfile=False):
    """ Dump a text string to temperary file on disk """
    _ = tempfile.mkdtemp()  # init
    tdir = tempfile.tempdir
    if tdir is None:
        tdir = tempfile.mkdtemp()
    tfile = os.path.join(tdir, 'qtt-%s.txt' % tag)
    dumpstring.tfile = tfile  # store temporary file in object
    with open(tfile, 'a+t') as fid:
        fid.write(txt + '\n')

# %% Decorator for generic function


def functioncalldecorator(f, name=None):
    """ Decorate a function to log input and output arguments """
    if name is None:
        try:
            # try to get name from parent class
            c = f.__self__
            name = c.name
        except:
            name = 'none'

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        dstr = str(datetime.datetime.now())
        clock = time.perf_counter()
        r = f(*args, **kwargs)
        ss = 'function %s: %.6f, %s\n' % (name, clock, dstr) + 'function %s: arguments %s, %s' % (name, args, kwargs)
        dumpstring(ss + '\n' + 'function %s: output %s\n' % (name, r, ), tag='functionlog')
        #print ('method %s: output %s' % (f, r) )
        return r
    return wrapped

# %%


def logInstrument(instrument):
    """ Decorate all parameters of an instrument with logging methods """
    for k in instrument.parameters:
        p = instrument.parameters[k]
        if hasattr(p, 'get'):
            print('decorate %s' % p)
            p.get = functioncalldecorator(p.get, '%s.get' % p.name)
        if hasattr(p, 'get') and hasattr(p, 'set'):
            p.set = functioncalldecorator(p.set, '%s.set' % p.name)
