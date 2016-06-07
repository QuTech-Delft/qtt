# set up the qtt namespace
# flake8: noqa (we don't need the "<...> imported but unused" error)

import numpy as np

import qtpy
import matplotlib
try:
    print(qtpy.API_NAME)
    if qtpy.API_NAME=='PyQt4 (API v2)':
        matplotlib.use('Qt4Agg')
except:
    pass
import qtt.live
import qtt.tools
from qtt.tools import *
from qtt.data import *

from qtt.algorithms import *
