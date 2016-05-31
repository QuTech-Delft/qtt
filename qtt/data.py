import qtpy
#print(qtpy.API_NAME)

import numpy as np
import scipy
import os
import sys
import copy
import logging
import time
import qcodes
import qcodes as qc
import datetime

import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets

import matplotlib.pyplot as plt

from qtt.tools import tilefigs


#%%

#%%

def pix2scan(pt, dd2d):
    """ Convert pixels coordinates to scan coordinates (mV)
    Arguments
    ---------
    pt : array
        points in pixel coordinates
    dd2d : dictionary
        contains scan data

    """
    extent, g0,g1,vstep, vsweep, arrayname=dataset2Dmetadata(dd2d, array=None, verbose=0)
    #xx, _, _, zdata = get2Ddata(dd2d, verbose=0, fig=None)
    nx = vsweep.size #  zdata.shape[0]
    ny = vstep.size # zdata.shape[1]

    xx = extent
    x = pt
    nn = pt.shape[1]
    ptx = np.zeros((2, nn))
    ptx[1, :] = np.interp(x[1, :], [0, ny - 1], [xx[3], xx[2]])    # step
    ptx[0, :] = np.interp(x[0, :], [0, nx - 1], [xx[0], xx[1]])    # sweep
    return ptx
    
def dataset2Dmetadata(alldata, array=None, verbose=1):    
    if array is None:
        array = 'amplitude'
   
    A = alldata.arrays[array]


    g0=A.set_arrays[0].name
    g1=A.set_arrays[1].name
    vstep = np.array(A.set_arrays[0])
    vsweep = np.array(A.set_arrays[1])[0]
    extent = [vstep[0], vstep[-1], vsweep[0], vsweep[-1]]
    
    print('2D scan: gates %s %s' % (g0, g1))
    return extent, g0, g1, vstep, vsweep, array

   
if __name__=='__main__':
    extent, g0,g1,vstep, vsweep, arrayname=dataset2Dmetadata(alldata, array=None)
    _=pix2scan( np.zeros( (2,4) ), alldata )
    
#%%


def loadDataset(path):
    ''' Wrapper function

    :param path: filename without extension    
    '''
    dataset = qcodes.load_data(path)
    return dataset
    
def writeDataset(path, dataset):
    ''' Wrapper function

    :param path: filename without extension    
    '''
    dataset.write(path=path)
    dataset.save_metadata(path=path)

    