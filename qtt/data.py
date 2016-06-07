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

from .tools import tilefigs, diffImageSmooth
#from .algorithms import *

#%%
def dataset2image(dataset):
    """ Extract image from dataset """
    extentscan, g0,g2,vstep, vsweep, arrayname=dataset2Dmetadata(dataset, verbose=0, array=None)
    tr = image_transform(dataset)
    im=None
    impixel  = None
    if arrayname is not None:
        im = dataset.arrays[arrayname]
        impixel = tr.transform(im)

    return im, impixel, tr

#%%
def show2D(dd, im=None, fig=101, verbose=1, dy=None, sigma=None, colorbar=False, title=None, midx=2, units=None):
    """ Show result of a 2D scan """
    if dd is None:
        return None

    extent, g0, g1, vstep, vsweep, arrayname = dataset2Dmetadata(dd)
    #arrayname=plot_parameter(dataset)
    array=getattr(dd, arrayname)       
    if im is None:
        im = array
    else:
        pass
    XX = array # dd['data_array']
    
    xx=extent   
    #vstep = np.unique(XX[:, 1])
    #vsweep = np.unique(XX[:, 0])
    #xx, xdata, ydata, im = get2Ddata(
    #    dd, fastscan=False, verbose=0, midx=midx, fig=None)
    ny = vstep.size
    nx = vsweep.size

    #im=diffImage(im, dy)
    im = diffImageSmooth(im, dy=dy, sigma=sigma)

    if verbose:
        print('show2D: nx %d, ny %d, XX.shape %s' % (nx, ny, XX.shape))

    # plt.clf()
    # plt.hist(im.flatten(), 256, fc='k', ec='k') # range=(0.0,1.0)

    if verbose >= 2:
        print('extent: %s' % xx)
    if units is None:
        unitstr=''
    else:
        unitstr=' (%s)' % units
    if fig is not None:
        pmatlab.cfigure(fig)
        plt.clf()
        if verbose >= 2:
            print('show2D: show image')
        plt.imshow(im, extent=xx, interpolation='nearest')
        if dd.metadata.get('sweepdata', None) is not None:
            plt.xlabel('%s' % dd['sweepdata']['gates'][0] + unitstr)
        else:
            try:
                plt.xlabel('%s' % dd2d['argsd']['sweep_gates'][0] + unitstr)
            except:
                pass

        if dd.metadata.get('stepdata', None) is not None:
            if units is None:
                plt.ylabel('%s' % dd['stepdata']['gates'][0])
            else:
                plt.ylabel('%s (%s)' % (dd['stepdata']['gates'][0], units) )
    
        if not title is None:
            plt.title(title)
    # plt.axis('image')
    # ax=plt.gca()
    # ax.invert_yaxis()
        if colorbar:
            plt.colorbar()
        if verbose >= 2:
            print('show2D: at show')
        try:
            plt.show(block=False)
        except:
            # ipython backend does not know about block keyword...
            plt.show()

    return xx, vstep, vsweep

       

#%%
import numpy.linalg
import pmatlab

''' Class to convert scan coordinate to image coordinates '''
class image_transform:

    def __init__(self, dataset=None, im=None, upp=None):
        self.H=np.eye(3) # scan to image transformation
        self.extent = [] # image extent in pixel
        self.verbose=1
        self.dataset=dataset
        extentscan, g0,g2,vstep, vsweep, arrayname=dataset2Dmetadata(dataset, array=None)

        if upp is not None:
            raise NotImplemented
        nx = len(vsweep)
        ny = len(vstep)
        self.flipX=False
        self.flipY=False


        Hx = np.diag([-1,1,1]); Hx[0,-1]=nx-1
        Hy = np.diag([1,-1,1]); Hy[1,-1]=ny-1
        if self.verbose:
            print('image_transform: vsweep[0] %s' % vsweep[0])
        if vsweep[0]>vsweep[1]:
            #print('flip...')
            self.flipX=True
            self.H = Hx.dot(self.H)
        #return im
        self.Hi=numpy.linalg.inv(self.H)



    def transform(self, im):
        if self.flipX:
                im = im[::, ::-1]
        #im=cv2.warpPerspective(im.astype(np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)

        return im
    def itransform(self, im):
        if self.flipX:
                im = im[::, ::-1]
        #im=cv2.warpPerspective(im.astype(np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)

        return im

    def pixel2scan(self, pt):
        """ Convert pixels coordinates to scan coordinates (mV)
        Arguments
        ---------
        pt : array
            points in pixel coordinates

        """
        ptx= pmatlab.projectiveTransformation(self.Hi, np.array(pt).astype(float) )

        extent, g0,g1,vstep, vsweep, arrayname=dataset2Dmetadata(self.dataset, array=None, verbose=0)
        nx = vsweep.size #  zdata.shape[0]
        ny = vstep.size # zdata.shape[1]

        xx = extent
        x = ptx
        nn = pt.shape[1]
        ptx = np.zeros((2, nn))
        ptx[1, :] = np.interp(x[1, :], [0, ny - 1], [xx[2], xx[3]])    # step
        ptx[0, :] = np.interp(x[0, :], [0, nx - 1], [xx[0], xx[1]])    # sweep
        return ptx

    def scan2pixel(self, pt):
        """ Convert scan coordinates to pixel coordinates 
        Arguments
        ---------
        pt : array
            points in scan coordinates
        Returns:
            ptpixel (ndaray): points in pixel coordinates

        """
        extent, g0,g1,vstep, vsweep, arrayname=dataset2Dmetadata(self.dataset, array=None, verbose=0)
        #xx, _, _, zdata = get2Ddata(dd2d, verbose=0, fig=None)
        nx = vsweep.size 
        ny = vstep.size
    
        xx=extent
        x=pt
        nn = pt.shape[1]
        ptpixel = np.zeros((2, nn))
        ptpixel[1, :] = np.interp(x[1, :], [xx[2], xx[3]], [0, ny - 1])
        ptpixel[0, :] = np.interp(x[0, :], [xx[0], xx[1]], [0, nx - 1])

        ptpixel= pmatlab.projectiveTransformation(self.H, np.array(ptpixel).astype(float) )

        return ptpixel


        ptx= pmatlab.projectiveTransformation(self.Hi, np.array(pt).astype(float) )

        extent, g0,g1,vstep, vsweep, arrayname=dataset2Dmetadata(self.dataset, array=None, verbose=0)
        nx = vsweep.size #  zdata.shape[0]
        ny = vstep.size # zdata.shape[1]

        xx = extent
        x = ptx
        nn = pt.shape[1]
        ptx = np.zeros((2, nn))
        ptx[1, :] = np.interp(x[1, :], [0, ny - 1], [xx[2], xx[3]])    # step
        ptx[0, :] = np.interp(x[0, :], [0, nx - 1], [xx[0], xx[1]])    # sweep
        return ptx

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

def dataset2Dmetadata(alldata, array=None, verbose=0):
    if array is None:
        array = 'amplitude'
        if not array in alldata.arrays:
            array=list(alldata.arrays.keys())[0]

    A = alldata.arrays[array]


    g0=A.set_arrays[0].name
    g1=A.set_arrays[1].name
    vstep = np.array(A.set_arrays[0])
    vsweep = np.array(A.set_arrays[1])[0]
    extent = [vstep[0], vstep[-1], vsweep[0], vsweep[-1]]

    if verbose:
        print('2D scan: gates %s %s' % (g0, g1))
    return extent, g0, g1, vstep, vsweep, array


if __name__=='__main__':
    extent, g0,g1,vstep, vsweep, arrayname=dataset2Dmetadata(alldata, array=None)
    _=pix2scan( np.zeros( (2,4) ), alldata )

#%%
import warnings

try:
    import deepdish
except:
    warnings.warn('could not load deepdish...')
    
def loadQttData(path : str):
    ''' Wrapper function

    :param path: filename without extension
    :returns dataset: The dataset
    '''
    mfile=path
    if not mfile.endswith('hp5'):
        mfile = mfile + '.hp5'
    dataset=deepdish.io.load(mfile)
    return dataset

def writeQttData(dataset, path, metadata=None):
    ''' Wrapper function

    :param path: filename without extension
    '''
    mfile=path
    if not mfile.endswith('hp5'):
        mfile = mfile + '.hp5'
    deepdish.io.save(mfile, dataset)




def loadDataset(path):
    ''' Wrapper function

    :param path: filename without extension
    :returns dateset, metadata: 
    '''
    dataset = qcodes.load_data(path)

    mfile=os.path.join(path, 'qtt.hp5' )
    metadata=deepdish.io.load(mfile)
    return dataset, metadata

def writeDataset(path, dataset, metadata=None):
    ''' Wrapper function

    :param path: filename without extension
    '''
    dataset.write_copy(path=path)
    
    # already done in write_copy...
    #dataset.save_metadata(path=path)

    if metadata is None:
        metadata=dataset.metadata

    mfile=os.path.join(path, 'qtt.hp5' )
    _=deepdish.io.save(mfile, metadata)



#%%

if __name__=='__main__':
    import numpy as np
    import qcodes.tests.data_mocks
    alldata=qcodes.tests.data_mocks.DataSet2D()
    X=alldata.z
    print(np.array(X))
    s=np.linalg.svd(X)
    print(s)
