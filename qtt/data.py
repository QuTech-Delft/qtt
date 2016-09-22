import qtpy
# print(qtpy.API_NAME)

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
import warnings
import pickle
import scipy

import numpy.linalg
from qtt import pmatlab

try:
    import hickle
except:
    pass

import qtt.tools

# import qtpy.QtGui as QtGui
# import qtpy.QtWidgets as QtWidgets

import matplotlib.pyplot as plt

from qtt.tools import diffImageSmooth
from qcodes import DataArray, new_data

#%%


def getDefaultParameterName(data, defname='amplitude'):
    if defname in data.arrays.keys():
        return defname
    if (defname + '_0') in data.arrays.keys():
        return getattr(data, defname + '_0')

    vv = [v for v in data.arrays.keys() if v.endswith(defname)]
    if (len(vv) > 0):
        return vv[0]
    try:
        name = next(iter(data.arrays.keys()))
        return name
    except:
        pass
    return None


def getDefaultParameter(data, defname='amplitude'):
    name = getDefaultParameterName(data)
    if name is not None:
        return getattr(data, name)
    else:
        return None

#%%


def dataset2image(dataset, mode='pixel'):
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
        dataset, verbose=0, arrayname=None)
    tr = image_transform(dataset, mode=mode)
    im = None
    if arrayname is not None:
        imraw = dataset.arrays[arrayname]
        im = tr.transform(imraw)
    return im, tr


def dataset2image2(dataset):
    """ Extract image from dataset

    Arguments
        dataset (DataSet): measured data
    Returns:
        im (array): raw image
        impixel (array): image in pixel coordinates
        tr (transformation): transformation object
    """
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
        dataset, verbose=0, arrayname=None)
    tr = image_transform(dataset, mode='pixel')
    im = None
    impixel = None
    if arrayname is not None:
        im = dataset.arrays[arrayname]
        impixel = tr.transform(im)

    return im, impixel, tr

#%%


def dataset_get_istep(alldata, mode=None):
    istep = np.abs(alldata.metadata['scanjob']['sweepdata']['step'])
    return istep


def dataset1Ddata(alldata):
    ''' Parse a dataset into the x and y scan values '''
    y = alldata.default_parameter_array()
    x = y.set_arrays[0]
    return x, y


def dataset_labels(alldata, tag=None):
    if tag == 'x':
        d = alldata.default_parameter_array()
        return d.set_arrays[0].label
    if tag == 'y':
        d = alldata.default_parameter_array()
        return d.set_arrays[1].label
    if tag is None:
        d = alldata.default_parameter_array()
        return d.label
    return '?'


def show2D(dd, impixel=None, im=None, fig=101, verbose=1, dy=None, sigma=None, colorbar=False, title=None, midx=2, units=None):
    """ Show result of a 2D scan """
    if dd is None:
        return None

    extent, g0, g1, vstep, vsweep, arrayname = dataset2Dmetadata(dd)
    tr = image_transform(dd, mode='pixel')
    array = getattr(dd, arrayname)

    if impixel is None:
        if im is None:
            im = np.array(array)
            impixel = tr.transform(im)

        else:
            pass
            # impixel = tr.transform(im)
    else:
        pass
    # XX = array # dd['data_array']

    labels = [s.name for s in array.set_arrays]

    xx = extent
    xx = tr.extent_image()
    ny = vstep.size
    nx = vsweep.size

    # im=diffImage(im, dy)
    im = diffImageSmooth(impixel, dy=dy, sigma=sigma)

    if verbose:
        print('show2D: nx %d, ny %d' % (nx, ny,))

    # plt.clf()
    # plt.hist(im.flatten(), 256, fc='k', ec='k') # range=(0.0,1.0)

    if verbose >= 2:
        print('extent: %s' % xx)
    if units is None:
        unitstr = ''
    else:
        unitstr = ' (%s)' % units
    if fig is not None:
        scanjob = dd.metadata.get('scanjob', dict())
        pmatlab.cfigure(fig)
        plt.clf()

        if impixel is None:
            if verbose >= 2:
                print('show2D: show raw image')
            plt.pcolormesh(vstep, vsweep, im)
        else:
            if verbose >= 2:
                print('show2D: show image')
            plt.imshow(impixel, extent=xx, interpolation='nearest')
        labelx = labels[1]
        labely = labels[0]
        if scanjob.get('sweepdata', None) is not None:
            labelx = scanjob['sweepdata']['gates'][0]
            plt.xlabel('%s' % labelx + unitstr)
        else:
            pass
            # try:
            #    plt.xlabel('%s' % dd2d['argsd']['sweep_gates'][0] + unitstr)
            # except:
            #    pass

        if scanjob.get('stepdata', None) is not None:
            if units is None:
                plt.ylabel('%s' % scanjob['stepdata']['gates'][0])
            else:
                plt.ylabel('%s (%s)' %
                           (scanjob['stepdata']['gates'][0], units))

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

''' Class to convert scan coordinate to image coordinates '''


class image_transform:

    def __init__(self, dataset=None, arrayname=None, mode='pixel', verbose=0):
        self.H = np.eye(3)  # raw image to pixel image transformation
        self.extent = []  # image extent in pixel
        self.verbose = verbose
        self.dataset = dataset
        extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
            dataset, arrayname=arrayname)
        self.vstep = vstep
        self.vsweep = vsweep
        nx = len(vsweep)
        ny = len(vstep)
        self.flipX = False
        self.flipY = False

        Hx = np.diag([-1, 1, 1])
        Hx[0, -1] = nx - 1
        Hy = np.diag([1, -1, 1])
        Hy[1, -1] = ny - 1
        if self.verbose:
            print('image_transform: vsweep[0] %s' % vsweep[0])

        if mode == 'raw':
            pass
        else:
            if vsweep[0] > vsweep[1]:
                # print('flip...')
                self.flipX = True
                self.H = Hx.dot(self.H)

        # return im
        self.Hi = numpy.linalg.inv(self.H)

    def extent_image(self):
        """ Return matplotlib style image extent """
        vsweep = self.vsweep
        vstep = self.vstep
        extentImage = [vsweep[0], vsweep[-1], vstep[-1], vstep[0]]
        if self.flipX:
            extentImage = [extentImage[1], extentImage[
                0], extentImage[2], extentImage[3]]
        if self.flipY:
            extentImage = [extentImage[0], extentImage[
                1], extentImage[3], extentImage[2]]
        return extentImage

    def transform(self, im):
        """ Transform raw image to image in pixel coordinates such that the imageExtent is increasing

        TODO: explain        
        """
        if self.flipX:
            im = im[::, ::-1]
        if self.flipY:
            im = im[::-1, ::]
        #im=cv2.warpPerspective(im.astype(np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)

        return im

    def itransform(self, im):
        if self.flipX:
            im = im[::, ::-1]
        if self.flipY:
            im = im[::-1, ::]
        #im=cv2.warpPerspective(im.astype(np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)

        return im

    def pixel2scan(self, pt):
        """ Convert pixels coordinates to scan coordinates (mV)
        Arguments
        ---------
        pt : array
            points in pixel coordinates (x,y)
        Returns
        -------
          ptx (array): point in scan coordinates (sweep, step)

        """
        ptx = pmatlab.projectiveTransformation(
            self.Hi, np.array(pt).astype(float))

        extent, g0, g1, vstep, vsweep, arrayname = dataset2Dmetadata(
            self.dataset, arrayname=None, verbose=0)
        nx = vsweep.size  # zdata.shape[0]
        ny = vstep.size  # zdata.shape[1]

        xx = extent
        x = ptx
        nn = pt.shape[1]
        ptx = np.zeros((2, nn))
        # ptx[1, :] = np.interp(x[1, :], [0, ny - 1], [xx[2], xx[3]])    # step
        # ptx[0, :] = np.interp(x[0, :], [0, nx - 1], [xx[0], xx[1]])    # sweep
        
        f = scipy.interpolate.interp1d([0, ny - 1], [xx[2], xx[3]], assume_sorted = False, fill_value='extrapolate')
        ptx[1, :] = f(x[1, :])  # step
        f = scipy.interpolate.interp1d([0, nx - 1], [xx[0], xx[1]], assume_sorted = False, fill_value='extrapolate')
        ptx[0, :] = f(x[0,:])  # sweep

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
        extent, g0, g1, vstep, vsweep, arrayname = dataset2Dmetadata(
            self.dataset, arrayname=None, verbose=0)
        #xx, _, _, zdata = get2Ddata(dd2d, verbose=0, fig=None)
        nx = vsweep.size
        ny = vstep.size

        xx = extent
        x = pt
        nn = pt.shape[1]
        ptpixel = np.zeros((2, nn))
        f = scipy.interpolate.interp1d(
            [xx[2], xx[3]], [0, ny - 1], assume_sorted=False)
        ptpixel[1, :] = f(x[1, :])
        f = scipy.interpolate.interp1d(
            [xx[0], xx[1]], [0, nx - 1], assume_sorted=False)
        ptpixel[0, :] = f(x[0, :])  # sweep to pixel x
        #ptpixel[1, :] = np.interp(x[1, :], [xx[2], xx[3]], [0, ny - 1])
        #ptpixel[0, :] = np.interp(x[0, :], [xx[0], xx[1]], [0, nx - 1])

        ptpixel = pmatlab.projectiveTransformation(
            self.H, np.array(ptpixel).astype(float))

        return ptpixel

        ptx = pmatlab.projectiveTransformation(
            self.Hi, np.array(pt).astype(float))

        extent, g0, g1, vstep, vsweep, arrayname = dataset2Dmetadata(
            self.dataset, verbose=0)
        nx = vsweep.size  # zdata.shape[0]
        ny = vstep.size  # zdata.shape[1]

        xx = extent
        x = ptx
        nn = pt.shape[1]
        ptx = np.zeros((2, nn))
        ptx[1,:] = np.interp(x[1,:], [0, ny - 1], [xx[2], xx[3]])    # step
        ptx[0,:] = np.interp(x[0,:], [0, nx - 1], [xx[0], xx[1]])    # sweep
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
    warnings.warn('use transformation object instead')
    extent, g0, g1, vstep, vsweep, arrayname = dataset2Dmetadata(
        dd2d, verbose=0)
    #xx, _, _, zdata = get2Ddata(dd2d, verbose=0, fig=None)
    nx = vsweep.size  # zdata.shape[0]
    ny = vstep.size  # zdata.shape[1]

    xx = extent
    x = pt
    nn = pt.shape[1]
    ptx = np.zeros((2, nn))
    ptx[1,:] = np.interp(x[1,:], [0, ny - 1], [xx[3], xx[2]])    # step
    ptx[0,:] = np.interp(x[0,:], [0, nx - 1], [xx[0], xx[1]])    # sweep
    return ptx

#%%


def dataset2Dmetadata(alldata, arrayname=None, verbose=0):
    """ Extract metadata from a 2D scan

    Returns:

        extent (list): x1,x2,y1,y2
        g0 (string): step gate
        g1 (string): sweep gate
        vstep (array): step values
        vsweep (array): sweep values
        arrayname (string): identifier of the main array 

    """

    if arrayname is None:
        arrayname = alldata.default_parameter_name()

    A = alldata.arrays[arrayname]

    g0 = A.set_arrays[0].name
    g1 = A.set_arrays[1].name
    vstep = np.array(A.set_arrays[0])
    vsweep = np.array(A.set_arrays[1])[0]
    # extent = [vstep[0], vstep[-1], vsweep[0], vsweep[-1]] # change order?
    extent = [vsweep[0], vsweep[-1], vstep[0], vstep[-1]]  # change order?

    if verbose:
        print('2D scan: gates %s %s' % (g0, g1))
    return extent, g0, g1, vstep, vsweep, arrayname


if __name__ == '__main__':
    extent, g0, g1, vstep, vsweep, arrayname = dataset2Dmetadata(
        alldata, array=None)
    _ = pix2scan(np.zeros((2, 4)), alldata)

#%%

try:
    import deepdish
except:
    pass
#    warnings.warn('could not load deepdish...')


def data_extension():
    return 'pickle'


def load_data(mfile: str):
    ''' Load data from specified file '''
    # return hickle.load(mfile)
    ext = data_extension()
    if ext is not None:
        if not mfile.endswith(ext):
            mfile = mfile + '.' + ext
    with open(mfile, 'rb') as fid:
        return pickle.load(fid)


def write_data(mfile: str, data):
    ''' Write data to specified file '''
    ext = data_extension()
    if ext is not None:
        if not mfile.endswith(ext):
            mfile = mfile + '.' + ext
    with open(mfile, 'wb') as fid:
        pickle.dump(data, fid)
    # hickle.dump(metadata, mfile)
    #_=deepdish.io.save(mfile, data)


def loadQttData(path: str):
    ''' Wrapper function

    :param path: filename without extension
    :returns dataset: The dataset
    '''
    warnings.warn('please use load_data instead')
    mfile = path
    ext = data_extension()
    if not mfile.endswith(ext):
        mfile = mfile + '.' + ext
    # dataset=deepdish.io.load(mfile)
    dataset = load_data(mfile)
    return dataset


def writeQttData(dataset, path, metadata=None):
    ''' Wrapper function

    :param path: filename without extension
    '''
    warnings.warn('please use write_data instead')
    mfile = path
    ext = data_extension()
    if not mfile.endswith(ext):
        mfile = mfile + ext
    # deepdish.io.save(mfile, dataset)
    write_data(mfile, dataset)


def loadDataset(path):
    ''' Wrapper function

    :param path: filename without extension
    :returns dateset, metadata: 
    '''
    dataset = qcodes.load_data(path)

    mfile = os.path.join(path, 'qtt-metadata')
    metadata = load_data(mfile)
    return dataset, metadata


def writeDataset(path, dataset, metadata=None):
    ''' Wrapper function

    :param path: filename without extension
    '''
    dataset.write_copy(path=path)

    # already done in write_copy...
    # dataset.save_metadata(path=path)

    if metadata is None:
        metadata = dataset.metadata

    mfile = os.path.join(path, 'qtt-metadata')
    write_data(mfile, metadata)


def getTimeString(t=None):
    """ Return time string for datetime.datetime object """
    if t == None:
        t = datetime.datetime.now()
    if isinstance(t, float):
        t = datetime.datetime.fromtimestamp(t)
    dstr = t.strftime('%H-%M-%S')
    return dstr


def getDateString(t=None, full=False):
    """ Return date string

    Args:
        t : datetime.datetime
            time
    """
    if t is None:
        t = datetime.datetime.now()
    if isinstance(t, float):
        t = datetime.datetime.fromtimestamp(t)
    if full:
        dstr = t.strftime('%Y-%m-%d-%H-%M-%S')
    else:
        dstr = t.strftime('%Y-%m-%d')
    return dstr


def experimentFile(outputdir: str = '', tag=None, dstr=None, bname=None):
    """ Format experiment data file for later analysis """
    if tag is None:
        tag = getDateString()
    if dstr is None:
        dstr = getDateString()

    ext = data_extension()
    basename = '%s' % (dstr,)
    if bname is not None:
        basename = '%s-' % bname + basename
    if not outputdir is None:
        qtt.tools.mkdirc(os.path.join(outputdir, tag))
    pfile = os.path.join(outputdir, tag, basename + '.' + ext)
    return pfile


def loadExperimentData(outputdir, tag, dstr):
    path = experimentFile(outputdir, tag=tag, dstr=dstr)
    logging.info('loadExperimentdata %s' % path)
    dataset = load_data(path)
    return dataset


def saveExperimentData(outputdir, dataset, tag, dstr):
    path = experimentFile(outputdir, tag=tag, dstr=dstr)
    logging.info('saveExperimentData %s' % path)
    write_data(path, dataset)


def makeDataSet1D(p, mname='measured', location=None, preset_data=None):
    ''' Make DataSet with one 1D array and one setpoint array
    
    Arguments:
        p (array): the setpoint array of data
    '''
    xx = np.array(p)
    yy = np.ones(xx.size)
    x = DataArray(name=p.name, array_id=p.name,
                  label=p.parameter.label, preset_data=xx, is_setpoint=True)
    y = DataArray(name=mname, array_id=mname, label=mname,
                  preset_data=yy, set_arrays=(x,))
    dd = new_data(arrays=(), location=location)
    dd.add_array(x)
    dd.add_array(y)
    
    if preset_data is not None:
        dd.measured.ndarray = np.array(preset_data)

    return dd
    

def makeDataSet2D(p1, p2, mname='measured', location=None, preset_data=None):
    ''' Make DataSet with one 2D array and two setpoint arrays 
    
    Arguments:
        p1 (array): first setpoint array of data
        p2 (array): second setpoint array of data
    '''
    xx = np.array(p1)
    yy0 = np.array(p2)
    yy = np.tile(yy0, [xx.size, 1])
    zz = np.NaN * np.ones((xx.size, yy0.size))
    x = DataArray(name=p1.name, array_id=p1.name,
                  label=p1.parameter.label, preset_data=xx, is_setpoint=True)
    y = DataArray(name=p2.name,  array_id=p2.name, label=p2.parameter.label,
                  preset_data=yy, set_arrays=(x,), is_setpoint=True)
    z = DataArray(name=mname, array_id=mname, label=mname,
                  preset_data=zz, set_arrays=(x, y))
    dd = new_data(arrays=(), location=location)
    dd.add_array(z)
    dd.add_array(x)
    dd.add_array(y)
    
    if preset_data is not None:
        dd.measured.ndarray = np.array(preset_data)
    
    return dd


#%%

if __name__ == '__main__':
    import numpy as np
    import qcodes.tests.data_mocks
    alldata = qcodes.tests.data_mocks.DataSet2D()
    X = alldata.z
    print(np.array(X))
    s = np.linalg.svd(X)
    print(s)
