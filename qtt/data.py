
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
try:
    import hickle
except:
    pass

import numpy.linalg
from qtt import pmatlab

import qtt.tools
from qtt.tools import deprecated

import matplotlib.pyplot as plt

from qtt.tools import diffImageSmooth
from qcodes import DataArray, new_data

#%%


def getDefaultParameterName(data, defname='amplitude'):
    print('do not use this function, use the function from the object...')
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
    """ Extract image from a dataset

    Args:
        dataset
        mode (str): if value is 'pixel' then the image is converted so that 
        it is in conventional coordinates, e.g. the step values (vertical axis)
        go from low to high (bottom to top).

    Returns:
        im (numpy array)
        tr (image_transform object)

    """
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
        dataset, verbose=0, arrayname=None)
    tr = image_transform(dataset, mode=mode)
    im = None
    if arrayname is not None:
        imraw = dataset.arrays[arrayname].ndarray
        im = tr.transform(imraw)
    return im, tr


def dataset2image2(dataset):
    """ Extract image from dataset

    Arguments
        dataset (DataSet): measured data
    Returns:
        imraw (array): raw image
        impixel (array): image in pixel coordinates
        tr (image_transform object): transformation object

    See also: dataset2image
    """
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
        dataset, verbose=0, arrayname=None)
    tr = image_transform(dataset, mode='pixel')
    imraw = None
    impixel = None
    if arrayname is not None:
        imraw = dataset.arrays[arrayname]
        impixel = tr.transform(imraw)

    return imraw, impixel, tr

#%%


def dataset_get_istep(alldata, mode=None):
    """ Return number of mV per pixel in scan """
    try:
        istep = np.abs(alldata.metadata['scanjob']['stepdata']['step'])
    except:
        try:
            extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(alldata, verbose=0, arrayname=None)
            istep = np.abs(np.nanmean(np.diff(vstep)))
        except:
            _, _, _, istep, _ = dataset1Dmetadata(alldata)
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


def uniqueArrayName(dataset, name0):
    ''' Generate a unique name for a DataArray in a dataset '''
    ii = 0
    name = name0
    while name in dataset.arrays:
        name = name0 + '_%d' % ii
        ii = ii + 1
        if ii > 1000:
            raise Exception('too many arrays in DataSet')
    return name

from qcodes.plots.qcmatplotlib import MatPlot


def diffDataset(alldata, diff_dir='y', fig=None):
    """ Differentiate a dataset and plot the result """
    imx = qtt.diffImageSmooth(alldata.measured.ndarray, dy=diff_dir)
    name = 'diff_dir_%s' % diff_dir
    name = uniqueArrayName(alldata, name)
    data_arr = qcodes.DataArray(name=name, label=name, array_id=name, set_arrays=alldata.measured.set_arrays, preset_data=imx)

    alldata.add_array(data_arr)

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plot = MatPlot(interval=0, num=fig)
        plot.add(alldata.arrays[name])
        plot.fig.axes[0].autoscale(tight=True)
        plot.fig.axes[1].autoscale(tight=True)

    return alldata

#%%


def sweepgate(scanjob):
    g = scanjob['sweepdata'].get('gate', None)
    if g is None:
        g = scanjob['sweepdata'].get('gates', [None])[0]
    return g


def stepgate(scanjob):
    g = scanjob['stepdata'].get('gate', None)
    if g is None:
        g = scanjob['stepdata'].get('gates', [None])[0]
    return g


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
    xx = tr.matplotlib_image_extent()
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
            labelx = sweepgate(scanjob)
            plt.xlabel('%s' % labelx + unitstr)
        else:
            pass
            # try:
            #    plt.xlabel('%s' % dd2d['argsd']['sweep_gates'][0] + unitstr)
            # except:
            #    pass

        if scanjob.get('stepdata', None) is not None:
            if units is None:
                plt.ylabel('%s' % stepgate(scanjob))
            else:
                plt.ylabel('%s (%s)' % (stepgate(scanjob), units))

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


class image_transform:

    def __init__(self, dataset=None, arrayname=None, mode='pixel', verbose=0):
        """ Class to convert scan coordinate to image coordinates
        
        Args:
            dataset (DataSet):
            arrayname (str or None)
            mode (str): 'pixel' or 'raw'
        
        """
        self.H = np.eye(3)  # raw image to pixel image transformation
        self.extent = []  # image extent in pixel
        self.verbose = verbose
        self.dataset = dataset
        extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
            dataset, arrayname=arrayname)
        self.vstep = vstep
        self.vsweep = vsweep

        self._istep = dataset_get_istep(dataset)
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
                self.flipX = True
                self.H = Hx.dot(self.H)
            if vstep[0] < vstep[1]:
                self.flipY = True
                self.H = Hy.dot(self.H)

        # return im
        self.Hi = numpy.linalg.inv(self.H)

    def istep(self):
        return self._istep

    def scan_image_extent(self):
        """ Scan extent """
        vsweep = self.vsweep
        vstep = self.vstep
        extentImage = [vsweep[0], vsweep[-1], vstep[0], vstep[-1]]
        if self.flipX:
            extentImage = [extentImage[1], extentImage[
                0], extentImage[2], extentImage[3]]
        if self.flipY:
            extentImage = [extentImage[0], extentImage[
                1], extentImage[3], extentImage[2]]
        self.extent = extentImage
        return extentImage

    def matplotlib_image_extent(self):
        """ Return matplotlib style image extent

        Returns:
            extentImage (4 floats): x1, x2, y1, y2
                        the y1 value is bottom left
        """
        vsweep = self.vsweep
        vstep = self.vstep
        extentImage = [vsweep[0], vsweep[-1], vstep[-1], vstep[0]]
        if self.flipX:
            extentImage = [extentImage[1], extentImage[
                0], extentImage[2], extentImage[3]]
        if self.flipY:
            extentImage = [extentImage[0], extentImage[
                1], extentImage[3], extentImage[2]]
        self.extent = extentImage
        return extentImage

    def transform(self, im):
        """ Transform raw image to image in pixel coordinates such that the imageExtent is increasing

        TODO: explain        
        """
        if self.flipX:
            im = im[::, ::-1]
        if self.flipY:
            im = im[::-1, ::]
        # im=cv2.warpPerspective(im.astype(np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)

        return im

    def itransform(self, im):
        if self.flipX:
            im = im[::, ::-1]
        if self.flipY:
            im = im[::-1, ::]
        # im=cv2.warpPerspective(im.astype(np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)

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

        f = scipy.interpolate.interp1d([0, ny - 1], [xx[2], xx[3]], assume_sorted=False, fill_value='extrapolate')
        ptx[1, :] = f(x[1, :])  # step
        f = scipy.interpolate.interp1d([0, nx - 1], [xx[0], xx[1]], assume_sorted=False, fill_value='extrapolate')
        ptx[0, :] = f(x[0, :])  # sweep

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
        # xx, _, _, zdata = get2Ddata(dd2d, verbose=0, fig=None)
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
        # ptpixel[1, :] = np.interp(x[1, :], [xx[2], xx[3]], [0, ny - 1])
        # ptpixel[0, :] = np.interp(x[0, :], [xx[0], xx[1]], [0, nx - 1])

        ptpixel = pmatlab.projectiveTransformation(
            self.H, np.array(ptpixel).astype(float))

        return ptpixel


@deprecated
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
    # xx, _, _, zdata = get2Ddata(dd2d, verbose=0, fig=None)
    nx = vsweep.size  # zdata.shape[0]
    ny = vstep.size  # zdata.shape[1]

    xx = extent
    x = pt
    if len(pt.shape) == 1:
        ptx = np.zeros((2, 1))
        ptx[1] = np.interp(x[1], [0, ny - 1], [xx[3], xx[2]])    # step
        ptx[0] = np.interp(x[0], [0, nx - 1], [xx[0], xx[1]])    # sweep
    else:
        nn = pt.shape[1]
        ptx = np.zeros((2, nn))
        ptx[1, :] = np.interp(x[1, :], [0, ny - 1], [xx[3], xx[2]])    # step
        ptx[0, :] = np.interp(x[0, :], [0, nx - 1], [xx[0], xx[1]])    # sweep
    return ptx

#%%


def dataset1Dmetadata(alldata, arrayname=None, verbose=0):
    """ Extract metadata from a 2D scan

    Returns:

        extent (list): x1,x2
        g0 (string): step gate
        vstep (array): step values
        istep (float)
        arrayname (string): identifier of the main array

    """

    if arrayname is None:
        arrayname = alldata.default_parameter_name()

    A = alldata.arrays[arrayname]

    g0 = A.set_arrays[0].name
    vstep = np.array(A.set_arrays[0])
    extent = [vstep[0], vstep[-1]]  # change order?

    istep = np.abs(np.mean(np.diff(vstep)))
    if verbose:
        print('1D scan: gates %s %s' % (g0,))
    return extent, g0, vstep, istep, arrayname


def dataset1Dmetadata(alldata, arrayname=None, verbose=0):
    """ Extract metadata from a 2D scan

    Returns:

        extent (list): x1,x2
        g0 (string): step gate
        vstep (array): step values
        istep (float)
        arrayname (string): identifier of the main array 

    """

    if arrayname is None:
        arrayname = alldata.default_parameter_name()

    A = alldata.arrays[arrayname]

    g0 = A.set_arrays[0].name
    vstep = np.array(A.set_arrays[0])
    extent = [vstep[0], vstep[-1]]  # change order?

    istep = np.abs(np.mean(np.diff(vstep)))
    if verbose:
        print('1D scan: gates %s %s' % (g0,))
    return extent, g0, vstep, istep, arrayname


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


def pickleload(pkl_file):
    """ Load objects from file with pickle """
    try:
        with open(pkl_file, 'rb') as output:
            data2 = pickle.load(output)
    except:
        if sys.version_info.major >= 3:
            # if pickle file was saved in python2 we might fix issues with a different encoding
            with open(pkl_file, 'rb') as output:
                data2 = pickle.load(output, encoding='latin')
            # pickle.load(pkl_file, fix_imports=True, encoding="ASCII", errors="strict")
        else:
            data2 = None
    return data2


def load_data(mfile: str):
    ''' Load data from specified file '''
    # return hickle.load(mfile)
    ext = data_extension()
    if ext is not None:
        if not mfile.endswith(ext):
            mfile = mfile + '.' + ext
    return pickleload(mfile)
    # with open(mfile, 'rb') as fid:
    #    return pickle.load(fid)


def write_data(mfile: str, data):
    ''' Write data to specified file '''
    ext = data_extension()
    if ext is not None:
        if not mfile.endswith(ext):
            mfile = mfile + '.' + ext
    if isinstance(data, qcodes.DataSet):
        data = qtt.tools.stripDataset(data)

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

    dataset = qtt.tools.stripDataset(dataset)

    print('write_copy to %s' % path)
    dataset.write_copy(path=path)
    print('write_copy to %s (done)' % path)

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
    from qtt import pmatlab
    dataset = pmatlab.load(path)

    dataset = load_data(path)
    return dataset


def saveExperimentData(outputdir, dataset, tag, dstr):
    path = experimentFile(outputdir, tag=tag, dstr=dstr)
    logging.info('saveExperimentData %s' % path)
    write_data(path, dataset)


def makeDataSet1Dplain(xname, x, yname, y, location=None):
    ''' Make DataSet with one 1D array and one setpoint array

    Arguments:
        xname (string): the name of the setpoint array
        x (array): the setpoint data
        yname (string): the name of the measured array
        y (array): the measured data
    '''
    xx = np.array(x)
    yy = np.array(y)
    x = DataArray(name=xname, array_id=xname, preset_data=xx, is_setpoint=True)
    y = DataArray(name=yname, array_id=yname, preset_data=yy, set_arrays=(x,))
    dd = new_data(arrays=(), location=location)
    dd.add_array(x)
    dd.add_array(y)

    return dd


def makeDataSet1D(x, yname='measured', y=None, location=None):
    ''' Make DataSet with one 1D array and one setpoint array

    Arguments:
        x (array): the setpoint array of data
    '''
    xx = np.array(x)
    yy = np.ones(xx.size)
    x = DataArray(name=x.name, array_id=x.name, label=x.parameter.label, preset_data=xx, is_setpoint=True)
    ytmp = DataArray(name=yname, array_id=yname, label=yname, preset_data=yy, set_arrays=(x,))
    dd = new_data(arrays=(), location=location)
    dd.add_array(x)
    dd.add_array(ytmp)

    if y is not None:
        dd.measured.ndarray = np.array(y)

    return dd


def makeDataSet2D(p1, p2, mname='measured', location=None, preset_data=None):
    """ Make DataSet with one 2D array and two setpoint arrays

    Args:
        p1 (array): first setpoint array of data
        p2 (array): second setpoint array of data
        mname (str): name of measured array
        location (str or None): location for the DataSet
        preset_data (array or None): optional array to fill the DataSet
    Returns:
        dd (DataSet)
    """
    xx = np.array(p1)
    yy0 = np.array(p2)
    yy = np.tile(yy0, [xx.size, 1])
    zz = np.NaN * np.ones((xx.size, yy0.size))
    x = DataArray(name=p1.name, array_id=p1.name, label=p1.parameter.label, preset_data=xx, is_setpoint=True)
    y = DataArray(name=p2.name, array_id=p2.name, label=p2.parameter.label, preset_data=yy, set_arrays=(x,), is_setpoint=True)
    z = DataArray(name=mname, array_id=mname, label=mname, preset_data=zz, set_arrays=(x, y))
    dd = new_data(arrays=(), location=location)
    dd.add_array(z)
    dd.add_array(x)
    dd.add_array(y)

    if preset_data is not None:
        dd.measured.ndarray = np.array(preset_data)

    dd.last_write = -1

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
