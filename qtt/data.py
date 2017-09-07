""" Utilities to work with data and datasets """
import numpy as np
import scipy
import os
import sys
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
from qtt import pgeometry as pmatlab

import qtt.tools
import qtt.algorithms.generic


import matplotlib.pyplot as plt

from qtt.tools import diffImageSmooth
from qcodes import DataArray, new_data


#%%


def dataset2image(dataset, arrayname=None, unitsperpixel=None, mode='pixel'):
    """ Extract image from a dataset

    Args:
        dataset
        arrayname (None or str): name of array to select
        mode (str): if value is 'pixel' then the image is converted so that 
             it is in conventional coordinates, e.g. the step values
             (vertical axis) go from low to high (bottom to top).

    Returns:
        im (numpy array)
        tr (image_transform object)

    """
    if arrayname is None:
        arrayname = dataset.default_parameter_name()
    tr = image_transform(dataset, mode=mode, unitsperpixel=unitsperpixel)
    im = None
    if arrayname is not None:
        imraw = dataset.arrays[arrayname].ndarray
        im = tr._transform(imraw)
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
        impixel = tr._transform(imraw)

    return imraw, impixel, tr

#%%


def dataset_get_istep(alldata, mode=None):
    """ Return number of mV per pixel in scan """
    try:
        istep = np.abs(alldata.metadata['scanjob']['stepdata']['step'])
    except:
        try:
            extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
                alldata, verbose=0, arrayname=None)
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


def diffDataset(alldata, diff_dir='y', fig=None, meas_arr_name='measured'):
    """ Differentiate a dataset and plot the result.

    Args:
        alldata (qcodes DataSet)
        diff_dir (str): direction to differentiate in
        meas_arr_name (str): name of the measured array to be differentiated
        fig (int): the number for the figure to plot
    """
    meas_arr_name = alldata.default_parameter_name(meas_arr_name)
    meas_array = alldata.arrays[meas_arr_name]
    imx = qtt.diffImageSmooth(meas_array.ndarray, dy=diff_dir)
    name = 'diff_dir_%s' % diff_dir
    name = uniqueArrayName(alldata, name)
    data_arr = qcodes.DataArray(
        name=name, label=name, array_id=name, set_arrays=meas_array.set_arrays, preset_data=imx)

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
    g = scanjob['sweepdata'].get('param', None)
    if isinstance(g, str):
        return g
    g = scanjob['sweepdata'].get('gate', None)
    if g is None:
        g = scanjob['sweepdata'].get('gates', [None])[0]
    return g


def stepgate(scanjob):
    g = scanjob['stepdata'].get('param', None)
    if isinstance(g, str):
        return g
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
            impixel = tr._transform(im)

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


#%% Extract metadata


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


if __name__ == '__main__' and 0:
    extent, g0, g1, vstep, vsweep, arrayname = dataset2Dmetadata(
        alldata, arrayname=None)
    _ = pix2scan(np.zeros((2, 4)), alldata)

#%%


class image_transform:

    def __init__(self, dataset=None, arrayname=None, mode='pixel', unitsperpixel=None, verbose=0):
        """ Class to convert scan coordinate to image coordinates

        Args:
            dataset (DataSet):
            arrayname (str or None): name of array to select from dataset
            mode (str): 'pixel' or 'raw'

        """
        self.H = np.eye(3)  # raw image to pixel image transformation
        self.extent = None  # image extent in pixel
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

        self._imraw = dataset.arrays[arrayname].ndarray

        if isinstance(unitsperpixel, (float, int)):
            unitsperpixel = [unitsperpixel, unitsperpixel]
        self.unitsperpixel = unitsperpixel
        if self.unitsperpixel is not None:
            imextent = self.scan_image_extent()
            if self.verbose:
                print('image_transform: unitsperpixel %s' %
                      (self.unitsperpixel, ))
            ims, Hs, _ = qtt.algorithms.generic.rescaleImage(
                self._imraw, imextent, mvx=unitsperpixel[0], mvy=unitsperpixel[1])
            self._im = ims
            self.H = Hs @ self.H

        self._im = self._transform(self._imraw)
        self.Hi = numpy.linalg.inv(self.H)

    def image(self):
        return self._im

    def istep(self):
        return self._istep

    def scan_image_extent(self):
        """ Scan extent

        Returns:
            extentImage (list): x0, x1, y0, y1
                            x0, y0 is top left
        """
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

    def transform_image(self, im):
        return self._transform(im)

    def _transform(self, im):
        """ Transform raw image to image in pixel coordinates such that the imageExtent is increasing

        """
        if self.flipX:
            im = im[::, ::-1]
        if self.flipY:
            im = im[::-1, ::]

        if self.unitsperpixel is not None:
            imextent = self.scan_image_extent()
            ims, Hs, _ = qtt.algorithms.generic.rescaleImage(
                self._imraw, imextent, mvx=self.unitsperpixel[0], mvy=self.unitsperpixel[1])
        else:
            ims = im
        return ims

    def _itransform(self, im):
        if self.flipX:
            im = im[::, ::-1]
        if self.flipY:
            im = im[::-1, ::]

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
        # ptx[0, :] = np.interp(x[0, :], [0, nx - 1], [xx[0], xx[1]])    #
        # sweep

        f = scipy.interpolate.interp1d(
            [0, ny - 1], [xx[2], xx[3]], assume_sorted=False, fill_value='extrapolate')
        ptx[1, :] = f(x[1, :])  # step
        f = scipy.interpolate.interp1d(
            [0, nx - 1], [xx[0], xx[1]], assume_sorted=False, fill_value='extrapolate')
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


def test_image_transform():
    from qcodes.tests.data_mocks import DataSet2D
    ds = DataSet2D()
    tr = image_transform(ds)
    im = tr.image()
    print('transform: im.shape %s' % (str(im.shape),))
    tr = image_transform(ds, unitsperpixel=[None, 2])
    im = tr.image()
    print('transform: im.shape %s' % (str(im.shape),))


if __name__ == '__main__':
    import pdb
    test_image_transform()

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
            # if pickle file was saved in python2 we might fix issues with a
            # different encoding
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
    dataset = pmatlab.load(path)

    dataset = load_data(path)
    return dataset


def saveExperimentData(outputdir, dataset, tag, dstr):
    path = experimentFile(outputdir, tag=tag, dstr=dstr)
    logging.info('saveExperimentData %s' % path)
    write_data(path, dataset)


def makeDataSet1Dplain(xname, x, yname, y, location=None, loc_record=None):
    ''' Make DataSet with one 1D array and one setpoint array

    Arguments:
        xname (string): the name of the setpoint array
        x (array): the setpoint data
        yname (str or list): the name of the measured array
        y (array): the measured data
    '''
    xx = np.array(x)
    yy = np.array(y)
    x = DataArray(name=xname, array_id=xname, preset_data=xx, is_setpoint=True)
    dd = new_data(arrays=(), location=location, loc_record=loc_record)
    dd.add_array(x)
    if isinstance(yname, str):
        y = DataArray(name=yname, array_id=yname,
                      preset_data=yy, set_arrays=(x,))
        dd.add_array(y)
    else:
        for ii, name in enumerate(yname):
            y = DataArray(name=name, array_id=name,
                          preset_data=yy[ii], set_arrays=(x,))
            dd.add_array(y)

    return dd


def makeDataSet1D(x, yname='measured', y=None, location=None, loc_record=None, return_names=False):
    ''' Make DataSet with one or multiple 1D arrays and one setpoint array.

    Arguments:
        x (array): the setpoint array of data
        yname (str or list): name(s) of measured array(s)
        y (array or None): optional array to fill the DataSet
        location (str or None): location for the DataSet
        loc_record (dict): will be added to the location
        return_names (bool): if True return array names in output
    '''
    xx = np.array(x)
    yy = np.NaN * np.ones(xx.size)
    x = DataArray(name=x.name, array_id=x.name, label=x.parameter.label,
                  unit=x.parameter.unit, preset_data=xx, is_setpoint=True)
    if isinstance(yname, str):
        measure_names = [yname]
        if y is not None:
            preset_data = [y]
    else:
        measure_names = yname
    mnamesx = measure_names
    measure_names = []
    for p in mnamesx:
        if isinstance(p, str):
            measure_names += [p]
        else:
            # assume p is a Parameter
            measure_names += [p.full_name]

    dd = new_data(arrays=(), location=location, loc_record=loc_record)

    for idm, mname in enumerate(measure_names):
        ytmp = DataArray(name=mname, array_id=mname, label=mname,
                         preset_data=np.copy(yy), set_arrays=(x,))
        dd.add_array(ytmp)
        if y is not None:
            getattr(dd, mname).ndarray = np.array(preset_data[idm])
    dd.add_array(x)

    if return_names:
        set_names = x.name
        return dd, (set_names, measure_names)
    else:
        return dd


def makeDataSet2D(p1, p2, measure_names='measured', location=None, loc_record=None,
                  preset_data=None, return_names=False):
    """ Make DataSet with one or multiple 2D array and two setpoint arrays.

    If the preset_data is used for multiple 2D arrays, then the order of 
    measure_names should match the order of preset_data.

    Args:
        p1 (array): first setpoint array of data
        p2 (array): second setpoint array of data
        mname (str or list): name(s) of measured array(s)
        location (str or None): location for the DataSet
        preset_data (array or None): optional array to fill the DataSet
        return_names (bool): if True return array names in output
    Returns:
        dd (DataSet)
        names (tuple, optional)
    """
    xx = np.array(p1)
    yy0 = np.array(p2)
    yy = np.tile(yy0, [xx.size, 1])
    zz = np.NaN * np.ones((xx.size, yy0.size))
    set_names = [p1.name, p2.name]
    x = DataArray(name=p1.name, array_id=p1.name, label=p1.parameter.label,
                  unit=p1.parameter.unit, preset_data=xx, is_setpoint=True)
    y = DataArray(name=p2.name, array_id=p2.name, label=p2.parameter.label,
                  unit=p2.parameter.unit, preset_data=yy, set_arrays=(x,), is_setpoint=True)
    if isinstance(measure_names, str):
        measure_names = [measure_names]
        if preset_data is not None:
            preset_data = [preset_data]
    mnamesx = measure_names
    measure_names = []
    for p in mnamesx:
        if isinstance(p, str):
            measure_names += [p]
        else:
            # assume p is a Parameter
            measure_names += [p.full_name]
    dd = new_data(arrays=(), location=location, loc_record=loc_record)
    for idm, mname in enumerate(measure_names):
        z = DataArray(name=mname, array_id=mname, label=mname,
                      preset_data=np.copy(zz), set_arrays=(x, y))
        dd.add_array(z)
        if preset_data is not None:
            getattr(dd, mname).ndarray = np.array(preset_data[idm])
    dd.add_array(x)
    dd.add_array(y)

    dd.last_write = -1

    if return_names:
        return dd, (set_names, measure_names)
    else:
        return dd


def test_makeDataSet2D():
    from qcodes import ManualParameter
    p = ManualParameter('dummy')
    p2 = ManualParameter('dummy2')
    ds = makeDataSet2D(p[0:10:1], p2[0:4:1], ['m1', 'm2'])

    _ = diffDataset(ds)


def test_makeDataSet1Dplain():
    x = np.arange(0, 10)
    y = np.vstack((x - 1, x + 10))
    ds = makeDataSet1Dplain('x', x, ['y1', 'y2'], y)

#%%


def compare_dataset_metadata(dataset1, dataset2, metakey='allgatevalues'):
    """ Compare metadata from two different datasets.

    Outputs the differences in metadata from dataset1 to dataset2.
    For now, only comparisons for the key 'allgatevalues' has been implemented.

    Args:
        dataset1 (DataSet): first dataset to compare
        dataset2 (DataSet): second dataset to compare
        metakey (str): key in the DataSet metadata to compare
    """
    if (metakey not in dataset1.metadata) or (metakey not in dataset2.metadata):
        print('key %s not in dataset metadata' % metakey)
        return
    if metakey == 'allgatevalues':
        for ikey, value1 in dataset1.metadata[metakey].items():
            if ikey in dataset2.metadata[metakey]:
                value2 = dataset2.metadata[metakey][ikey]
                if value1 != value2:
                    print('Gate %s from %.1f to %.1f' % (ikey, value1, value2))
            else:
                print('Gate %s not in second dataset' % (ikey))
    else:
        raise Exception('metadata key not yet supported')


def test_compare():
    import qcodes.tests.data_mocks
    ds = qcodes.tests.data_mocks.DataSet2D()
    compare_dataset_metadata(ds, ds)

#%%


def test_numpy_on_dataset():
    import qcodes.tests.data_mocks
    alldata = qcodes.tests.data_mocks.DataSet2D()
    X = alldata.z
    _ = np.array(X)
    s = np.linalg.svd(X)
    # print(s)


if __name__ == '__main__':
    import numpy as np
    import qcodes.tests.data_mocks

    test_numpy_on_dataset()
    test_makeDataSet2D()
