""" Utilities to work with data and datasets """

import numpy as np
import scipy
import os
import sys
import time
import qcodes
import datetime
import warnings
import pickle
import logging
import matplotlib.pyplot as plt

from qcodes import DataArray, new_data
from qtt import pgeometry
import qtt.algorithms.generic
from qcodes.plots.qcmatplotlib import MatPlot

#%% Serialization tools


def _DataArray_to_dictionary(da, include_data=True):
    """ Convert DataArray to dictionary

    Args:
        da (DataArray): data to convert
        include_data (bool): If True then include the .ndarray field
    Returns:
        dict: dictionary containing the serialized data

    """
    fields = ['label', 'name', 'unit', 'is_setpoint', 'full_name', 'array_id', 'shape', 'set_arrays']
    if include_data:
        fields += ['ndarray']
    dct = dict([(f, getattr(da, f)) for f in fields])
    dct['set_arrays'] = tuple([x.array_id for x in dct['set_arrays']])
    return dct


def _dictionary_to_DataArray(array_dictionary):
    
    array_id=array_dictionary.get('array_id', array_dictionary['name'])
    preset_data=array_dictionary['ndarray']
    data_array=qcodes.DataArray(name=array_dictionary['name'],
                                full_name=array_dictionary['full_name'],
                                label=array_dictionary['label'],
                                unit=array_dictionary['unit'],
                                is_setpoint=array_dictionary['is_setpoint'],
                                shape=array_dictionary['shape'],
                                array_id=array_id,
                                preset_data=preset_data
                                )
    return data_array

def dictionary_to_DataSet(data_set_dictionary):
    """ Convert dictionary to DataSet 

    Args:
        data_set_dictionary (dict): data to convert
    Returns:
        DataSet: converted data 
    """
    dataset = qcodes.new_data()
    dataset.metadata.update(data_set_dictionary['metadata'])
    
    for array_key, array_dct in data_set_dictionary['arrays'].items():
        data_array = _dictionary_to_DataArray(array_dct)        
        dataset.add_array(data_array)
    
    # update set arrays
    for array_key, array_dct in data_set_dictionary['arrays'].items():
        set_arrays_names=array_dct['set_arrays']
        set_arrays=tuple([dataset.arrays[name] for name in set_arrays_names])
        dataset.arrays[array_key].set_arrays = set_arrays
      
    return dataset

def DataSet_to_dictionary(data_set, include_data=True, include_metadata=True):
    """ Convert DataSet to dictionary

    Args:
        data_set (DataSet): data to convert
        include_data (bool): If True then include the .ndarray field
        include_metadata (bool): If True then include the metadata
    Returns:
        dict: dictionary containing the serialized data
    """
    d = {'extra': {}, 'metadata': None, 'arrays': {}}
    for array_id in data_set.arrays.keys():
        da = data_set.arrays[array_id]

        d['arrays'][array_id] = _DataArray_to_dictionary(da, include_data)

    if include_metadata:
        d['metadata'] = data_set.metadata

    d['extra']['location'] = data_set.location
    d['extra']['_version'] = qtt.__version__
    return d


def test_dataset_to_dictionary():
    import qcodes.tests.data_mocks
    input_dataset = qcodes.tests.data_mocks.DataSet2D()
    data_set_dictionary = DataSet_to_dictionary(input_dataset, include_data=False, include_metadata=False)
    assert(data_set_dictionary['metadata'] is None)
    data_set_dictionary = DataSet_to_dictionary(input_dataset, include_data=True, include_metadata=True)
    assert('metadata' in data_set_dictionary)

    dataset2 = dictionary_to_DataSet(data_set_dictionary)
    assert(dataset2.default_parameter_name()==input_dataset.default_parameter_name())
#%%


def get_dataset(ds):
    """ Get a dataset from a results dictionary, a string or a dataset

    Args:
        ds (str, dict or DataSet): either location of dataset, the dataset itself or a calibration structure

    Returns:
        ds (DataSet)
    """
    if isinstance(ds, dict):
        ds = ds.get('dataset', None)
    if ds is None:
        return None
    if isinstance(ds, str):
        ds = qtt.data.load_dataset(ds)
    return ds


def load_dataset(location, io=None, verbose=0):
    """ Load a dataset from storage

    An attempt is made to automatically detect the formatter. Supported are currently GNUPlotFormat and HDF5Format

    Args:
        location (str): either the relative or full location
        io (None or qcodes.DiskIO): 
    Returns:
        dataset (DataSet or None)
    """

    if io is None:
        io = qcodes.DataSet.default_io
    formatters = [qcodes.DataSet.default_formatter]

    try:
        from qcodes.data.hdf5_format_hickle import HDF5FormatHickle
        formatters += [HDF5FormatHickle()]
    except:
        pass

    from qcodes.data.hdf5_format import HDF5Format
    formatters += [HDF5Format()]

    from qcodes.data.gnuplot_format import GNUPlotFormat
    formatters += [GNUPlotFormat()]

    data = None
    for ii, hformatter in enumerate(formatters):
        try:
            if verbose:
                print('%d: %s' % (ii, hformatter))
            data = qcodes.load_data(location, formatter=hformatter, io=io)
            if len(data.arrays) == 0:
                data = None
                raise Exception('empty dataset, probably a HDF5 format misread by GNUPlotFormat')
            logging.debug('load_data: loaded %s with %s' % (location, hformatter))
        except Exception as ex:
            logging.info('load_data: location %s: failed for formatter %d: %s' % (location, ii, hformatter))
            if verbose:
                print(ex)
            pass
        finally:
            if data is not None:
                if isinstance(hformatter, GNUPlotFormat):
                    # workaround for bug in GNUPlotFormat not saving the units
                    if '__dataset_metadata' in data.metadata:
                        dataset_meta = data.metadata['__dataset_metadata']
                        for key, array_metadata in dataset_meta['arrays'].items():
                            if key in data.arrays:
                                if data.arrays[key].unit is None:
                                    if verbose:
                                        print('load_dataset: updating unit for %s' % key)
                                    data.arrays[key].unit = array_metadata['unit']

                if verbose:
                    print('succes with formatter %s' % hformatter)
                break
    if verbose:
        if data is None:
            print('could not load data from %s, returning None' % location)
    return data


def test_load_dataset(verbose=0):
    import tempfile
    h = qcodes.data.hdf5_format.HDF5Format()
    g = qcodes.data.gnuplot_format.GNUPlotFormat()

    io = qcodes.data.io.DiskIO(tempfile.mkdtemp())
    dd = []
    name = qcodes.DataSet.location_provider.base_record['name']
    for jj, fmt in enumerate([g, h]):
        ds = qcodes.tests.data_mocks.DataSet2D(name='format%d' % jj)
        ds.formatter = fmt
        ds.io = io
        ds.add_metadata({'1': 1, '2': [2, 'x'], 'np': np.array([1, 2.])})
        ds.write(write_metadata=True)
        dd.append(ds.location)
        time.sleep(.1)
    qcodes.DataSet.location_provider.base_record['name'] = name

    for ii, location in enumerate(dd):
        if verbose:
            print('load %s' % location)
        r = load_dataset(location, io=io)
        if verbose:
            print(r)


#%% Monkey patch qcodes to store latest dataset
from functools import wraps


@qtt.utilities.tools.deprecated
def store_latest_decorator(function, obj):
    """ Decorator to store latest result of a function in an object """
    if not hasattr(obj, '_latest'):
        obj._latest = None

    @wraps(function)
    def wrapper(*args, **kwargs):
        ds = function(*args, **kwargs)
        obj._latest = ds  # store the latest result
        return ds
    wrapper._special = 'yes'
    return wrapper


def get_latest_dataset():
    """ Return latest dataset that was created """
    return getattr(qcodes.DataSet._latest, None)


def add_comment(txt, dataset=None, verbose=0):
    """ Add a comment to a DataSet

    Args:
        comment (str): comment to be added to the DataSet metadata
        dataset (None or DataSet): DataSet to add the comments to

    """
    if dataset is None:
        if hasattr(qcodes.DataSet, '_latest_datasets'):
            try:
                dataset = qcodes.DataSet._latest_datasets[0]
            except:
                pass
        else:
            raise NotImplementedError('dataset not specified and _latest_datasets not available')
            dataset = qcodes.DataSet._latest
    if dataset is None:
        raise Exception('no DataSet to add comments to')

    dataset.add_metadata({'comment': txt})
    if verbose:
        print('added comments to DataSet %s' % dataset.location)


def test_add_comment():
    import qcodes.tests.data_mocks

    ds0 = qcodes.tests.data_mocks.DataSet2D()
    ds = qcodes.tests.data_mocks.DataSet2D()
    try:
        add_comment('hello world')
    except NotImplementedError as ex:
        ds.metadata['comment'] = 'hello world'
        pass
    add_comment('hello world 0', ds0)
    assert(ds.metadata['comment'] == 'hello world')
    assert(ds0.metadata['comment'] == 'hello world 0')


if __name__ == '__main__':
    test_add_comment()

    # ds=qcodes.tests.data_mocks.DataSet2D()
    #print('latest dataset %s' % (qcodes.DataSet._latest_datasets[0], ))

#%%


def datasetCentre(ds, ndim=None):
    """ Return centre position for dataset
    Args:
        ds (DataSet):
    Returns:
        cc (list of floats): centre position
    """
    p = ds.default_parameter_array()
    if ndim is None:
        ndim = len(p.set_arrays)
    if ndim == 1:
        x = p.set_arrays[0]
        mx = np.nanmean(x)
        cc = [mx]
    else:
        x = p.set_arrays[1]
        y = p.set_arrays[0]
        mx = np.nanmean(x)
        my = np.nanmean(y)
        cc = [mx, my]
    return cc


def test_dataset():
    import qcodes.tests.data_mocks
    ds = qcodes.tests.data_mocks.DataSet2D()
    cc = datasetCentre(ds)
    assert(cc[0] == 1.5)
    zz = dataset_labels(ds)


def drawCrosshair(ds, ax=None, ndim=None):
    """ Draw a crosshair on the centre of the dataset

    Args:
        ds (DataSet):
        ax (None or matplotlib axis handle)
        ndim (None or int): dimension of dataset
    """

    cc = datasetCentre(ds, ndim=ndim)

    if ax is None:
        ax = plt.gca()
    ax.axvline(x=cc[0], linestyle=':', color='c')
    if len(cc) == 2:
        ax.axhline(y=cc[1], linestyle=':', color='c')


#%%


def dataset2image(dataset, arrayname=None, unitsperpixel=None, mode='pixel'):
    """ Extract image from a dataset

    Args:
        dataset (DataSet): structure with 2D data
        arrayname (None or str): nafme of array to select
        mode (str): if value is 'pixel' then the image is converted so that 
             it is in conventional coordinates, e.g. the step values
             (vertical axis) go from low to high (bottom to top).

    Returns:
        im (numpy array)
        tr (image_transform object)

    """
    if arrayname is None:
        arrayname = dataset.default_parameter_name()
    tr = image_transform(dataset, arrayname=arrayname,
                         mode=mode, unitsperpixel=unitsperpixel)
    im = None
    if arrayname is not None:
        imraw = dataset.arrays[arrayname].ndarray
        im = tr._transform(imraw)
    return im, tr


def dataset2image2(dataset, arrayname=None):
    """ Extract image from dataset

    Args:
        dataset (DataSet): measured data
    Returns:
        imraw (array): raw image
        impixel (array): image in pixel coordinates
        tr (image_transform object): transformation object

    See also: dataset2image
    """
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
        dataset, verbose=0, arrayname=arrayname)
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
    ''' Parse a dataset into the x and y scan values

    Returns:
        x (array)
        y (array)
    '''
    y = alldata.default_parameter_array()
    x = y.set_arrays[0]
    return x, y


def dataset_labels(alldata, tag=None, add_unit=False):
    """ Return label for axis of dataset

    Args:
        ds (DataSet): dataset
        tag (str or None): can be 'x', 'y' or 'z' or the index of the axis
        add_units (bool): If True then add units
    """
    if tag == 'y' or tag==0:
        d = alldata.default_parameter_array()
        array = d.set_arrays[0]
    elif tag == 'x' or tag==1:
        d = alldata.default_parameter_array()
        array = d.set_arrays[1]
    elif tag is None or tag == 'z':
        array = alldata.default_parameter_array()
    else:
        raise Exception('invalid value %s for tag' % (tag, ))
    label= array.label
    
    if  add_unit:
        label+= ' [' + str(array.unit) + ']'
    return label

    
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


def diffDataset(alldata, diff_dir='y', sigma=2, fig=None, meas_arr_name='measured'):
    """ Differentiate a dataset and plot the result.

    Args:
        alldata (qcodes DataSet)
        diff_dir (str): direction to differentiate in
        meas_arr_name (str): name of the measured array to be differentiated
        fig (int): the number for the figure to plot
        sigma (float):  parameter for gaussian filter kernel
    """
    meas_arr_name = alldata.default_parameter_name(meas_arr_name)
    meas_array = alldata.arrays[meas_arr_name]
    imx = qtt.utilities.tools.diffImageSmooth(meas_array.ndarray, dy=diff_dir, sigma=sigma)
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
    """ Return the sweepgate in a scanjob """
    g = scanjob['sweepdata'].get('param', None)
    if isinstance(g, str):
        return g
    if isinstance(g, qcodes.Parameter):
        return g.name
    g = scanjob['sweepdata'].get('gate', None)
    if g is None:
        g = scanjob['sweepdata'].get('gates', [None])[0]
    return g


def stepgate(scanjob):
    """ Return the step gate in a scanjob """
    g = scanjob['stepdata'].get('param', None)
    if isinstance(g, str):
        return g
    if isinstance(g, qcodes.Parameter):
        return g.name
    g = scanjob['stepdata'].get('gate', None)
    if g is None:
        g = scanjob['stepdata'].get('gates', [None])[0]
    return g


def show2D(dd, impixel=None, im=None, fig=101, verbose=1, dy=None, sigma=None, colorbar=False, title=None, midx=2, units=None):
    """ Show result of a 2D scan 

    Args:
        dd (DataSet)
        impixel (array or None)
        im (array or None)
    """
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
    else:
        pass

    labels = [s.name for s in array.set_arrays]

    xx = extent
    xx = tr.matplotlib_image_extent()
    ny = vstep.size
    nx = vsweep.size

    im = qtt.utilities.tools.diffImageSmooth(impixel, dy=dy, sigma=sigma)
    if verbose:
        print('show2D: nx %d, ny %d' % (nx, ny,))

    if verbose >= 2:
        print('extent: %s' % xx)
    if units is None:
        unitstr = ''
    else:
        unitstr = ' (%s)' % units
    if fig is not None:
        scanjob = dd.metadata.get('scanjob', dict())
        pgeometry.cfigure(fig)
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

        if scanjob.get('stepdata', None) is not None:
            if units is None:
                plt.ylabel('%s' % stepgate(scanjob))
            else:
                plt.ylabel('%s (%s)' % (stepgate(scanjob), units))

        if not title is None:
            plt.title(title)
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
        g0 (str): step gate (array_id)
        g1 (str): sweep gate (array_id)
        vstep (array): step values
        vsweep (array): sweep values
        arrayname (string): identifier of the main array 

    """

    if arrayname is None:
        arrayname = alldata.default_parameter_name()

    A = alldata.arrays[arrayname]

    g0 = A.set_arrays[0].array_id
    g1 = A.set_arrays[1].array_id
    vstep = np.array(A.set_arrays[0])
    vsweep = np.array(A.set_arrays[1])[0]
    extent = [vsweep[0], vsweep[-1], vstep[0], vstep[-1]]  # change order?

    if verbose:
        print('2D scan: gates %s %s' % (g0, g1))
    return extent, g0, g1, vstep, vsweep, arrayname


if __name__ == '__main__' and 0:
    test_dataset()

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
        self.arrayname = arrayname
        extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
            dataset, arrayname=self.arrayname)
        self.vstep = vstep
        self.vsweep = vsweep

        self._istep = dataset_get_istep(dataset)
        if self.verbose:
            print('image_transform: istep %.2f, unitsperpixel %s' % (self._istep, unitsperpixel))

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

        self._imraw = np.array(dataset.arrays[arrayname])

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

        if verbose:
            print('image_transform: tr._imraw.shape %s' % (self._imraw.shape, ))
            print('image_transform: tr._im.shape %s' % (self._im.shape, ))
        self._im = self._transform(self._imraw)
        self.Hi = np.linalg.inv(self.H)

    def image(self):
        return self._im

    def istep_sweep(self):
        return np.mean(np.diff(self.vsweep))

    def istep_step(self):
        return np.mean(np.diff(self.vstep))

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
        ptx = pgeometry.projectiveTransformation(
            self.Hi, np.array(pt).astype(float))

        extent, g0, g1, vstep, vsweep, arrayname = dataset2Dmetadata(
            self.dataset, arrayname=self.arrayname, verbose=0)
        nx = vsweep.size
        ny = vstep.size

        xx = extent
        x = ptx
        nn = pt.shape[1]
        ptx = np.zeros((2, nn))

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
            self.dataset, arrayname=self.arrayname, verbose=0)
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

        ptpixel = pgeometry.projectiveTransformation(
            self.H, np.array(ptpixel).astype(float))

        return ptpixel


def test_image_transform(verbose=0):
    from qcodes.tests.data_mocks import DataSet2D
    ds = DataSet2D()
    tr = image_transform(ds)
    im = tr.image()
    if verbose:
        print('transform: im.shape %s' % (str(im.shape),))
    tr = image_transform(ds, unitsperpixel=[None, 2])
    im = tr.image()
    if verbose:
        print('transform: im.shape %s' % (str(im.shape),))


if __name__ == '__main__':
    test_image_transform()

#%%


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
        else:
            data2 = None
    return data2


def _data_extension():
    return 'pickle'


def load_data(mfile: str):
    ''' Load data from specified file '''
    # return hickle.load(mfile)
    ext = _data_extension()
    if ext is not None:
        if not mfile.endswith(ext):
            mfile = mfile + '.' + ext
    return pickleload(mfile)


def write_data(mfile: str, data):
    ''' Write data to specified file '''
    ext = _data_extension()
    if ext is not None:
        if not mfile.endswith(ext):
            mfile = mfile + '.' + ext
    if isinstance(data, qcodes.DataSet):
        data = qtt.utilities.tools.stripDataset(data)

    with open(mfile, 'wb') as fid:
        pickle.dump(data, fid)


@qtt.utilities.tools.rdeprecated(expire='1-1-2019')
def loadDataset(path):
    ''' Wrapper function

    :param path: filename without extension
    :returns dateset, metadata: 
    '''
    dataset = qcodes.load_data(path)

    mfile = os.path.join(path, 'qtt-metadata')
    metadata = load_data(mfile)
    return dataset, metadata


@qtt.utilities.tools.rdeprecated(expire='1-1-2019')
def writeDataset(path, dataset, metadata=None):
    ''' Wrapper function

    :param path: filename without extension
    '''

    dataset = qtt.utilities.tools.stripDataset(dataset)

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


def dateString(t=None):
    """ Return date string with timezone """
    if t is None:
        t = datetime.datetime.now()
    return t.strftime('%Y-%m-%d %H:%M:%S.%f %z %Z').strip()


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

    ext = _data_extension()
    basename = '%s' % (dstr,)
    if bname is not None:
        basename = '%s-' % bname + basename
    if not outputdir is None:
        qtt.utilities.tools.mkdirc(os.path.join(outputdir, tag))
    pfile = os.path.join(outputdir, tag, basename + '.' + ext)
    return pfile


def loadExperimentData(outputdir, tag, dstr):
    path = experimentFile(outputdir, tag=tag, dstr=dstr)
    logging.info('loadExperimentdata %s' % path)
    dataset = pgeometry.load(path)

    dataset = load_data(path)
    return dataset


def saveExperimentData(outputdir, dataset, tag, dstr):
    path = experimentFile(outputdir, tag=tag, dstr=dstr)
    logging.info('saveExperimentData %s' % path)
    write_data(path, dataset)


def makeDataSet1Dplain(xname, x, yname, y=None, xunit=None, yunit=None, location=None, loc_record=None):
    ''' Make DataSet with one 1D array and one setpoint array

    Arguments:
        xname (string): the name of the setpoint array
        x (array): the setpoint data
        yname (str or list): the name of the measured array
        y (array): the measured data
    '''
    xx = np.array(x)
    yy = np.NaN * np.ones(xx.size) if y is None else np.array(y)
    x = DataArray(name=xname, array_id=xname, preset_data=xx, unit=xunit, is_setpoint=True)
    dd = new_data(arrays=(), location=location, loc_record=loc_record)
    dd.add_array(x)
    if isinstance(yname, str):
        y = DataArray(name=yname, array_id=yname, preset_data=yy, unit=yunit, set_arrays=(x,))
        dd.add_array(y)
    else:
        for ii, name in enumerate(yname):
            y = DataArray(name=name, array_id=name, preset_data=yy[ii], unit=yunit, set_arrays=(x,))
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
    measure_units = []
    for p in mnamesx:
        if isinstance(p, str):
            measure_names += [p]
            measure_units += [None]
        else:
            # assume p is a Parameter
            measure_names += [p.full_name]
            measure_units += [p.unit]

    dd = new_data(arrays=(), location=location, loc_record=loc_record)

    for idm, mname in enumerate(measure_names):
        ytmp = DataArray(name=mname, array_id=mname, label=mname,
                         preset_data=np.copy(yy), set_arrays=(x,), unit=measure_units[idm])
        dd.add_array(ytmp)
        if y is not None:
            getattr(dd, mname).ndarray = np.array(preset_data[idm])
    dd.add_array(x)

    if return_names:
        set_names = x.name
        return dd, (set_names, measure_names)
    else:
        return dd


def makeDataSet2Dplain(xname, x, yname, y, zname='measured', z=None, xunit=None, yunit=None, zunit=None, location=None, loc_record=None):
    ''' Make DataSet with one 2D array and two setpoint arrays

    Arguments:
        xname, yname (string): the name of the setpoint array
        x, y (array): the setpoint data
        zname (str or list): the name of the measured array
        z (array or list): the measured data
    '''
    yy = np.array(y)
    xx0 = np.array(x)
    xx = np.tile(xx0, [yy.size, 1])
    zz = np.NaN * np.ones((yy.size, xx0.size))
    ya = DataArray(name=yname, array_id=yname, preset_data=yy,
                   unit=yunit, is_setpoint=True)
    xa = DataArray(name=xname, array_id=xname, preset_data=xx,
                   unit=xunit, set_arrays=(ya,), is_setpoint=True)
    dd = new_data(arrays=(), location=location, loc_record=loc_record)
    if isinstance(zname, str):
        zname = [zname]
        if isinstance(z, np.ndarray):
            z = [z]
    for ii, name in enumerate(zname):
        za = DataArray(name=name, array_id=name, label=name,
                       preset_data=np.copy(zz), unit=zunit, set_arrays=(ya, xa))
        dd.add_array(za)
        if z is not None:
            getattr(dd, name).ndarray = np.array(z[ii])
    dd.add_array(xa)
    dd.add_array(ya)

    dd.last_write = -1

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


def compare_dataset_metadata(dataset1, dataset2, metakey='allgatevalues', verbose=1):
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
    compare_dataset_metadata(ds, ds, verbose=0)

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
    test_makeDataSet1Dplain()
    test_compare()
