""" Utilities to work with data and datasets """

import datetime
import logging
import os
import pickle
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import qcodes
import qcodes_loop
import scipy
from qcodes_loop.data.data_array import DataArray
from qcodes_loop.data.data_set import DataSet, new_data
from qcodes_loop.plots.qcmatplotlib import MatPlot

import qtt.algorithms.generic
import qtt.utilities.json_serializer
from qtt import pgeometry

logger = logging.getLogger(__name__)


def load_example_dataset(filename: str, verbose: int = 0) -> Optional[DataSet]:
    """ Return an example dataset from qtt

    Args:
        filename: Name of the dataset
        verbose: Verbosity level
    Returns:
        Example dataset or None of no dataset can be found
    """
    exampledatadir = os.path.join(qtt.__path__[0], 'exampledata')  # type: ignore # mypy issue #1422

    dataset = qtt.data.load_dataset(os.path.join(exampledatadir, filename), verbose=verbose)
    return dataset


def _data_array_to_dictionary(data_array, include_data=True):
    """ Convert DataArray to a dictionary.

    Args:
        data_array (DataArray): The data to convert.
        include_data (bool): If True then include the ndarray field.

    Returns:
        dict: A dictionary containing the serialized data.
    """
    keys = ['label', 'name', 'unit', 'is_setpoint', 'full_name', 'array_id', 'shape']
    if include_data:
        keys.append('ndarray')

    data_dictionary = {key: getattr(data_array, key) for key in keys}
    data_dictionary['set_arrays'] = tuple(array.array_id for array in data_array.set_arrays)

    return data_dictionary


def _dictionary_to_data_array(array_dictionary):
    preset_data = array_dictionary['ndarray']
    array_id = array_dictionary.get('array_id', array_dictionary['name'])
    array_name = array_dictionary['name']
    if array_name is None:
        array_name = array_id
    array_full_name = array_dictionary['full_name']
    if array_full_name is None:
        array_full_name = array_name
    data_array = DataArray(name=array_name,
                           full_name=array_dictionary['full_name'],
                           label=array_dictionary['label'],
                           unit=array_dictionary['unit'],
                           is_setpoint=array_dictionary['is_setpoint'],
                           shape=tuple(array_dictionary['shape']),
                           array_id=array_id,
                           preset_data=preset_data)
    return data_array


def dictionary_to_dataset(data_dictionary: dict) -> DataSet:
    """ Convert dictionary to DataSet.

    Args:
        data_dictionary: data to convert

    Returns:
        DataSet with converted data.
    """
    dataset = new_data()
    dataset.metadata.update(data_dictionary['metadata'])

    for array_key, array_dict in data_dictionary['arrays'].items():
        data_array = _dictionary_to_data_array(array_dict)
        dataset.add_array(data_array)

    for array_key, array_dict in data_dictionary['arrays'].items():
        set_arrays_names = array_dict['set_arrays']
        set_arrays = tuple(dataset.arrays[name] for name in set_arrays_names)
        dataset.arrays[array_key].set_arrays = set_arrays

    return dataset


def dataset_to_dictionary(data_set: DataSet, include_data: bool = True,
                          include_metadata: bool = True) -> Dict[str, Any]:
    """ Convert DataSet to dictionary.

    Args:
        data_set: The data to convert.
        include_data: If True then include the ndarray field.
        include_metadata: If True then include the metadata.

    Returns:
        Dictionary containing the serialized data.
    """
    data_dictionary: Dict[str, Any] = {'extra': {}, 'metadata': None, 'arrays': {}}

    for array_id, data_array in data_set.arrays.items():
        data_dictionary['arrays'][array_id] = _data_array_to_dictionary(data_array, include_data)

    data_dictionary['extra']['location'] = data_set.location
    data_dictionary['extra']['_version'] = qtt.__version__
    if include_metadata:
        data_dictionary['metadata'] = data_set.metadata

    return data_dictionary


def get_dataset(dataset_handle):
    """ Get a dataset from a results dictionary, a string or a dataset.

    Args:
        dataset_handle (str, dict or DataSet): either location of dataset,
            the dataset itself or a calibration structure.

    Returns:
        DataSet: The dataset from the handle.
    """
    if isinstance(dataset_handle, dict):
        dataset_handle = dataset_handle.get('dataset', None)

    if isinstance(dataset_handle, str):
        return qtt.data.load_dataset(dataset_handle)

    if isinstance(dataset_handle, DataSet):
        return dataset_handle

    raise ValueError(f'Invalid dataset argument type ({type(dataset_handle)})!')


def load_dataset(location, io=None, verbose=0):
    """ Load a dataset from storage

    An attempt is made to automatically detect the formatter. Supported are currently qcodes GNUPlotFormat,
    qcodes HDF5Format and json format.

    Args:
        location (str): either the relative or full location
        io (None or qcodes_loop.data.io.DiskIO):
    Returns:
        dataset (DataSet or None)
    """

    if io is None:
        io = DataSet.default_io
    formatters = [DataSet.default_formatter]

    from qcodes_loop.data.hdf5_format import HDF5FormatMetadata
    formatters += [HDF5FormatMetadata()]
    try:
        from qcodes_loop.data.hdf5_format_hickle import HDF5FormatHickle
        formatters += [HDF5FormatHickle()]
    except ImportError as ex:
        logging.info(f'HDF5FormatHickle not available {ex}')

    from qcodes_loop.data.hdf5_format import HDF5Format
    formatters += [HDF5Format()]

    from qcodes_loop.data.gnuplot_format import GNUPlotFormat
    formatters += [GNUPlotFormat()]

    data = None

    if location.endswith('.json'):
        dataset_dictionary = qtt.utilities.json_serializer.load_json(location)
        data = qtt.data.dictionary_to_dataset(dataset_dictionary)
    else:
        # assume we have a QCoDeS dataset
        for ii, hformatter in enumerate(formatters):
            try:
                if verbose:
                    print('%d: %s' % (ii, hformatter))
                data = qcodes_loop.data.data_set.load_data(location, formatter=hformatter, io=io)
                if len(data.arrays) == 0:
                    data = None
                    raise Exception('empty dataset, probably a HDF5 format misread by GNUPlotFormat')
                logging.debug('load_data: loaded %s with %s' % (location, hformatter))
            except Exception as ex:
                logging.info('load_data: location %s: failed for formatter %d: %s' % (location, ii, hformatter))
                if verbose:
                    print(ex)
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


def default_setpoint_array(dataset, measured_name='measured'):
    """ Return the default setpoint array for a dataset """
    setpoint_array = dataset.default_parameter_array(measured_name).set_arrays[0]
    return setpoint_array


# %% Monkey patch qcodes to store latest dataset


def add_comment(txt, dataset=None, verbose=0):
    """ Add a comment to a DataSet

    Args:
        comment (str): comment to be added to the DataSet metadata
        dataset (None or DataSet): DataSet to add the comments to

    """
    if dataset is None:
        if hasattr(DataSet, '_latest_datasets'):
            try:
                dataset = DataSet._latest_datasets[0]
            except BaseException:
                pass
        else:
            raise NotImplementedError('dataset not specified and _latest_datasets not available')
    if dataset is None:
        raise Exception('no DataSet to add comments to')

    dataset.add_metadata({'comment': txt})
    if verbose:
        print('added comments to DataSet %s' % dataset.location)


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


def drawCrosshair(ds, ax=None, ndim=None, **kwargs):
    """ Draw a crosshair on the centre of the dataset

    Args:
        ds (DataSet):
        ax (None or matplotlib axis handle)
        ndim (None or int): dimension of dataset
        kwargs: Arguments passed to the plotting command
    """

    cc = datasetCentre(ds, ndim=ndim)

    if not 'linestyle' in kwargs:
        kwargs['linestyle'] = ':'
    if not 'color' in kwargs:
        kwargs['color'] = 'c'

    if ax is None:
        ax = plt.gca()
    ax.axvline(x=cc[0], **kwargs)
    if len(cc) == 2:
        ax.axhline(y=cc[1], **kwargs)


# %%


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


# %%


def dataset_get_istep(alldata, mode=None):
    """ Return number of mV per pixel in scan """
    try:
        resolution = np.abs(alldata.metadata['scanjob']['stepdata']['step'])
    except BaseException:
        try:
            extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
                alldata, verbose=0, arrayname=None)
            resolution = np.abs(np.nanmean(np.diff(vstep)))
        except BaseException:
            _, _, _, resolution, _ = dataset1Dmetadata(alldata)
    return resolution


def dataset1Ddata(alldata):
    """ Parse a dataset into the x and y scan values

    Returns:
        x (array)
        y (array)
    """
    y = alldata.default_parameter_array()
    x = y.set_arrays[0]
    return x, y


def dataset_labels(alldata: Union[DataSet, DataArray], tag: Optional[Union[str, int]] = None, add_unit: bool = False):
    """ Return label for axis of dataset

    Args:
        alldata: dataset or dataarray
        tag: can be 'x', 'y' or 'z' or the index of the axis. For DataArrays there is only a single axis.
        add_unit: If True then add units
    Returns:
        String with label for the axis
    """

    if isinstance(alldata, DataArray):
        array = alldata
    else:
        dataset_dimension = len(alldata.default_parameter_array().set_arrays)
        d = alldata.default_parameter_array()

        if isinstance(tag, int):
            array = d.set_arrays[0]
        else:
            if dataset_dimension == 1:
                if tag == 'x':
                    array = d.set_arrays[0]
                elif tag is None or tag == 'z' or tag == 'y':
                    array = d
                else:
                    raise Exception('invalid value %s for tag' % (tag,))
            else:
                if tag == 'y':
                    array = d.set_arrays[0]
                elif tag == 'x':
                    array = d.set_arrays[1]
                elif tag is None or tag == 'z':
                    array = d
                else:
                    raise Exception('invalid value %s for tag' % (tag,))

    label = array.label
    if add_unit:
        label += ' [' + str(array.unit) + ']'
    return label


def uniqueArrayName(dataset, name0):
    """ Generate a unique name for a DataArray in a dataset """
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
    data_arr = DataArray(
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


def plot_dataset(dataset: DataSet, parameter_names: Optional[list] = None, fig: Optional[int] = 1) -> None:
    """ Plot a dataset to matplotlib figure window

    Args:
        dataset: DataSet to be plotted
        parameter_names: List of arrays to be plotted
        fig: Specification if Matplotlib figure window

    """
    if parameter_names is None:
        parameter_names = [dataset.default_parameter_name()]

    if parameter_names == 'all':
        parameter_names = [name for name in dataset.arrays.keys() if not dataset.arrays[name].is_setpoint]

    default_array = dataset.default_parameter_array()

    if fig:
        plt.figure(fig)
        plt.clf()

        if len(default_array.shape) >= 2:
            if len(parameter_names) > 1:
                arrays = [dataset.default_parameter_array(parameter_name) for parameter_name in parameter_names]
                plot_handle = MatPlot(*arrays, num=fig)

            else:
                plot_handle = MatPlot(dataset.default_parameter_array(parameter_names[0]), num=fig)
        else:
            for idx, parameter_name in enumerate(parameter_names):
                if idx == 0:
                    plot_handle = MatPlot(dataset.default_parameter_array(parameter_name), num=fig)
                else:
                    plot_handle.add(dataset.default_parameter_array(parameter_name,))

# %%


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


def show2D(dd, impixel=None, im=None, fig=101, verbose=1, dy=None,
           sigma=None, colorbar=False, title=None, midx=2, units=None):
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

        if title is not None:
            plt.title(title)
        if colorbar:
            plt.colorbar()
        if verbose >= 2:
            print('show2D: at show')
        try:
            plt.show(block=False)
        except BaseException:
            # ipython backend does not know about block keyword...
            plt.show()

    return xx, vstep, vsweep


# %% Extract metadata


def dataset1Dmetadata(alldata, arrayname=None, verbose=0):
    """ Extract metadata from a 1D scan

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
        print('dataset1Dmetadata: gate %s' % (g0,))
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


# %%


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
                      (self.unitsperpixel,))
            ims, Hs, _ = qtt.algorithms.generic.rescaleImage(
                self._imraw, imextent, mvx=unitsperpixel[0], mvy=unitsperpixel[1])
            self._im = ims
            self.H = Hs @ self.H

        if verbose:
            print('image_transform: tr._imraw.shape %s' % (self._imraw.shape,))
            print('image_transform: tr._im.shape %s' % (self._im.shape,))
        self._im = self._transform(self._imraw)
        self.Hi = np.linalg.inv(self.H)

    def image(self):
        return self._im

    def istep_sweep(self):
        return np.mean(np.diff(self.vsweep))

    def istep_step(self):
        return np.mean(np.diff(self.vstep))

    def scan_resolution(self):
        """ Return the scan resolution in [units]/pixel """
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
            ims, _, _ = qtt.algorithms.generic.rescaleImage(
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


def pickleload(pkl_file):
    """ Load objects from file with pickle """
    with open(pkl_file, 'rb') as output:
        data2 = pickle.load(output)
    return data2


def _data_extension():
    return 'pickle'


def load_data(mfile: str):
    """ Load data from specified file """
    ext = _data_extension()
    if ext is not None:
        if not mfile.endswith(ext):
            mfile = mfile + '.' + ext
    return pickleload(mfile)


def write_data(mfile: str, data):
    """ Write data to specified file """
    ext = _data_extension()
    if ext is not None:
        if not mfile.endswith(ext):
            mfile = mfile + '.' + ext
    if isinstance(data, DataSet):
        data = qtt.utilities.tools.stripDataset(data)

    with open(mfile, 'wb') as fid:
        pickle.dump(data, fid)


def getTimeString(t=None):
    """ Return time string for datetime.datetime object """
    if t is None:
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
    if outputdir is not None:
        qtt.utilities.tools.mkdirc(os.path.join(outputdir, tag))
    pfile = os.path.join(outputdir, tag, basename + '.' + ext)
    return pfile


def determine_parameter_unit(parameter):
    """ Determine unit associated with a parameter

    Arguments:
        parameter (Any): the parameter to get the unit from

    Returns:
        The unit associated with the parameter when the parameter is a qcodes parameter, otherwise None
    """
    if isinstance(parameter, qcodes.Parameter):
        return parameter.unit
    else:
        return None


def _check_parameter(parameter):
    """ Check parameter to be of correct type - used in MakeDataSet1D and MakeDataSet2D.

    Arguments:
        parameter (qcodes.SweepFixedValues): the parameter to check

    Raises:
        TypeError: When parameter does not have attribute 'parameter'
        TypeError: When attribute parameter.parameter is not an instance of qcodes.parameter.
    """
    if hasattr(parameter, 'parameter'):
        if not isinstance(parameter.parameter, qcodes.Parameter):
            raise TypeError('Type of parameter.parameter must be qcodes.Parameter')
    else:
        raise TypeError('Type of parameter must be qcodes.SweepFixedValues')


def _make_data_set(measured_data_list, measurement_list, measurement_unit, location, loc_record, preset_data,
                   setpoints):
    """ Generic code to make the data set for makeDataSet1D, makeDataSet1DPlain, makeDataSet2D, makeDataSet2DPlain

        Warnings logged:
        1. When the shape of the measured data is not matching the shape of the setpoint array. When the
           measured data is a list, each list item must match the shape of the setpoint array.

    Raises:
        ValueError: When the number of measurements names in the measurement_list does not match the number of
                    measurements.
                    If len(measurement_list) > len(measured_data_list) we would otherwise get an
                    IndexError later on. When len(measurement_list) < len(measured_data_list) now a ValueError is
                    raised because a part of the measure data is not stored silently otherwise.

        TypeError: When a measurement name in the measurement list has an invalid type.

    Returns:
        The resulting data set and the measure names list.
    """
    data_set = new_data(arrays=(), location=location, loc_record=loc_record)

    if len(setpoints) > 1:
        set_arrays = (setpoints[0], setpoints[1])
    else:
        set_arrays = (setpoints[0], )

    if measured_data_list is not None:
        if len(measurement_list) != len(measured_data_list):
            raise ValueError(f'The number of measurement names {len(measurement_list)} does not match the number '
                             f'of measurements {len(measured_data_list)}')

    measure_names = []
    measure_units = []
    for parameter in measurement_list:
        if isinstance(parameter, str):
            # parameter is a str
            measure_names += [parameter]
            measure_units += [measurement_unit]
        elif isinstance(parameter, qcodes.Parameter):
            # parameter is a Parameter
            measure_names += [parameter.name]
            measure_units += [parameter.unit]
        else:
            raise TypeError('Type of measurement names must be str or qcodes.Parameter')

    for idm, mname in enumerate(measure_names):
        if measured_data_list is not None and measured_data_list[idm] is not None:
            measured_array = np.array(measured_data_list[idm])
            if measured_array.shape != preset_data.shape:
                logger.warning(f'Shape of measured data {measured_array.shape} does not match '
                               f'setpoint shape {preset_data.shape}')
        else:
            measured_array = np.copy(preset_data)
        preset_data_array = DataArray(name=mname, array_id=mname, label=mname, unit=measure_units[idm],
                                      preset_data=measured_array, set_arrays=set_arrays)
        data_set.add_array(preset_data_array)

    if len(setpoints) > 1:
        data_set.add_array(setpoints[1])
        data_set.add_array(setpoints[0])
    else:
        data_set.add_array(setpoints[0])

    return data_set, measure_names


def makeDataSet1Dplain(xname, x, yname, y=None, xunit=None, yunit=None, location=None, loc_record=None):
    """ Make DataSet with one 1D array and one setpoint array

    Arguments:
        xname (string): the name of the setpoint array
        x (array or ndarray or list): the setpoint data
        yname (str or qcodes.Parameter or list): the name of the measured array
        y (array or ndarray or list): the measured data
        xunit (str or None): optional, the unit of the values stored in x array.
        yunit (str or None): optional, the unit of the values stored in y array.
        location (str, callable, bool or None): If you provide a string,
            it must be an unused location in the io manager.
            Can also be:
            - a callable `location provider` with one required parameter \
              (the io manager), and one optional (`record` dict),        \
              which returns a location string when called.
            - `False` - denotes an only-in-memory temporary DataSet.
        loc_record (dict or None): If location is a callable, this will be
            passed to it as `record`.

    Raises:
        See _make_data_set for the ValueError and TypeError exceptions that can be raised

    Returns:
        The resulting dataset.
    """
    setpoint_data = np.array(x)
    preset_data = np.NaN * np.ones(setpoint_data.size)
    if y is not None:
        if isinstance(y, np.ndarray):
            y = np.array(y, dtype=y.dtype)
        elif isinstance(y, list) and y:
            y = np.array(y, dtype=type(y[0]))
        else:
            y = np.array(y)

    setpoint = DataArray(name=xname, array_id=xname, preset_data=setpoint_data, unit=xunit, is_setpoint=True)

    if isinstance(yname, (str, qcodes.Parameter)):
        measured_data_list = [y]
        measurement_list = [yname]
    else:
        measured_data_list = y
        measurement_list = yname

    measurement_unit = yunit

    data_set, _ = _make_data_set(measured_data_list, measurement_list, measurement_unit, location, loc_record,
                                 preset_data, [setpoint])

    return data_set


def makeDataSet1D(p, yname='measured', y=None, location=None, loc_record=None, return_names=False):
    """ Make DataSet with one or multiple 1D arrays and one setpoint array.

    Arguments:
        p (qcodes.SweepFixedValues): the setpoint array of data
        yname (str or list of str or Parameter or list of Parameter):
            when type is str or list of str : the name of measured array(s)
            when type is parameter or list of parameter: the measured Parameters
        y (array or list of array or None): optional (measured) data to fill the DataSet
        location (str, callable, bool or None): If you provide a string,
            it must be an unused location in the io manager.
            Can also be:
            - a callable `location provider` with one required parameter \
              (the io manager), and one optional (`record` dict),        \
              which returns a location string when called.
            - `False` - denotes an only-in-memory temporary DataSet.
        loc_record (dict or None): If location is a callable, this will be
            passed to it as `record`.
        return_names (bool): if True return array names in output

    Raises:
        See _make_data_set for the ValueError and TypeError exceptions that can be raised
        See _check_parameter for the TypeError exceptions that can be raised

    Returns:
        Depending on parameter return_names
            True: The resulting dataset and a tuple with the names of the added arrays (setpoint and measurements).
            False: The resulting dataset.
    """
    _check_parameter(p)

    setpoint_data = np.array(p)
    preset_data = np.NaN * np.ones(setpoint_data.size)

    setpoint = DataArray(name=p.name, array_id=p.name, label=p.parameter.label,
                         unit=p.parameter.unit, preset_data=setpoint_data, is_setpoint=True)

    if isinstance(yname, (str, qcodes.Parameter)):
        measured_data_list = [y]
        measurement_list = [yname]
    else:
        measured_data_list = y
        measurement_list = yname

    measurement_unit = None

    data_set, measure_names = _make_data_set(measured_data_list, measurement_list, measurement_unit, location,
                                             loc_record, preset_data, [setpoint])

    data_set.metadata['default_parameter_name'] = measure_names[0]
    if return_names:
        set_names = setpoint.name
        return data_set, (set_names, measure_names)
    else:
        return data_set


def makeDataSet2Dplain(xname, x, yname, y, zname='measured', z=None, xunit=None,
                       yunit=None, zunit=None, location=None, loc_record=None):
    """ Make DataSet with one 2D array and two setpoint arrays

    Arguments:
        xname (string): the name of the setpoint x array.
        x (array or ndarray or list): the x setpoint data.
        yname (string): the name of the setpoint y array.
        y (array or ndarray or list): the y setpoint data.
        zname (str or list of str): the name of the measured array.
        z (array or list or None): optional the measured data.
        xunit (str or None): optional, the unit of the values stored in x.
        yunit (str or None): optional, the unit of the values stored in y.
        zunit (str or None): optional, the unit of the measured data.
        location (str, callable, bool or None): If you provide a string,
            it must be an unused location in the io manager.
            Can also be:
            - a callable `location provider` with one required parameter \
                (the io manager), and one optional (`record` dict),      \
                which returns a location string when called.
            - `False` - denotes an only-in-memory temporary DataSet.
        loc_record (dict or None): If location is a callable, this will be
            passed to it as `record`.

    Raises:
        See _make_data_set for the ValueError and TypeError exceptions that can be raised

    Returns:
        The resulting dataset.

    """
    setpoint_datay = np.array(y)
    setpoint_datax = np.array(x)
    setpoint_dataxy = np.tile(setpoint_datax, [setpoint_datay.size, 1])
    preset_data = np.NaN * np.ones((setpoint_datay.size, setpoint_datax.size))
    setpointy = DataArray(name=yname, array_id=yname, preset_data=setpoint_datay,
                          unit=yunit, is_setpoint=True)
    setpointx = DataArray(name=xname, array_id=xname, preset_data=setpoint_dataxy,
                          unit=xunit, set_arrays=(setpointy,), is_setpoint=True)

    if isinstance(zname, (str, qcodes.Parameter)):
        if isinstance(z, np.ndarray):
            measured_data_list = [z]
        else:
            measured_data_list = z
        measurement_list = [zname]
    else:
        measured_data_list = z
        measurement_list = zname

    measurement_unit = zunit

    data_set, _ = _make_data_set(measured_data_list, measurement_list, measurement_unit, location, loc_record,
                                 preset_data, [setpointy, setpointx])

    data_set.last_write = -1

    return data_set


def makeDataSet2D(p1, p2, measure_names='measured', location=None, loc_record=None,
                  preset_data=None, return_names=False):
    """ Make DataSet with one or multiple 2D array and two setpoint arrays.

    If the preset_data is used for multiple 2D arrays, then the order of
    measure_names should match the order of preset_data.

    Args:
        p1 (qcodes.SweepFixedValues): first setpoint array of data
        p2 (qcodes.SweepFixedValues): second setpoint array of data
        measure_names (str or list): name(s) of measured array(s)
        location (str, callable, bool or None): If you provide a string,
            it must be an unused location in the io manager.
            Can also be:
            - a callable `location provider` with one required parameter \
              (the io manager), and one optional (`record` dict),        \
              which returns a location string when called.
            - `False` - denotes an only-in-memory temporary DataSet.
        loc_record (dict or None): If location is a callable, this will be
            passed to it as `record`.
        preset_data (array or ndarray or list or None): optional array to fill the DataSet
        return_names (bool): if True return array names in output

    Raises:
        See _make_data_set for the ValueError and TypeError exceptions that can be raised
        See _check_parameter for the TypeError exceptions that can be raised

    Returns:
        Depending on parameter return_names:
            True: The resulting dataset and a tuple with the names of the added arrays (setpoint and measurements).
            False: The resulting dataset.
    """
    _check_parameter(p1)
    _check_parameter(p2)

    y = p1
    x = p2
    z = preset_data

    setpoint_datay = np.array(y)
    setpoint_datax = np.array(x)
    setpoint_dataxy = np.tile(setpoint_datax, [setpoint_datay.size, 1])
    preset_data = np.NaN * np.ones((setpoint_datay.size, setpoint_datax.size))
    setpointy = DataArray(name=y.name, array_id=y.name, preset_data=setpoint_datay,
                          unit=y.parameter.unit, is_setpoint=True)
    setpointx = DataArray(name=x.name, array_id=x.name, preset_data=setpoint_dataxy,
                          unit=x.parameter.unit, set_arrays=(setpointy,), is_setpoint=True)

    if isinstance(measure_names, (str, qcodes.Parameter)):
        measured_data_list = [z]
        measurement_list = [measure_names]
    else:
        measured_data_list = z
        measurement_list = measure_names

    measurement_unit = None

    data_set, measure_names = _make_data_set(measured_data_list, measurement_list, measurement_unit, location,
                                             loc_record, preset_data, [setpointy, setpointx])

    data_set.last_write = -1

    if return_names:
        set_names = [y.name, x.name]
        return data_set, (set_names, measure_names)
    else:
        return data_set


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
        if verbose:
            print('key %s not in dataset metadata' % metakey)
        return
    if metakey == 'allgatevalues':
        for ikey, value1 in dataset1.metadata[metakey].items():
            if ikey in dataset2.metadata[metakey]:
                value2 = dataset2.metadata[metakey][ikey]
                if value1 != value2:
                    if verbose:
                        print('Gate %s from %.1f to %.1f' % (ikey, value1, value2))
            else:
                if verbose:
                    print('Gate %s not in second dataset' % ikey)
    else:
        raise Exception('metadata key not yet supported')
