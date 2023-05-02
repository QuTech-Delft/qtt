import contextlib
import copy
import datetime
import functools
import importlib
import inspect
import logging
import os
import pickle
import platform
import pprint
import sys
import tempfile
import time
import uuid
import warnings
from collections import OrderedDict
from functools import wraps
from itertools import chain
from typing import Any, Dict, Optional, Tuple, Type, Union

import dateutil
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qcodes
import qcodes_loop
import scipy.ndimage as ndimage
from matplotlib.widgets import Button
from qcodes_loop.data.data_array import DataArray
from qcodes_loop.data.data_set import DataSet

import qtt.pgeometry
from qtt.pgeometry import mkdirc  # import for backwards compatibility
from qtt.pgeometry import mpl2clipboard

# explicit import

try:
    from qcodes_loop.plots.pyqtgraph import QtPlot
except BaseException:
    pass


# do NOT load any other qtt submodules here

try:
    import qtpy.QtCore as QtCore
    import qtpy.QtGui as QtGui
    import qtpy.QtWidgets as QtWidgets
except BaseException:
    pass


def profile_expression(expression: str, N: Optional[int] = 1, gui: str = "snakeviz"):
    """Profile an expression with cProfile and display the results using snakeviz

    Args:
        expression: Code to be profiled
        N: Number of iterations. If None, then automatically determine a suitable number of iterations
        gui: Can be `tuna` or `snakeviz`
    """
    import cProfile  # lazy import
    import subprocess

    tmpdir = tempfile.mkdtemp()
    statsfile = os.path.join(tmpdir, "profile_expression_stats")

    assert isinstance(expression, str), "expression should be a string"

    if N is None:
        t0 = time.perf_counter()
        cProfile.run(expression, filename=statsfile)
        dt = time.perf_counter() - t0
        N = int(1. / max(dt - 0.6e-3, 1e-6))
        if N <= 1:
            print(f"profiling: 1 iteration, {dt:.2f} [s]")
            r = subprocess.Popen([gui, statsfile])
            return r
    else:
        N = int(N)
    print(f"profile_expression: running {N} loops")
    if N > 1:
        loop_expression = f"for ijk_kji_no_name in range({N}):\n"
        loop_expression += "\n".join(["  " + term for term in expression.split("\n")])
        loop_expression += "\n# loop done"
        logging.info(loop_expression)
        expression = loop_expression
    t0 = time.perf_counter()
    cProfile.run(expression, statsfile)
    dt = time.perf_counter() - t0

    print(f"profiling: {N} iterations, {dt:.2f} [s]")
    r = subprocess.Popen([gui, statsfile])
    return r

# %%


class measure_time():
    """ Create context manager that measures execution time and prints to stdout """

    def __init__(self, message: Optional[str] = 'dt: '):
        self.message = message
        self.dt = -1

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    @property
    def current_delta_time(self) -> float:
        """ Return time since start of the context

        Returns:
            Time in seconds
        """
        return time.perf_counter() - self.start_time

    @property
    def delta_time(self) -> float:
        """ Return time spend in the context

        If still in the context, return -1.
        Returns:
            Time in seconds
        """
        return self.dt

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.dt = time.perf_counter() - self.start_time

        if self.message is not None:
            print(f'{self.message} {self.dt:.3f} [s]')


@contextlib.contextmanager
def logging_context(level: int = logging.INFO, logger: Optional[logging.Logger] = None):
    """ A context manager that changes the logging level

    Args:
        level: Logging level to set in the context
        logger: Logger to update, if None then update the default logger

    """
    if logger is None:
        logger = logging.getLogger()
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(level)

    try:
        yield
    finally:
        logger.setLevel(previous_level)


def is_spyder_environment():
    """ Return True if the process is running in a Spyder environment """
    return 'SPY_TESTING' in os.environ


def get_module_versions(modules, verbose=0):
    """ Returns the module version of the given pip packages.

    Args:
        modules ([str]): a list with pip packages, e.g. ['numpy', 'scipy'].
        verbose (int): verbosity (0 == silent).

    Returns:
        r (dict): dictionary with package names and version number for each given module.

    """
    module_versions = {}
    for module in modules:
        try:
            package = importlib.import_module(module)
            version = getattr(package, '__version__', 'none')
            module_versions[module] = version
        except ModuleNotFoundError:
            module_versions[module] = 'none'
        if verbose:
            print(f'module {package} {version}')
    return module_versions


def get_git_versions(repos, get_dirty_status=False, verbose=0):
    return {}, {}


def get_python_version(verbose=0):
    """ Returns the python version."""
    version = sys.version
    if verbose:
        print('Python', version)
    return version


def code_version(repository_names=None, package_names=None, get_dirty_status=False, verbose=0):
    """ Returns the python version, module version for; numpy, scipy, qupulse
        and the git guid and dirty status for; qcodes, qtt, spin-projects and pycqed,
        if present on the machine. NOTE: currently the dirty status is not working
        correctly due to a bug in dulwich...

    Args:
        repository_names ([str]): a list with repositories, e.g. ['qtt', 'qcodes'].
        package_names ([str]): a list with pip packages, e.g. ['numpy', 'scipy'].
        get_dirty_status (bool): selects whether the local code has changed for the repositories.
        verbose (int): output level.

    Returns:
        status (dict): python, modules and git repos status.

    """
    _default_git_versions = ['qcodes', 'qtt', 'projects', 'pycqed']
    _default_module_versions = ['numpy', 'scipy', 'qupulse', 'skimage']
    if not repository_names:
        repository_names = _default_git_versions
    if not package_names:
        package_names = _default_module_versions
    result = {}
    repository_stats, dirty_stats = get_git_versions(repository_names, get_dirty_status, verbose)
    result['python'] = get_python_version(verbose)
    result['git'] = repository_stats
    result['version'] = get_module_versions(package_names, verbose)
    result['timestamp'] = datetime.datetime.now().isoformat()  # ISO 8601
    result['system'] = {'node': platform.node()}
    if get_dirty_status:
        result['dirty'] = dirty_stats

    return result


# %% Debugging


def deprecated(func):
    """ This is a decorator which can be used to mark functions as deprecated. It will result in a warning being
    emitted when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        try:
            filename = inspect.getfile(func)
        except BaseException:
            filename = '?'
        try:
            lineno = inspect.getlineno(func)
        except BaseException:
            lineno = -1
        warnings.warn_explicit(
            f"Call to deprecated function {func.__name__}.",
            category=UserWarning,
            filename=filename,
            lineno=lineno,
        )
        return func(*args, **kwargs)

    return new_func


def rdeprecated(txt=None, expire=None):
    """ This is a decorator which can be used to mark functions as deprecated.

    It will result in a warning being emitted when the function is used. After the expiration data the decorator
    will generate an Exception.

    Args:
        txt (str): reason for deprecation.
        expire (str): date of expiration.
    """
    import datetime

    from dateutil import parser
    if expire is not None:
        now = datetime.datetime.now()
        expiredate = parser.parse(expire)
        dt = expiredate - now
        expired = dt.total_seconds() < 0
    else:
        expired = None

    def deprecated_inner(func):
        """ This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            try:
                filename = inspect.getfile(func)
            except BaseException:
                filename = '?'
            try:
                lineno = inspect.getlineno(func)
            except BaseException:
                lineno = -1
            if txt is None:
                etxt = ''
            else:
                etxt = ' ' + txt

            if expire is not None:
                if expired:
                    raise Exception(f"Call to deprecated function {func.__name__}.{etxt}")
                else:
                    warnings.warn_explicit(
                        "Call to deprecated function {} (will expire on {}).{}".format(
                            func.__name__, expiredate, etxt),
                        category=UserWarning,
                        filename=filename,
                        lineno=lineno,
                    )
            else:
                warnings.warn_explicit(
                    f"Call to deprecated function {func.__name__}.{etxt}",
                    category=UserWarning,
                    filename=filename,
                    lineno=lineno,
                )
            return func(*args, **kwargs)

        return new_func

    return deprecated_inner


# %%


def update_dictionary(alldata: dict, **kwargs: dict):
    """ Update elements of a dictionary.

    Args:
        alldata (dict): dictionary to be updated.
        kwargs (dict): keyword arguments.

    """
    alldata.update(kwargs)


def stripDataset(dataset):
    """ Make sure a dataset can be pickled .

    Args:
        dataset (qcodes DataSet): TODO.

    Returns:
        dataset (qcodes DataSet): the dataset from the function argument.

    """
    dataset.sync()
    dataset.data_manager = None
    dataset.background_functions = {}
    try:
        dataset.formatter.close_file(dataset)
    except BaseException:
        pass

    if 'scanjob' in dataset.metadata:
        if 'minstrumenthandle' in dataset.metadata['scanjob']:
            dataset.metadata['scanjob']['minstrumenthandle'] = str(
                dataset.metadata['scanjob']['minstrumenthandle'])

    return dataset


# %%

def interruptable_sleep(seconds: float, step: float = .2) -> None:
    """ Alternative to `time.sleep` that can be interrupted

    Args:
        seconds: Number of seconds to sleep
        step: Step size in seconds
    """
    t0 = time.perf_counter()

    while (sleep_time := seconds-(time.perf_counter()-t0)) > 0:
        time.sleep(min(sleep_time, step))


def checkPickle(obj, verbose=0):
    """ Check whether an object can be pickled.

    Args:
        obj (object): object to be checked.
        verbose (int): verbosity (0 == silent).

    Returns:
        c (bool): True of the object can be pickled.

    """
    try:
        _ = pickle.dumps(obj)
    except Exception as ex:
        if verbose:
            print(ex)
        return False
    return True


def freezeclass(cls):
    """ Decorator to freeze a class."""
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and not hasattr(self, key):
            print("Class {} is frozen. Cannot set {} = {}"
                  .format(cls.__name__, key, value))
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True

        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


# %%


def resampleImage(im):
    """ Resample the image so it has the similar sample rates (samples/mV) in both axis.

    Args:
        im (DataArray): input image.
    Returns:
        imr (numpy array): resampled image.
        setpoints (list of 2 numpy arrays): setpoint arrays from resampled image.
    """
    setpoints = im.set_arrays
    mVrange = [abs(setpoints[0][-1] - setpoints[0][0]),
               abs(setpoints[1][0, -1] - setpoints[1][0, 0])]
    samprates = [im.shape[0] // mVrange[0], im.shape[1] // mVrange[1]]
    factor = int(max(samprates) // min(samprates))
    if factor >= 2:
        axis = int(samprates[0] - samprates[1] < 0)
        if axis == 0:
            facrem = im.shape[0] % factor
            if facrem > 0:
                im = im[:-facrem, :]
            facrem = facrem + 1
            im = im.reshape(im.shape[0] // factor, factor, im.shape[1]).mean(1)
            spy = np.linspace(setpoints[0][0], setpoints[
                0][-facrem], im.shape[0])
            spx = np.tile(np.expand_dims(np.linspace(
                setpoints[1][0, 0], setpoints[1][0, -1], im.shape[1]), 0), im.shape[0])
            setpointy = DataArray(name='Resampled_' + setpoints[0].array_id,
                                  array_id='Resampled_' + setpoints[0].array_id, label=setpoints[0].label,
                                  unit=setpoints[0].unit, preset_data=spy, is_setpoint=True)
            setpointx = DataArray(name='Resampled_' + setpoints[1].array_id,
                                  array_id='Resampled_' + setpoints[1].array_id, label=setpoints[1].label,
                                  unit=setpoints[1].unit, preset_data=spx, is_setpoint=True)
            setpoints = [setpointy, setpointx]
        else:
            facrem = im.shape[1] % factor
            if facrem > 0:
                im = im[:, :-facrem]
            facrem = facrem + 1
            im = im.reshape(im.shape[0], im.shape[1] //
                            factor, factor).mean(-1)
            spx = np.tile(np.expand_dims(np.linspace(setpoints[1][0, 0], setpoints[
                1][0, -facrem], im.shape[1]), 0), [im.shape[0], 1])
            idx = setpoints[1].array_id
            if idx is None:
                idx = 'x'
            idy = setpoints[1].array_id
            if idy is None:
                idy = 'y'
            setpointx = DataArray(name='Resampled_' + idx, array_id='Resampled_' + idy, label=setpoints[1].label,
                                  unit=setpoints[1].unit, preset_data=spx, is_setpoint=True)
            setpoints = [setpoints[0], setpointx]

    return im, setpoints


def diffImage(im, dy, size=None):
    """ Simple differentiation of an image.

    Args:
        im (numpy array): input image.
        dy (integer or string): method of differentiation. For an integer it is the axis of differentiation.
            Allowed strings are 'x', 'y', 'xy'.
        size (str): describes the size e.g. 'same'.

    """
    if dy == 0 or dy == 'x':
        im = np.diff(im, n=1, axis=1)
        if size == 'same':
            im = np.hstack((im, im[:, -1:]))
    elif dy == 1 or dy == 'y':
        im = np.diff(im, n=1, axis=0)
        if size == 'same':
            im = np.vstack((im, im[-1:, :]))
    elif dy == -1:
        im = -np.diff(im, n=1, axis=0)
        if size == 'same':
            im = np.vstack((im, im[-1:, :]))
    elif dy == 2 or dy == 'xy':
        imx = np.diff(im, n=1, axis=1)
        imy = np.diff(im, n=1, axis=0)
        im = imx[0:-1, :] + imy[:, 0:-1]
    elif dy is None:
        pass
    else:
        raise Exception('differentiation method %s not supported' % dy)
    return im


def diffImageSmooth(im, dy='x', sigma=2):
    """ Simple differentiation of an image.

    Args:
        im (array): input image.
        dy (string or integer): direction of differentiation. can be 'x' (0) or 'y' (1) or 'xy' (2) or 'g' (3).
        sigma (float): parameter for gaussian filter kernel.

    """
    if sigma is None:
        imx = diffImage(im, dy)
        return imx

    if dy is None:
        imx = im.copy()
    elif dy == 0 or dy == 'x':
        if len(im.shape) == 1:
            raise Exception(f'invalid parameter dy={dy} for 1D image')
        else:
            imx = ndimage.gaussian_filter1d(im, axis=1, sigma=sigma, order=1, mode='nearest')
    elif dy == 1 or dy == 'y':
        imx = ndimage.gaussian_filter1d(im, axis=0, sigma=sigma, order=1, mode='nearest')
    elif dy == -1:
        imx = -ndimage.gaussian_filter1d(im, axis=0,
                                         sigma=sigma, order=1, mode='nearest')
    elif dy == 2 or dy == 3 or dy == 'xy' or dy == 'xmy' or dy == 'xmy2' or dy == 'g' or dy == 'x2my2' or dy == 'x2y2':
        if len(np.array(im).shape) != 2:
            raise Exception(f'differentiation mode {dy} cannot be combined with input shape {np.array(im).shape}')
        imx0 = ndimage.gaussian_filter1d(
            im, axis=1, sigma=sigma, order=1, mode='nearest')
        imx1 = ndimage.gaussian_filter1d(
            im, axis=0, sigma=sigma, order=1, mode='nearest')
        if dy == 2 or dy == 'xy':
            imx = imx0 + imx1
        if dy == 'xmy':
            imx = imx0 - imx1
        if dy == 3 or dy == 'g':
            imx = np.sqrt(imx0 ** 2 + imx1 ** 2)
        if dy == 'xmy2':
            warnings.warn('please do not use this option')
            imx = np.sqrt(imx0 ** 2 + imx1 ** 2)
        if dy == 'x2y2':
            imx = imx0 ** 2 + imx1 ** 2
        if dy == 'x2my2':
            imx = imx0 ** 2 - imx1 ** 2
    else:
        raise Exception('differentiation method %s not supported' % dy)
    return imx


def scanTime(dd):
    """ Return date a scan was performed."""
    w = dd.metadata.get('scantime', None)
    if isinstance(w, str):
        w = dateutil.parser.parse(w)
    return w


# %%

def showImage(im, extent=None, fig=None, title=None):
    """ Show image in figure window.

    Args:
        im (array): TODO.
        extent (list): matplotlib style image extent.
        fig (None or int): figure window to show image.
        title (None or str): figure title.

    """
    import matplotlib.pyplot as plt
    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plt.imshow(im, extent=extent, interpolation='nearest')
        if extent is not None:
            if extent[0] > extent[1]:
                plt.gca().invert_xaxis()
        if title is not None:
            plt.title(title)


# %% Tools from pgeometry


def cfigure(*args, **kwargs):
    """ Create Matplotlib figure with copy to clipboard functionality.

    By pressing the 'c' key figure is copied to the clipboard.

    """
    if 'facecolor' in kwargs:
        fig = plt.figure(*args, **kwargs)
    else:
        fig = plt.figure(*args, facecolor='w', **kwargs)

    def ff(xx, figx=fig):
        return mpl2clipboard(fig=figx)

    fig.canvas.mpl_connect('key_press_event', ff)  # mpl2clipboard)
    return fig


def static_var(varname, value):
    """ Helper function to create a static variable."""

    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


try:
    import qtpy.QtGui as QtGui
    import qtpy.QtWidgets as QtWidgets

    def monitorSizes(verbose=0):
        """ Return monitor sizes."""
        _qd = QtWidgets.QDesktopWidget()
        if sys.platform == 'win32' and _qd is None:
            import ctypes
            user32 = ctypes.windll.user32
            wa = [
                [0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]]
        else:
            _applocalqt = QtWidgets.QApplication.instance()

            if _applocalqt is None:
                _applocalqt = QtWidgets.QApplication([])
                _qd = QtWidgets.QDesktopWidget()
            else:
                _qd = QtWidgets.QDesktopWidget()

            nmon = _qd.screenCount()
            wa = [_qd.screenGeometry(ii) for ii in range(nmon)]
            wa = [[w.x(), w.y(), w.width(), w.height()] for w in wa]

            if verbose:
                for ii, w in enumerate(wa):
                    print('monitor %d: %s' % (ii, str(w)))
        return wa
except BaseException:
    def monitorSizes(verbose=0):
        """ Dummy function for monitor sizes."""
        return [[0, 0, 1600, 1200]]

    pass


# %% Helper tools

def in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False


def pythonVersion():
    try:
        import IPython
        ipversion = '.'.join('%s' % x for x in IPython.version_info[:-1])
    except BaseException:
        ipversion = 'None'

    pversion = '.'.join('%s' % x for x in sys.version_info[0:3])
    print('python %s, ipython %s, notebook %s' %
          (pversion, ipversion, in_ipynb()))


# %%

def _covert_integer_to_rgb_color(integer_value):
    if integer_value < 0 or integer_value > 256 ** 3:
        raise ValueError('Integer value cannot be converted to RGB!')

    red = integer_value & 0xFF
    green = (integer_value >> 8) & 0xFF
    blue = (integer_value >> 16) & 0xFF
    return red, green, blue


def _convert_rgb_color_to_integer(rgb_color):
    if not isinstance(rgb_color, tuple) or not all(isinstance(i, int) for i in rgb_color):
        raise ValueError('Color should be an RGB integer tuple.')

    if len(rgb_color) != 3:
        raise ValueError('Color should be an RGB integer tuple with three items.')

    if any(i < 0 or i > 255 for i in rgb_color):
        raise ValueError('Color should be an RGB tuple in the range 0 to 255.')

    red = rgb_color[0]
    green = rgb_color[1] << 8
    blue = rgb_color[2] << 16
    return int(red + green + blue)


def set_ppt_slide_background(slide, color, verbose=0):
    """ Sets the background color of PPT slide.

    Args:
        slide (object): PowerPoint COM object for slide.
        color (tuple): tuple with RGB color specification.
        verbose (int): verbosity (0 == silent).
    """
    fore_color = slide.Background.Fill.ForeColor
    ppt_color = _convert_rgb_color_to_integer(color)
    if verbose > 1:
        print('Setting PPT slide background color:')
        print(f' - Current color: {_covert_integer_to_rgb_color(fore_color.RGB)}')
        print(f' - Setting to {color} -> {ppt_color}')

    slide.FollowMasterBackground = 0
    fore_color.RGB = ppt_color


# %%


def _ppt_determine_image_position(ppt, figsize, fname, verbose=1):
    top = 120

    if figsize is not None:
        left = (ppt.PageSetup.SlideWidth - figsize[0]) / 2
        width = figsize[0]
        height = figsize[1]
    else:
        slidewh = [ppt.PageSetup.SlideWidth, ppt.PageSetup.SlideHeight]
        width = 16 * ((slidewh[0] * .75) // 16)
        height = 16 * (((slidewh[1] - 120) * .9) // 16)
        height = min(height, 350)
        left = (slidewh[0] - width) / 2

        try:
            import cv2
            imwh = cv2.imread(fname).shape[1], cv2.imread(fname).shape[0]
        except BaseException:
            imwh = None
        if imwh is not None:
            imratio = imwh[0] / imwh[1]
            slideratio = slidewh[0] / slidewh[1]
            if verbose >= 1:
                print(' image aspect ratio %.2f, slide aspect ratio %.2f' % (imratio, slideratio))
            if slideratio > imratio:
                # wide slide, so make the image width smaller
                if verbose >= 2:
                    print('adjust width %d->%d' % (width, height * imratio))
                width = height * imratio
            else:
                # wide image, so make the image height smaller
                if verbose >= 2:
                    print('adjust height %d->%d' % (height, width / imratio))
                height = int(width / imratio)

        if verbose:
            print('slide width height: %s' % (slidewh,))
            print('image width height: %d, %d' % (width, height))
    return left, top, width, height


def create_figure_ppt_callback(fig: Optional[Union[int, matplotlib.figure.Figure]] = None, title: Optional[str] = None,
                               notes: Optional[Union[str, DataSet]] = None,
                               position: Tuple[float, float, float, float] = (0.9, 0.925, 0.075, 0.05)) -> None:
    """ Create a button on a matplotlib figure to copy data to PowerPoint slide.

    The figure is copied to PowerPoint using @ref addPPTslide.

    Args:
        fig: Handle to matplotlib window. If None, then use the current figure
        title: title for the slide.
        notes: notes to add to the slide.
        position: position specified as fraction left, right, width, height.

    Example:
        >>> plt.figure(10)
        >>> plt.plot(np.arange(100), np.random.rand(100), 'o', label='input data')
        >>> create_figure_ppt_callback(fig=10, title='test')
        >>> plt.show()
    """
    if fig is None:
        fig = plt.gcf()
    if isinstance(fig, int):
        fig = plt.figure(fig)

    ax = fig.gca()
    ppt_axis = fig.add_axes(position, label=f'figure_ppt_callback_axis {uuid.uuid1()}')
    ppt_button = Button(ppt_axis, 'ppt')
    ppt_axis._button = ppt_button
    ppt_axis.set_alpha(.5)
    plt.sca(ax)

    def figure_ppt_callback(event):
        print(f'creating PowerPoint slide for figure {fig}')
        ppt_axis.set_visible(False)
        addPPTslide(fig=fig, title=title, notes=notes)
        ppt_axis.set_visible(True)

    ppt_button.on_clicked(figure_ppt_callback)


try:
    import win32com
    import win32com.client

    def generate_powerpoint_notes(notes: Optional[Union[str, qcodes.Station, DataSet]], extranotes: Optional[str], maximum_notes_size: int):
        """ Generate text to be added as notes to a PPT slide """
        if notes is None:
            warnings.warn(
                'please set notes for the powerpoint slide. e.g. use the station or reshape_metadata')

        if isinstance(notes, qcodes.Station):
            station = notes
            gates = getattr(station, 'gates', None)
            notes = reshape_metadata(station, printformat='s', add_scanjob=True)
            if extranotes is None:
                pass
            else:
                notes = '\n' + extranotes + '\n' + notes  # type: ignore
            if gates is not None:
                notes = 'gates: ' + str(gates.allvalues()) + '\n\n' + notes  # type: ignore
        elif isinstance(notes, DataSet):
            notes = reshape_metadata(notes, printformat='s', add_gates=True)

        if not isinstance(notes, str):
            warnings.warn(f'type of notes argument is {type(notes)}, converting to string')
            notes = str(notes)

        if notes == '':
            notes = ' '
        if len(notes) > maximum_notes_size:
            warnings.warn(f'notes for powerpoint are {len(notes)} characters, reducing to {maximum_notes_size}')
            notes = notes[:maximum_notes_size]
        return notes

    def addPPTslide(title: Optional[str] = None, fig: Optional[Union[int, np.ndarray, plt.Figure, Any]] = None, txt: Optional[str] = None,
                    notes: Optional[Union[str, qcodes.Station]] = None, figsize: Optional[Tuple[int, int]] = None,
                    subtitle: Optional[str] = None, maintext: Optional[str] = None, show: bool = False, verbose: int = 1,
                    activate_slide: bool = True, ppLayout: Optional[int] = None, extranotes: Optional[str] = None, background_color: Optional[Tuple] = None,
                    maximum_notes_size: int = 10000) -> Tuple[Any, Any]:
        """ Add slide to current active Powerpoint presentation.

        Arguments:
            title (str): title added to slide.
            fig (matplotlib.figure.Figure or qcodes_loop.plots.pyqtgraph.QtPlot or integer):
                figure added to slide.
            txt (str) : Deprecated, use subtitle instead.
            notes: notes added to slide.
            figsize: size (width,height) of figurebox to add to powerpoint.
            subtitle : text added to slide as subtitle.
            maintext: text in textbox added to slide.
            show : shows the powerpoint application.
            verbose: print additional information.
            activate_slide: activate the current slide.
            ppLayout: layout of PP-slide (TitleOnly = 11, Text = 2).
            extranotes: notes for slide.
            background_color: background color for the slide.
            maximum_notes_size: Maximum size of the notes in number of characters

        Returns:
            Tuple of the PowerPoint presentation and PowerPoint slide.

        The interface to Powerpoint used is described here:
            https://msdn.microsoft.com/en-us/library/office/ff743968.aspx

        Example:
            >>> title = 'An example title'
            >>> fig = plt.figure(10)
            >>> txt = 'Some comments on the figure'
            >>> notes = 'some additional information'
            >>> addPPTslide(title, fig, subtitle=txt, notes=notes)
        """
        Application = win32com.client.Dispatch("PowerPoint.Application")

        if verbose >= 2:
            print('Number of open PPTs: %d.' % Application.presentations.Count)

        try:
            ppt = Application.ActivePresentation
        except Exception:
            print('Could not open active Powerpoint presentation, opening blank presentation.')
            try:
                ppt = Application.Presentations.Add()
            except Exception as ex:
                warnings.warn(f'Could not make connection to Powerpoint presentation. {ex}')
                return None, None

        if show:
            Application.Visible = True  # shows what's happening, not required, but helpful for now

        if verbose >= 2:
            print('addPPTslide: presentation name: %s' % ppt.Name)

        ppLayoutTitleOnly = 11
        ppLayoutText = 2

        if txt is not None:
            if subtitle is None:
                warnings.warn('please do not use the txt field any more')
                subtitle = txt
            else:
                raise ValueError('please do not use the txt field any more')

            txt = None

        if fig is None:
            # no figure, text box over entire page
            if ppLayout is None:
                ppLayout = ppLayoutText
        else:
            # we have a figure, assume textbox is for dataset name only
            ppLayout = ppLayoutTitleOnly

        max_slides_count_warning = 750
        max_slides_count = 950
        maximum_notes_size = int(maximum_notes_size)

        if ppt.Slides.Count > max_slides_count_warning:
            warning_message = "Your presentation has more than {} slides! \
                Please start a new measurement logbook.".format(max_slides_count_warning)
            warnings.warn(warning_message)
        if ppt.Slides.Count > max_slides_count:
            error_message = "Your presentation has more than {} slides! \
                Please start a new measurement logbook.".format(max_slides_count)
            raise MemoryError(error_message)

        if verbose:
            print('addPPTslide: presentation name: %s, adding slide %d' %
                  (ppt.Name, ppt.Slides.count + 1))

        slide = ppt.Slides.Add(ppt.Slides.Count + 1, ppLayout)

        if background_color is not None:
            set_ppt_slide_background(slide, background_color, verbose=verbose)

        if fig is None:
            mainbox = slide.shapes.Item(2)
            if maintext is None:
                raise TypeError('maintext argument is None')
            mainbox.TextFrame.TextRange.Text = maintext
        else:
            mainbox = None
            if maintext is not None:
                warnings.warn('maintext not implemented when figure is set')

        if title is not None:
            slide.shapes.title.textframe.textrange.text = title
        else:
            slide.shapes.title.textframe.textrange.text = 'QCoDeS measurement'

        from qtt.measurements.videomode import VideoMode  # import here, to prevent default imports of gui code

        def add_figure_to_slide(fig, slide, figsize, verbose):
            """ Add figure to PPT slide """
            fname = tempfile.mktemp(prefix='qcodesimageitem-', suffix='.png')
            if isinstance(fig, matplotlib.figure.Figure):
                fig.savefig(fname)
            elif isinstance(fig, int):
                fig = plt.figure(fig)
                fig.savefig(fname)
            elif isinstance(fig, VideoMode) or \
                    fig.__class__.__name__ == 'VideoMode':
                if isinstance(fig.lp, list):
                    # do NOT change this into a list comprehension
                    ff = []
                    for jj in range(len(fig.lp)):
                        ff.append(fig.lp[jj].plotwin.grab())

                    sz = ff[0].size()
                    sz = QtCore.QSize(sz.width() * len(ff), sz.height())
                    figtemp = QtGui.QPixmap(sz)
                    p = QtGui.QPainter(figtemp)
                    offset = 0
                    for ii in range(len(ff)):
                        p.drawPixmap(offset, 0, ff[ii])
                        offset += ff[ii].size().width()
                    figtemp.save(fname)
                    p.end()
                else:
                    figtemp = fig.lp.plotwin.grab()
                    figtemp.save(fname)
            elif isinstance(fig, QtWidgets.QWidget):
                # generic method
                figtemp = fig.plotwin.grab()
                figtemp.save(fname)
            elif isinstance(fig, QtWidgets.QWidget):
                try:
                    figtemp = QtGui.QPixmap.grabWidget(fig)
                except BaseException:
                    figtemp = fig.grab()
                figtemp.save(fname)
            elif isinstance(fig, qcodes_loop.plots.pyqtgraph.QtPlot):
                fig.save(fname)
            elif isinstance(fig, np.ndarray):
                imageio.imwrite(fname, fig)
            elif isinstance(fig, str) and fig.endswith('.png'):
                fname = fig
            else:
                if verbose:
                    raise TypeError('figure is of an unknown type %s' % (type(fig),))
            slide_margin_left, slide_margin_top, width, height = _ppt_determine_image_position(
                ppt, figsize, fname, verbose=verbose >= 2)

            slide.Shapes.AddPicture(FileName=fname, LinkToFile=False,
                                    SaveWithDocument=True, Left=slide_margin_left, Top=slide_margin_top, Width=width, Height=height)

        if fig is not None:
            add_figure_to_slide(fig, slide, figsize, verbose)

        if subtitle is not None:
            # add subtitle
            subtitlebox = slide.Shapes.AddTextbox(
                1, Left=100, Top=80, Width=500, Height=300)
            subtitlebox.Name = 'subtitle box'
            subtitlebox.TextFrame.TextRange.Text = subtitle

        notes_text = generate_powerpoint_notes(notes, extranotes, maximum_notes_size)
        slide.notespage.shapes.placeholders[
            2].textframe.textrange.insertafter(notes_text)

        if activate_slide:
            idx = int(slide.SlideIndex)
            if verbose >= 2:
                print('addPPTslide: goto slide %d' % idx)
            Application.ActiveWindow.View.GotoSlide(idx)
        return ppt, slide

    def addPPT_dataset(dataset, title=None, notes=None,
                       show=False, verbose=1, paramname='measured',
                       printformat='fancy', customfig=None, extranotes=None, **kwargs):
        """ Add slide based on dataset to current active Powerpoint presentation.

        Args:
            dataset (DataSet): data and metadata from DataSet added to slide.
            customfig (QtPlot): custom QtPlot object to be added to
                                slide (for dataviewer).
            notes (string): notes added to slide.
            show (bool): shows the powerpoint application.
            verbose (int): print additional information.
            paramname (None or str): passed to dataset.default_parameter_array.
            printformat (string): 'fancy' for nice formatting or 'dict'
                                  for easy copy to python.
        Returns:
            ppt: PowerPoint presentation.
            slide: PowerPoint slide.

        Example:
            >>> notes = 'some additional information'
            >>> addPPT_dataset(dataset, notes)
        """
        if len(dataset.arrays) < 2:
            raise IndexError('The dataset contains less than two data arrays')

        if customfig is None:

            if isinstance(paramname, str):
                if title is None:
                    parameter_name = dataset.default_parameter_name(paramname=paramname)
                    title = 'Parameter: %s' % parameter_name
                temp_fig = QtPlot(dataset.default_parameter_array(
                    paramname=paramname), show_window=False)
            else:
                if title is None:
                    title = 'Parameter: %s' % (str(paramname),)
                for idx, parameter_name in enumerate(paramname):
                    if idx == 0:
                        temp_fig = QtPlot(dataset.default_parameter_array(
                            paramname=parameter_name), show_window=False)
                    else:
                        temp_fig.add(dataset.default_parameter_array(
                            paramname=parameter_name))

        else:
            temp_fig = customfig

        text = 'Dataset location: %s' % dataset.location
        if notes is None:
            try:
                metastring = reshape_metadata(dataset,
                                              printformat=printformat)
            except Exception as ex:
                metastring = 'Could not read metadata: %s' % str(ex)
            notes = 'Dataset %s metadata:\n\n%s' % (dataset.location,
                                                    metastring)
            scanjob = dataset.metadata.get('scanjob', None)
            if scanjob is not None:
                s = pprint.pformat(scanjob)
                notes = 'scanjob: ' + str(s) + '\n\n' + notes

            gatevalues = dataset.metadata.get('allgatevalues', None)
            if gatevalues is not None:
                notes = 'gates: ' + str(gatevalues) + '\n\n' + notes

        ppt, slide = addPPTslide(title=title, fig=temp_fig, subtitle=text,
                                 notes=notes, show=show, verbose=verbose,
                                 extranotes=extranotes,
                                 **kwargs)
        return ppt, slide

except ImportError:
    def addPPTslide(title: Optional[str] = None, fig: Optional[Union[int, np.ndarray, plt.Figure, Any]] = None, txt: Optional[str] = None,
                    notes: Optional[Union[str, qcodes.Station]] = None, figsize: Optional[Tuple[int, int]] = None,
                    subtitle: Optional[str] = None, maintext: Optional[str] = None, show: bool = False, verbose: int = 1,
                    activate_slide: bool = True, ppLayout: Optional[int] = None, extranotes: Optional[str] = None, background_color: Optional[Tuple] = None,
                    maximum_notes_size: int = 10000) -> Tuple[Any, Any]:
        """ Add slide to current active Powerpoint presentation.

        Dummy implementation.
        """
        warnings.warn(
            'addPPTslide is not available on your system. Install win32com from https://pypi.org/project/pypiwin32/.')
        return None, None

    def addPPT_dataset(dataset, title=None, notes=None,
                       show=False, verbose=1, paramname='measured',
                       printformat='fancy', customfig=None, extranotes=None, **kwargs):
        """ Add slide based on dataset to current active Powerpoint presentation.

        Dummy implementation.
        """
        warnings.warn(
            'addPPT_dataset is not available on your system. Install win32com from https://pypi.org/project/pypiwin32/.')


# %%


def reshape_metadata(dataset, printformat='dict', add_scanjob=True, add_gates=True, add_analysis_results=True, verbose=0) -> str:
    """ Reshape the metadata of a DataSet.

    Args:
        dataset (DataSet or qcodes.Station): a dataset of which the metadata will be reshaped.
        printformat (str): can be 'dict' or 'txt','fancy' (text format).
        add_scanjob (bool): If True, then add the scanjob at the beginning of the notes.
        add_analysis_results (bool): If True, then add the analysis_results at the beginning of the notes.
        add_gates (bool): If True, then add the scanjob at the beginning of the notes.
        verbose (int): verbosity (0 == silent).

    Returns:
        The reshaped metadata.

    """
    if isinstance(dataset, qcodes.Station):
        station = dataset
        all_md = station.snapshot(update=False)['instruments']
        header = ''
    else:
        tmp = dataset.metadata.get('station', None)
        if tmp is None:
            all_md = {}
        else:
            all_md = tmp['instruments']

        header = 'dataset: %s' % dataset.location

        if hasattr(dataset.io, 'base_location'):
            header += ' (base %s)' % dataset.io.base_location

    if add_gates:
        gate_values = dataset.metadata.get('allgatevalues', None)

        if gate_values is not None:
            gate_values = {key: np.around(value, 3) for key, value in gate_values.items()}
            header += '\ngates: ' + str(gate_values) + '\n'

    scanjob = dataset.metadata.get('scanjob', None)
    if scanjob is not None and add_scanjob:
        s = pprint.pformat(scanjob)
        header += '\n\nscanjob: ' + str(s) + '\n'

    analysis_results = dataset.metadata.get('analysis_results', None)
    if analysis_results is not None and add_analysis_results:
        s = pprint.pformat(analysis_results)
        header += '\n\analysis_results: ' + str(s) + '\n'

    metadata: Dict[Any, Dict] = {}
    # make sure the gates instrument is in front
    all_md_keys = sorted(sorted(all_md), key=lambda x: x == 'gates', reverse=True)
    for x in all_md_keys:
        metadata[x] = {}
        if 'IDN' in all_md[x]['parameters']:
            metadata[x]['IDN'] = dict({'name': 'IDN', 'value': all_md[
                x]['parameters']['IDN']['value']})
            metadata[x]['IDN']['unit'] = ''
        for y in sorted(all_md[x]['parameters'].keys()):
            try:
                if y != 'IDN':
                    metadata[x][y] = {}
                    param_md = all_md[x]['parameters'][y]
                    metadata[x][y]['name'] = y
                    if isinstance(param_md['value'], (float, np.float64)):
                        metadata[x][y]['value'] = float(
                            format(param_md['value'], '.3f'))
                    else:
                        metadata[x][y]['value'] = str(param_md['value'])
                    metadata[x][y]['unit'] = param_md.get('unit', None)
                    metadata[x][y]['label'] = param_md.get('label', None)
            except KeyError as ex:
                if verbose:
                    print('failed on parameter %s / %s: %s' % (x, y, str(ex)))

    if printformat == 'dict':
        ss = str(metadata).replace('(', '').replace(
            ')', '').replace('OrderedDict', '')
    else:  # 'txt' or 'fancy'
        ss = ''
        for k in metadata:
            if verbose:
                print('--- %s' % k)
            element = metadata[k]
            ss += '\n## %s:\n' % k
            for parameter in element:
                pp = element[parameter]
                if verbose:
                    print('  --- %s: %s' % (parameter, pp.get('value', '??')))
                ss += '%s: %s (%s)' % (pp['name'], pp.get('value', '?'), pp.get('unit', ''))
                ss += '\n'

    if header is not None:
        ss = header + '\n\n' + ss
    return ss


# %%

def setupMeasurementWindows(*args, **kwargs):
    raise Exception('use qtt.gui.live_plotting.setupMeasurementWindows instead')


def updatePlotTitle(qplot, basetxt='Live plot'):
    """ Update the plot title of a QtPlot window."""
    txt = basetxt + ' (%s)' % time.asctime()
    qplot.win.setWindowTitle(txt)


# %%


def flatten(lst):
    """ Flatten a list.

    Args:
        lst (list): list to be flattened.

    Returns:
        list: flattened list.

    Example:
        >>> flatten([ [1,2], [3,4], [10] ])
        [1, 2, 3, 4, 10]
    """
    return list(chain(*lst))


# %%


def cutoffFilter(x, thr, omega):
    """ Smooth cutoff filter.

    Filter definition from: http://paulbourke.net/miscellaneous/imagefilter/

    Example
    -------
    >>> plt.clf()
    >>> x=np.arange(0, 4, .01)
    >>> _=plt.plot(x, cutoffFilter(x, 2, .25), '-r')

    """
    y = .5 * (1 - np.sin(np.pi * (x - thr) / (2 * omega)))
    y[x < thr - omega] = 1
    y[x > thr + omega] = 0
    return y


# %%


def smoothFourierFilter(fs=100, thr=6, omega=2, fig=None):
    """ Create smooth ND filter for Fourier high or low-pass filtering.

    Example
    -------
    >>> F=smoothFourierFilter([24,24], thr=6, omega=2)
    >>> _=plt.figure(10); plt.clf(); _=plt.imshow(F, interpolation='nearest')
    """
    rr = np.meshgrid(*(range(f) for f in fs))

    x = np.dstack(rr)
    x = x - (np.array(fs) / 2 - .5)
    x = np.linalg.norm(x, axis=2)

    F = cutoffFilter(x, thr, omega)

    if fig is not None:
        plt.figure(10)
        plt.clf()
        plt.imshow(F, interpolation='nearest')

    return F


F = smoothFourierFilter([36, 36])


# %%

def fourierHighPass(imx, nc=40, omega=4, fs=1024, fig=None):
    """ Implement simple high pass filter using the Fourier transform."""
    f = np.fft.fft2(imx, s=[fs, fs])  # do the fourier transform

    fx = np.fft.fftshift(f)

    if fig:
        plt.figure(fig)
        plt.clf()
        plt.imshow(np.log(np.abs(f) + 1), interpolation='nearest')
        plt.title('Fourier spectrum (real part)')
        plt.figure(fig + 1)
        plt.clf()
        plt.imshow(np.log(np.abs(fx) + 1), interpolation='nearest')
        plt.title('Fourier spectrum (real part)')

    if nc > 0 and omega == 0:
        f[0:nc, 0:nc] = 0
        f[-nc:, -nc:] = 0
        f[-nc:, 0:nc] = 0
        f[0:nc, -nc:] = 0
        img_back = np.fft.ifft2(f)  # inverse fourier transform

    else:
        # smooth filtering

        F = 1 - smoothFourierFilter(fx.shape, thr=nc, omega=omega)
        fx = F * fx
        ff = np.fft.ifftshift(fx)  # inverse shift
        img_back = np.fft.ifft2(ff)  # inverse fourier transform

    imf = img_back.real
    imf = imf[0:imx.shape[0], 0:imx.shape[1]]

    return imf


# %%


def slopeClick(drawmode='r--', **kwargs):
    """ Calculate slope for line piece of two points clicked by user. Works
    with matplotlib but not with pyqtgraph. Uses the currently active figure.

    Args:
        drawmode (string): plotting style.

    Returns:
        coords (2 x 2 array): coordinates of the two clicked points.
        signedslope (float): slope of linepiece connecting the two points.

    """
    ax = plt.gca()
    ax.set_autoscale_on(False)
    coords = qtt.pgeometry.ginput(2, drawmode, **kwargs)
    plt.pause(1e-6)
    signedslope = (coords[1, 0] - coords[1, 1]) / (coords[0, 0] - coords[0, 1])

    return coords, signedslope


def clickGatevals(plot, drawmode='ro'):
    """ Get gate values for all gates at clicked point in a heatmap.

    Args:
        plot (qcodes MatPlot object): plot of measurement data.
        drawmode (string): plotting style.

    Returns:
        gatevals (dict): values of the gates at clicked point.

    """
    # TODO: implement for virtual gates
    if not isinstance(plot, qcodes_loop.plots.qcmatplotlib.MatPlot):
        raise Exception(
            'The plot object is not based on the MatPlot class from qcodes.')

    ax = plt.gca()
    ax.set_autoscale_on(False)
    coords = qtt.pgeometry.ginput(drawmode=drawmode)
    data_array = plot.traces[0]['config']['z']
    dataset = data_array.data_set

    gatevals = copy.deepcopy(dataset.metadata['allgatevalues'])
    if len(data_array.set_arrays) != 2:
        raise Exception('The DataArray does not have exactly two set_arrays.')

    for arr in data_array.set_arrays:
        if len(arr.ndarray.shape) == 1:
            gatevals[arr.name] = coords[1, 0]
        else:
            gatevals[arr.name] = coords[0, 0]

    return gatevals


# %%


def connect_slot(target):
    """ Create a slot by dropping signal arguments."""

    def signal_drop_arguments(*args, **kwargs):
        target()

    return signal_drop_arguments
