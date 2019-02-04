import sys
import os
import numpy as np
import pprint
import matplotlib
import logging
import qcodes
import warnings
import functools
import pickle
import inspect
import tempfile
from itertools import chain
import scipy.ndimage as ndimage
from functools import wraps
import datetime
import subprocess
import glob
import time
from colorama import Fore
import importlib
import platform
import matplotlib.pyplot as plt

try:
    from dulwich.repo import Repo, NotGitRepository
    from dulwich import porcelain
except ModuleNotFoundError:
    warnings.warn('please install dulwich: pip install dulwich --global-option="--pure"')
    NotGitRepository = Exception


# explicit import
from qcodes.plots.qcmatplotlib import MatPlot
try:
    from qcodes.plots.pyqtgraph import QtPlot
except:
    pass
from qcodes import DataArray

from qtt import pgeometry
from qtt.pgeometry import mpl2clipboard

# do NOT load any other qtt submodules here

try:
    import qtpy.QtGui as QtGui
    import qtpy.QtCore as QtCore
    import qtpy.QtWidgets as QtWidgets
except:
    pass


# %%


def get_module_versions(modules, verbose=0):
    """ Returns the module version of the given pip packages.

        Args:
            modules ([str]): a list with pip packages, e.g. ['numpy', 'scipy']

        Returns:
            r (dict): dictionary with package names and version number for each given module.
    """
    module_versions = dict()
    for module in modules:
        try:
            package = importlib.import_module(module)
            version = getattr(package, '__version__', 'none')
            module_versions[module] = version
        except ModuleNotFoundError:
            module_versions[module] = 'none'
        if verbose:
            print('module {0} {1}'.format(package, version))
    return module_versions


def get_git_versions(repos, get_dirty_status=False, verbose=0):
    """ Returns the repository head guid and dirty status and package version number if installed via pip.
        The version is only returned if the repo is installed as pip package without edit mode.
        NOTE: currently the dirty status is not working correctly due to a bug in dulwich...

        Args:
            repos ([str]): a list with repositories, e.g. ['qtt', 'qcodes']
            get_dirty_status (bool): selects whether to use the dulwich package and collect the local code
                                changes for the repositories.

        Retuns:
            r (dict): dictionary with repo names, head guid and (optionally) dirty status for each given repository.
    """
    heads = dict()
    dirty_stats = dict()
    for repo in repos:
        try:
            package = importlib.import_module(repo)
            init_location = os.path.split(package.__file__)[0]
            repo_location = os.path.join(init_location, '..')
            repository = Repo(repo_location)
            heads[repo] = repository.head().decode('ascii')
            if get_dirty_status:
                status = porcelain.status(repository)
                is_dirty = len(status.unstaged) == 0 or any(len(item) != 0 for item in status.staged.values())
                dirty_stats[repo] = is_dirty
        except (AttributeError, ModuleNotFoundError, NotGitRepository):
            heads[repo] = 'none'
            if get_dirty_status:
                dirty_stats[repo] = 'none'
        if verbose:
            print('{0}: {1}'.format(repo, heads[repo]))
    return (heads, dirty_stats)


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
        verbose (int): output level

    Returns:
        status (dict): python, modules and git repos status.
    """
    _default_git_versions = ['qcodes', 'qtt', 'projects', 'pycqed']
    _default_module_versions = ['numpy', 'scipy', 'qupulse', 'h5py', 'skimage']
    if not repository_names:
        repository_names = _default_git_versions
    if not package_names:
        package_names = _default_module_versions
    result = dict()
    repository_stats, dirty_stats = get_git_versions(repository_names, get_dirty_status, verbose)
    result['python'] = get_python_version(verbose)
    result['git'] = repository_stats
    result['version'] = get_module_versions(package_names, verbose)
    result['timestamp'] = datetime.datetime.now().isoformat()  # ISO 8601
    result['system'] = {'node': platform.node()}
    if get_dirty_status:
        result['dirty'] = dirty_stats

    return result


def test_python_code_modules_and_versions():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="qupulse")
        _ = get_python_version()
        _ = get_module_versions(['numpy'])
        _ = get_git_versions(['qtt'])
        c = code_version()
        assert('python' in c)
        assert('timestamp' in c)
        assert('system' in c)

# %% Jupyter kernel tools


def get_jupyter_kernel(verbose=2):
    """ Return the most recently created jupyter kernel

    Args:
        verbose (int): verbosity level

    """
    cmd = r'jupyter --paths'

    print(Fore.BLUE + 'get_jupyuter_kernel: running %s' % cmd + Fore.RESET)

    rr = subprocess.check_output(cmd, shell=True)
    rr = rr.decode('ASCII')
    r = 0
    rundir = None
    for ii, l in enumerate(rr.split('\n')):
        if r:
            rundir = l.strip()
            break
        if verbose >= 3:
            print(l)
        if l.startswith('runtime:'):
            r = 1

    if rundir is not None:
        print(Fore.BLUE + '  rundir: %s' % rundir + Fore.RESET)

        # remove anything from the list that is not a file (directories,
        # symlinks)
        files = list(filter(os.path.isfile, glob.glob(
            rundir + os.sep + "kernel*json")))
        files.sort(key=lambda x: os.path.getctime(x))
        files = files[::-1]

        if verbose >= 2:
            print('all kernels (< 2 days old): ')
            for k in files:
                dt = time.time() - os.path.getctime(k)
                if dt < 3600 * 24 * 2:
                    print(' %.1f [s]:  ' % (float(dt), ) + k)
        if len(files) > 0:
            kernel = files[0]
            print(Fore.BLUE + '  found kernel: %s' % kernel + Fore.RESET)
            kernelbase = os.path.split(kernel)[1]
            print(Fore.BLUE + 'connect with: ' + Fore.GREEN +
                  ' jupyter console --existing %s' % kernelbase + Fore.RESET)
            return kernelbase
    return None

#%% Debugging


def deprecated(func):
    """ This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used. """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        try:
            filename = inspect.getfile(func)
        except:
            filename = '?'
        try:
            lineno = inspect.getlineno(func)
        except:
            lineno = -1
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=UserWarning,
            filename=filename,
            lineno=lineno,
        )
        return func(*args, **kwargs)
    return new_func


def rdeprecated(txt=None, expire=None):
    """ This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used. 

    Args:
        txt (str): reason for deprecation
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
        when the function is used. """

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            try:
                filename = inspect.getfile(func)
            except:
                filename = '?'
            try:
                lineno = inspect.getlineno(func)
            except:
                lineno = -1
            if txt is None:
                etxt = ''
            else:
                etxt = ' ' + txt

            if expire is not None:
                if expired:
                    raise Exception("Call to deprecated function {}.{}".format(func.__name__, etxt))
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
                    "Call to deprecated function {}.{}".format(func.__name__, etxt),
                    category=UserWarning,
                    filename=filename,
                    lineno=lineno,
                )
            return func(*args, **kwargs)
        return new_func
    return deprecated_inner


def test_rdeprecated():

    @rdeprecated('hello')
    def dummy():
        pass

    @rdeprecated('hello', expire='1-1-2400')
    def dummy2():
        pass

#%%


def update_dictionary(alldata, **kwargs):
    """ Update elements of a dictionary

    Args:
        alldata (dict): dictionary to be updated
        kwargs (dict): keyword arguments

    """
    for k in kwargs:
        alldata[k] = kwargs[k]


def stripDataset(dataset):
    """ Make sure a dataset can be pickled 

    Args: 
        dataset (qcodes DataSet)
    Returns:
        dataset (qcodes DataSet): the dataset from the function argument
    """
    dataset.sync()
    dataset.data_manager = None
    dataset.background_functions = {}
    #dataset.formatter = qcodes.DataSet.default_formatter
    try:
        dataset.formatter.close_file(dataset)
    except:
        pass

    if 'scanjob' in dataset.metadata:
        if 'minstrumenthandle' in dataset.metadata['scanjob']:
            dataset.metadata['scanjob']['minstrumenthandle'] = str(
                dataset.metadata['scanjob']['minstrumenthandle'])

    return dataset

#%%


def negfloat(x):
    ''' Helper function '''
    return -float(x)


def checkPickle(obj, verbose=0):
    """ Check whether an object can be pickled

    Args:
        obj (object): object to be checked
    Returns:
        c (bool): True of the object can be pickled
    """
    try:
        _ = pickle.dumps(obj)
    except Exception as ex:
        if verbose:
            print(ex)
        return False
    return True


def freezeclass(cls):
    """ Decorator to freeze a class """
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

#%%


def resampleImage(im):
    """ Resample the image so it has the similar sample rates (samples/mV) in both axis

    Args:
        im (DataArray): input image
    Returns:
        imr (numpy array): resampled image
        setpoints (list of 2 numpy arrays): setpoint arrays from resampled image
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
            setpointy = qcodes.DataArray(name='Resampled_' + setpoints[0].array_id, array_id='Resampled_' + setpoints[0].array_id, label=setpoints[0].label,
                                         unit=setpoints[0].unit, preset_data=spy, is_setpoint=True)
            setpointx = qcodes.DataArray(name='Resampled_' + setpoints[1].array_id, array_id='Resampled_' + setpoints[1].array_id, label=setpoints[1].label,
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
            setpointx = qcodes.DataArray(name='Resampled_' + idx, array_id='Resampled_' + idy, label=setpoints[1].label,
                                         unit=setpoints[1].unit, preset_data=spx, is_setpoint=True)
            setpoints = [setpoints[0], setpointx]

    return im, setpoints


def diffImage(im, dy, size=None):
    """ Simple differentiation of an image

    Args:
        im (numpy array): input image
        dy (integer or string): method of differentiation
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
    """ Simple differentiation of an image

    Parameters
    ----------
    im : array
        input image
    dy : string or integer
        direction of differentiation. can be 'x' (0) or 'y' (1) or 'xy' (2) or 'g' (3)
        or 
    sigma : float
        parameter for gaussian filter kernel

    """
    if sigma is None:
        imx = diffImage(im, dy)
        return imx

    if dy is None:
        imx = im.copy()
    elif dy == 0 or dy == 'x':
        imx = ndimage.gaussian_filter1d(
            im, axis=1, sigma=sigma, order=1, mode='nearest')
    elif dy == 1 or dy == 'y':
        imx = ndimage.gaussian_filter1d(
            im, axis=0, sigma=sigma, order=1, mode='nearest')
    elif dy == -1:
        imx = -ndimage.gaussian_filter1d(im, axis=0,
                                         sigma=sigma, order=1, mode='nearest')
    elif dy == 2 or dy == 3 or dy == 'xy' or dy == 'xmy' or dy == 'xmy2' or dy == 'g' or dy == 'x2my2' or dy == 'x2y2':
        imx0 = ndimage.gaussian_filter1d(
            im, axis=1, sigma=sigma, order=1, mode='nearest')
        imx1 = ndimage.gaussian_filter1d(
            im, axis=0, sigma=sigma, order=1, mode='nearest')
        if dy == 2 or dy == 'xy':
            imx = imx0 + imx1
        if dy == 'xmy':
            imx = imx0 - imx1
        if dy == 3 or dy == 'g':
            imx = np.sqrt(imx0**2 + imx1**2)
        if dy == 'xmy2':
            warnings.warn('please do not use this option')
            imx = np.sqrt(imx0**2 + imx1**2)
        if dy == 'x2y2':
            imx = imx0**2 + imx1**2
        if dy == 'x2my2':
            imx = imx0**2 - imx1**2
    else:
        raise Exception('differentiation method %s not supported' % dy)
    return imx


def test_array(location=None, name=None):
    # DataSet with one 2D array with 4 x 6 points
    yy, xx = np.meshgrid(np.arange(0, 10, .5), range(3))
    zz = xx**2 + yy**2
    # outer setpoint should be 1D
    xx = xx[:, 0]
    x = DataArray(name='x', label='X', preset_data=xx, is_setpoint=True)
    y = DataArray(name='y', label='Y', preset_data=yy, set_arrays=(x,),
                  is_setpoint=True)
    z = DataArray(name='z', label='Z', preset_data=zz, set_arrays=(x, y))
    return z


def test_image_operations(verbose=0):
    import qcodes.tests.data_mocks

    if verbose:
        print('testing resampleImage')
    ds = qcodes.tests.data_mocks.DataSet2D()
    imx, setpoints = resampleImage(ds.z)

    z = test_array()
    imx, setpoints = resampleImage(z)
    if verbose:
        print('testing diffImage')
    d = diffImage(ds.z, dy='x')

#%%


import dateutil


def scanTime(dd):
    """ Return date a scan was performed """
    w = dd.metadata.get('scantime', None)
    if isinstance(w, str):
        w = dateutil.parser.parse(w)
    return w


@deprecated
def plot_parameter(data, default_parameter='amplitude'):
    """ Return parameter to be plotted """
    if 'main_parameter' in data.metadata.keys():
        return data.metadata['main_parameter']
    if default_parameter in data.arrays.keys():
        return default_parameter
    try:
        key = next(iter(data.arrays.keys()))
        return key
    except:
        return None


@deprecated
def plot1D(dataset, fig=1):
    """ Simlpe plot function """
    if isinstance(dataset, qcodes.DataArray):
        array = dataset
        dataset = None
    else:
        # assume we have a dataset
        arrayname = plot_parameter(dataset)
        array = getattr(dataset, arrayname)

    if fig is not None and array is not None:
        MatPlot(array, num=fig)


#%%

def showImage(im, extent=None, fig=None, title=None):
    """ Show image in figure window

    Args:
        im (array)
        extend (list): matplotlib style image extent
        fig (None or int): figure window to show image        
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


#%% Measurement tools

@deprecated  # part of the gates object
def resetgates(gates, activegates, basevalues=None, verbose=2):
    """ Reset a set of gates to default values

    Parameters
    ----------
    activegates : list or dict
        list of gates to reset
    basevalues: dict
        new values for the gates
    verbose : integer
        output level

    """
    if verbose:
        print('resetgates: setting gates to default values')
    for g in (activegates):
        if basevalues == None:
            val = 0
        else:
            if g in basevalues.keys():
                val = basevalues[g]
            else:
                val = 0
        if verbose >= 2:
            print('  setting gate %s to %.1f [mV]' % (g, val))
        gates.set(g, val)

#%% Tools from pgeometry


@deprecated
def plot2Dline(line, *args, **kwargs):
    """ Plot a 2D line in a matplotlib figure

    line: 3x1 array

    >>> plot2Dline([-1,1,0], 'b')
    """
    if np.abs(line[1]) > .001:
        xx = plt.xlim()
        xx = np.array(xx)
        yy = (-line[2] - line[0] * xx) / line[1]
        plt.plot(xx, yy, *args, **kwargs)
    else:
        yy = np.array(plt.ylim())
        xx = (-line[2] - line[1] * yy) / line[0]
        plt.plot(xx, yy, *args, **kwargs)


def cfigure(*args, **kwargs):
    """ Create Matplotlib figure with copy to clipboard functionality

    By pressing the 'c' key figure is copied to the clipboard

    """
    if 'facecolor' in kwargs:
        fig = plt.figure(*args, **kwargs)
    else:
        fig = plt.figure(*args, facecolor='w', **kwargs)
    ff = lambda xx, figx=fig: mpl2clipboard(fig=figx)
    fig.canvas.mpl_connect('key_press_event', ff)  # mpl2clipboard)
    return fig


def static_var(varname, value):
    """ Helper function to create a static variable """
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


try:
    import qtpy.QtGui as QtGui
    import qtpy.QtWidgets as QtWidgets

    def monitorSizes(verbose=0):
        """ Return monitor sizes """
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
except:
    def monitorSizes(verbose=0):
        """ Dummy function for monitor sizes """
        return [[0, 0, 1600, 1200]]
    pass


from qtt.pgeometry import tilefigs, mkdirc  # import for backwards compatibility


#%% Helper tools

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
    except:
        ipversion = 'None'

    pversion = '.'.join('%s' % x for x in sys.version_info[0:3])
    print('python %s, ipython %s, notebook %s' %
          (pversion, ipversion, in_ipynb()))


#%%
try:
    import win32com
    import win32com.client

    def addPPTslide(title=None, fig=None, txt=None, notes=None, figsize=None,
                    subtitle=None, maintext=None, show=False, verbose=1,
                    activate_slide=True, ppLayout=None, extranotes=None):
        """ Add slide to current active Powerpoint presentation

        Arguments:
            title (str): title added to slide
            fig (matplotlib.figure.Figure or qcodes.plots.pyqtgraph.QtPlot or integer): 
                figure added to slide
            subtitle (str): text added to slide as subtitle
            maintext (str): text in textbox added to slide
            notes (str or QCoDeS station): notes added to slide
            figsize (list): size (width,height) of figurebox to add to powerpoint
            show (boolean): shows the powerpoint application
            verbose (int): print additional information
        Returns:
            ppt: PowerPoint presentation
            slide: PowerPoint slide

        The interface to Powerpoint used is described here:
            https://msdn.microsoft.com/en-us/library/office/ff743968.aspx

        Example:
            >>> title = 'An example title'
            >>> fig = plt.figure(10)
            >>> txt = 'Some comments on the figure'
            >>> notes = 'some additional information' 
            >>> addPPTslide(title,fig, subtitle = txt,notes = notes)
        """
        Application = win32com.client.Dispatch("PowerPoint.Application")

        if verbose >= 2:
            print('num of open PPTs: %d' % Application.presentations.Count)

        # ppt = Application.Presentations.Add()
        try:
            ppt = Application.ActivePresentation
        except Exception:
            print(
                'could not open active Powerpoint presentation, opening blank presentation')
            try:
                ppt = Application.Presentations.Add()
            except Exception as ex:
                warnings.warn(
                    'could not make connection to Powerpoint presentation')
                return None, None

        if show:
            Application.Visible = True  # shows what's happening, not required, but helpful for now

        if verbose >= 2:
            print('addPPTslide: presentation name: %s' % ppt.Name)

        ppLayoutTitleOnly = 11
        ppLayoutTitle = 1
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

        if verbose:
            print('addPPTslide: presentation name: %s, adding slide %d' %
                  (ppt.Name, ppt.Slides.count + 1))

        slide = ppt.Slides.Add(ppt.Slides.Count + 1, ppLayout)
        if fig is None:
            titlebox = slide.shapes.Item(1)
            mainbox = slide.shapes.Item(2)
            if maintext is None:
                raise TypeError('maintext argument is None')
            mainbox.TextFrame.TextRange.Text = maintext
        else:
            titlebox = slide.shapes.Item(1)
            mainbox = None
            if maintext is not None:
                warnings.warn('maintext not implemented when figure is set')

        if title is not None:
            slide.shapes.title.textframe.textrange.text = title
        else:
            slide.shapes.title.textframe.textrange.text = 'QCoDeS measurement'

        import qtt.measurements.ttrace  # should be moved to top when circular references are fixed
        from qtt.measurements.videomode import VideoMode  # import here, to prevent default imports of gui code

        if fig is not None:
            fname = tempfile.mktemp(prefix='qcodesimageitem', suffix='.png')
            if isinstance(fig, matplotlib.figure.Figure):
                fig.savefig(fname)
            elif isinstance(fig, int):
                fig = plt.figure(fig)
                fig.savefig(fname)
            elif isinstance(fig, qtt.measurements.ttrace.MultiTracePlot) or \
                    fig.__class__.__name__ == 'MultiTracePlot':
                figtemp = fig.plotwin.grab()
                figtemp.save(fname)
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
                    # new Qt style
                    figtemp = fig.lp.plotwin.grab()
                    figtemp.save(fname)
            elif isinstance(fig, QtWidgets.QWidget):
                # generic method
                figtemp = fig.plotwin.grab()
                figtemp.save(fname)
            elif isinstance(fig, QtWidgets.QWidget):
                try:
                    figtemp = QtGui.QPixmap.grabWidget(fig)
                except:
                    # new Qt style
                    figtemp = fig.grab()
                figtemp.save(fname)
            elif isinstance(fig, qcodes.plots.pyqtgraph.QtPlot):
                fig.save(fname)
            else:
                if verbose:
                    raise TypeError('figure is of an unknown type %s' % (type(fig), ))
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
                except:
                    imwh = None
                if imwh is not None:
                    imratio = imwh[0] / imwh[1]
                    slideratio = slidewh[0] / slidewh[1]
                    if verbose >= 1:
                        print(' image aspect ratio %.2f, slide aspect ratio %.2f' % (imratio, slideratio))
                    if slideratio > imratio:
                        # wide slide, so make the image width smaller
                        print('adjust width %d->%d' % (width, height * imratio))
                        width = height * imratio
                    else:
                        # wide image, so make the image height smaller
                        print('adjust height %d->%d' % (height, width / imratio))
                        height = int(width / imratio)

                if verbose:
                    print('slide width height: %s' % (slidewh,))
                    print('image width height: %d, %d' % (width, height))
            if verbose >= 2:
                print('fname %s' % fname)
            slide.Shapes.AddPicture(FileName=fname, LinkToFile=False,
                                    SaveWithDocument=True, Left=left, Top=top, Width=width, Height=height)

        if subtitle is not None:
            # add subtitle
            subtitlebox = slide.Shapes.AddTextbox(
                1, Left=100, Top=80, Width=500, Height=300)
            subtitlebox.Name = 'subtitle box'
            subtitlebox.TextFrame.TextRange.Text = subtitle

        if notes is None:
            warnings.warn(
                'please set notes for the powerpoint slide. e.g. use the station or reshape_metadata')

        if isinstance(notes, qcodes.Station):
            station = notes
            gates = getattr(station, 'gates', None)
            notes = reshape_metadata(station, printformat='s', add_scanjob=True)
            if extranotes is not None:
                notes = '\n' + extranotes + '\n' + notes
            if gates is not None:
                notes = 'gates: ' + str(gates.allvalues()) + '\n\n' + notes
        if isinstance(notes, qcodes.DataSet):
            notes = reshape_metadata(notes, printformat='s')

        if notes is not None:
            if notes == '':
                notes = ' '
            slide.notespage.shapes.placeholders[
                2].textframe.textrange.insertafter(notes)

        # ActivePresentation.Slides(ActiveWindow.View.Slide.SlideNumber).
        # s=Application.ActiveWindow.Selection

        # slide.SetName('qcodes measurement')

        if activate_slide:
            idx = int(slide.SlideIndex)
            if verbose >= 2:
                print('addPPTslide: goto slide %d' % idx)
            Application.ActiveWindow.View.GotoSlide(idx)
        return ppt, slide

    def addPPT_dataset(dataset, title=None, notes=None,
                       show=False, verbose=1, paramname='measured',
                       printformat='fancy', customfig=None, extranotes=None, **kwargs):
        """ Add slide based on dataset to current active Powerpoint presentation

        Arguments:
            dataset (DataSet): data and metadata from DataSet added to slide
            customfig (QtPlot): custom QtPlot object to be added to
                                slide (for dataviewer)
            notes (string): notes added to slide
            show (boolean): shows the powerpoint application
            verbose (int): print additional information
            paramname (None or str): passed to dataset.default_parameter_array
            printformat (string): 'fancy' for nice formatting or 'dict'
                                  for easy copy to python
        Returns:
            ppt: PowerPoint presentation
            slide: PowerPoint slide

        Example
        -------
        >>> notes = 'some additional information'
        >>> addPPT_dataset(dataset,notes)
        """
        if len(dataset.arrays) < 2:
            raise IndexError('The dataset contains less than two data arrays')

        if customfig is None:
            
            if title is None:
                parameter_name = dataset.default_parameter_name(paramname=paramname)
                title = 'Parameter: %s'  % parameter_name
            temp_fig = QtPlot(dataset.default_parameter_array(
                              paramname=paramname), show_window=False)
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

except:
    def addPPTslide(title=None, fig=None, subtitle=None, maintext=None,
                    notes=None, show=False, verbose=1, ppLayout=1):
        ''' Dummy implementation '''
        warnings.warn('addPPTslide is not available on your system')

    def addPPT_dataset(dataset, title=None, notes=None, show=False, verbose=1):
        ''' Dummy implementation '''
        warnings.warn('addPPT_dataset is not available on your system')

#%%
from collections import OrderedDict


def reshape_metadata(dataset, printformat='dict', add_scanjob=True, verbose=0):
    '''Reshape the metadata of a DataSet

    Arguments:
        dataset (DataSet or qcodes.Station): a dataset of which the metadata 
                                             will be reshaped.
        printformat (str): can be 'dict' or 'txt','fancy' (text format)
    Returns:
        metadata (string): the reshaped metadata
    '''

    if isinstance(dataset, qcodes.Station):
        station = dataset
        all_md = station.snapshot(update=False)['instruments']
        header = None
    else:
        if not 'station' in dataset.metadata:
            return 'dataset %s: no metadata available' % (str(dataset.location), )

        tmp = dataset.metadata.get('station', None)
        if tmp is None:
            all_md = {}
        else:
            all_md = tmp['instruments']

        header = 'dataset: %s' % dataset.location

        if hasattr(dataset.io, 'base_location'):
            header += ' (base %s)' % dataset.io.base_location

    scanjob = dataset.metadata.get('scanjob', None)
    if scanjob is not None and add_scanjob:
        s = pprint.pformat(scanjob)
        header += '\n\nscanjob: ' + str(s) + '\n'

    metadata = OrderedDict()
    # make sure the gates instrument is in front
    all_md_keys = sorted(sorted(all_md), key=lambda x: x ==
                         'gates',  reverse=True)
    for x in all_md_keys:
        metadata[x] = OrderedDict()
        if 'IDN' in all_md[x]['parameters']:
            metadata[x]['IDN'] = dict({'name': 'IDN', 'value': all_md[
                                      x]['parameters']['IDN']['value']})
            metadata[x]['IDN']['unit'] = ''
        for y in sorted(all_md[x]['parameters'].keys()):
            try:
                if y != 'IDN':
                    metadata[x][y] = OrderedDict()
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
            s = metadata[k]
            ss += '\n## %s:\n' % k
            for p in s:
                pp = s[p]
                if verbose:
                    print('  --- %s: %s' % (p, pp.get('value', '??')))
                ss += '%s: %s (%s)' % (pp['name'],
                                       pp.get('value', '?'), pp.get('unit', ''))
                ss += '\n'

    if header is not None:
        ss = header + '\n\n' + ss
    return ss


def test_reshape_metadata():
    import qtt.measurements.scans
    param = qcodes.ManualParameter('dummy')
    try:
        dataset = qcodes.Loop(param[0:1:10]).each(param).run()
    except:
        dataset = None
        pass
    if dataset is not None:
        _ = reshape_metadata(dataset, printformat='dict')
    instr = qcodes.Instrument(qtt.measurements.scans.instrumentName('_dummy_test_reshape_metadata_123'))
    st = qcodes.Station(instr)
    _ = reshape_metadata(st, printformat='dict')
    instr.close()


#%%

def setupMeasurementWindows(*args, **kwargs):
    raise Exception('use qtt.gui.live_plotting.setupMeasurementWindows instead')


def updatePlotTitle(qplot, basetxt='Live plot'):
    """ Update the plot title of a QtPlot window """
    txt = basetxt + ' (%s)' % time.asctime()
    qplot.win.setWindowTitle(txt)


@rdeprecated(expire='1 Sep 2018')
def timeProgress(data):
    ''' Simpe progress meter, should be integrated with either loop or data object '''
    data.sync()
    tt = data.arrays['timestamp']
    vv = ~np.isnan(tt)
    ttx = tt[vv]
    t0 = ttx[0]
    t1 = ttx[-1]

    logging.debug('t0 %f t1 %f' % (t0, t1))

    fraction = ttx.size / tt.size[0]
    remaining = (t1 - t0) * (1 - fraction) / fraction
    return fraction, remaining

#%%


def flatten(lst):
    ''' Flatten a list

    Args:
        lst (list): list to be flattened
    Returns:
        lstout (list): flattened list
    Example:
        >>> flatten([ [1,2], [3,4], [10] ])
        [1, 2, 3, 4, 10]
    '''
    return list(chain(*lst))

#%%


def cutoffFilter(x, thr, omega):
    """ Smooth cutoff filter

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

#%%


def smoothFourierFilter(fs=100, thr=6, omega=2, fig=None):
    """ Create smooth ND filter for Fourier high or low-pass filtering

    Example
    -------
    >>> F=smoothFourierFilter([24,24], thr=6, omega=2)
    >>> _=plt.figure(10); plt.clf(); _=plt.imshow(F, interpolation='nearest')
    """
    rr = np.meshgrid(*[range(f) for f in fs])

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


#%%

def fourierHighPass(imx, nc=40, omega=4, fs=1024, fig=None):
    """ Implement simple high pass filter using the Fourier transform """
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


#%%
import copy


def slopeClick(drawmode='r--', **kwargs):
    ''' Calculate slope for linepiece of two points clicked by user. Works 
    with matplotlib but not with pyqtgraph. Uses the currently active 
    figure.

    Arguments:
        drawmode (string): plotting style

    Returns:
        coords (2 x 2 array): coordinates of the two clicked points
        signedslope (float): slope of linepiece connecting the two points
    '''
    ax = plt.gca()
    ax.set_autoscale_on(False)
    coords = pgeometry.ginput(2, drawmode, **kwargs)
    plt.pause(1e-6)
    signedslope = (coords[1, 0] - coords[1, 1]) / (coords[0, 0] - coords[0, 1])

    return coords, signedslope


def clickGatevals(plot, drawmode='ro'):
    ''' Get gate values for all gates at clicked point in a heatmap.

    Arguments:
        plot (qcodes MatPlot object): plot of measurement data
        drawmode (string): plotting style

    Returns:    
        gatevals (dict): values of the gates at clicked point
    '''
    # TODO: implement for virtual gates
    if type(plot) != qcodes.plots.qcmatplotlib.MatPlot:
        raise Exception(
            'The plot object is not based on the MatPlot class from qcodes.')

    ax = plt.gca()
    ax.set_autoscale_on(False)
    coords = pgeometry.ginput(drawmode=drawmode)
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

#%%


def connect_slot(target):
    """ Create a slot by dropping signal arguments. """
    def signal_drop_arguments(*args, **kwargs):
        target()
    return signal_drop_arguments
