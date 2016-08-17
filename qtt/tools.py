import qtpy
# print(qtpy.API_NAME)

import numpy as np
import scipy
import matplotlib
import sys, os
import logging
import qcodes
import pickle

# explicit import
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot

import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets

# do NOT load any other qtt submodules here
import tempfile

#%% Debugging

def dumpstring(txt):
    with open(os.path.join(tempfile.tempdir, 'qtt-dump.txt'), 'a+t') as fid:
        fid.write(txt + '\n')

#%%
try:
    from qcodes.process.heartbeat import *
    import time

    def pauseHeartBeat():
        m = initHeartBeat(bfile, reinit=False)
        setHeartBeat(m, 0)
        for ii in range(5):
            print('pause %d: ...' % ii)
            time.sleep(1)
        setHeartBeat(m, 1)
except:
    pass
#%%

def negfloat(x):
    ''' Helper function '''
    return -float(x)


def checkPickle(obj):
    try:
        _ = pickle.dumps(obj)
    except:
        return False
    return True
# checkPickle(ivvi1)

#%%

import scipy.ndimage as ndimage


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
    if dy == 1 or dy == 'y':
        im = np.diff(im, n=1, axis=0)
        if size == 'same':
            im = np.vstack((im, im[-1:,:]))
    if dy == -1:
        im = -np.diff(im, n=1, axis=0)
        if size == 'same':
            im = np.vstack((im, im[-1:,:]))
    if dy == 2:
        imx = np.diff(im, n=1, axis=1)
        imy = np.diff(im, n=1, axis=0)
        im = imx[0:-1,:] + imy[:, 0:-1]
    return im


def diffImageSmooth(im, dy='x', sigma=2., size=None):
    """ Simple differentiation of an image

    Input
    -----
    im : array
        input image
    dy : string or integer
        direction of differentiation. can be 'x' (0) or 'y' (1) or 'xy' (2)
    sigma : float
        parameter for differentiation kernel

    """
    if sigma is None:
        imx = diffImage(im, dy)
        return imx

    if dy is None:
        imx = im.copy()
    if dy == 0 or dy == 'x':
        imx = ndimage.gaussian_filter1d(
            im, axis=1, sigma=sigma, order=1, mode='nearest')
    if dy == 1 or dy == 'y':
        imx = ndimage.gaussian_filter1d(im, axis=0, sigma=sigma, order=1, mode='nearest')
    if dy == -1:
        imx = -ndimage.gaussian_filter1d(im, axis=0, sigma=sigma, order=1, mode='nearest')
    if dy == 2 or dy == 3 or dy == 'xy' or dy == 'xmy' or dy == 'xmy2':
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
            imx = np.sqrt(imx0**2 + imx1**2)

    return imx

#%%

import dateutil
def scanTime(dd):
    w = dd.metadata.get('scantime', None)
    if isinstance(w, str):
        w = dateutil.parser.parse(w)
    return w

def plot_parameter(data, default_parameter='amplitude'):
    ''' Return parameter to be plotted '''
    if 'main_parameter'  in data.metadata.keys():
        return data.metadata['main_parameter']
    if default_parameter in data.arrays.keys():
        return default_parameter
    try:
        key = next(iter (data.arrays.keys()))
        return key
    except:
        return None

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

# import pmatlab



if __name__ == '__main__':
    plot1D(dataset, fig=10)
    plot1D(dataset.amplitude, fig=12)

#%%
def showImage(im, extent=None, fig=None):
    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plt.imshow(im, extent=extent, interpolation='nearest')
        if extent is not None:
            if extent[0] > extent[1]:
                plt.gca().invert_xaxis()


#%% Measurement tools
def resetgates(gates, activegates, basevalues=None, verbose=2):
    """ Reset a set of gates to default values

    Arguments
    ---------
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
            # print(g)
            # print(basevalues)
            if g in basevalues.keys():
                val = basevalues[g]
            else:
                val = 0
        if verbose >= 2:
            print('  setting gate %s to %.1f [mV]' % (g, val))
        gates.set(g, val)

#%% Tools from pmatlab

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
    fig.canvas.mpl_connect('key_press_event', ff) # mpl2clipboard)
    return fig

def static_var(varname, value):
    """ Helper function to create a static variable """
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


try:
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

@static_var('monitorindex', -1)
def tilefigs(lst, geometry=[2, 2], ww=None, raisewindows=False, tofront=False, verbose=0):
    """ Tile figure windows on a specified area """
    mngr = plt.get_current_fig_manager()
    be = matplotlib.get_backend()
    if ww is None:
        ww = monitorSizes()[tilefigs.monitorindex]

    w = ww[2] / geometry[0]
    h = ww[3] / geometry[1]

    # wm=plt.get_current_fig_manager()

    if isinstance(lst, int):
        lst = [lst]

    if verbose:
        print('tilefigs: ww %s, w %d h %d' % (str(ww), w, h))
    for ii, f in enumerate(lst):
        if type(f) == matplotlib.figure.Figure:
            fignum = f.number
        else:
            fignum = f
        if not plt.fignum_exists(fignum):
            if verbose >= 2:
                print('tilefigs: fignum: %s' % str(fignum))
            continue
        fig = plt.figure(fignum)
        iim = ii % np.prod(geometry)
        ix = iim % geometry[0]
        iy = np.floor(float(iim) / geometry[0])
        x = ww[0] + ix * w
        y = ww[1] + iy * h
        if verbose:
            print('ii %d: %d %d: f %d: %d %d %d %d' %
                  (ii, ix, iy, fignum, x, y, w, h))
            if verbose >= 2:
                print('  window %s' % mngr.get_window_title())
        if be == 'WXAgg':
            fig.canvas.manager.window.SetPosition((x, y))
            fig.canvas.manager.window.SetSize((w, h))
        if be == 'WX':
            fig.canvas.manager.window.SetPosition((x, y))
            fig.canvas.manager.window.SetSize((w, h))
        if be == 'agg':
            fig.canvas.manager.window.SetPosition((x, y))
            fig.canvas.manager.window.resize(w, h)
        if be == 'Qt4Agg' or be == 'QT4' or be == 'QT5Agg':
            # assume Qt canvas
            try:
                fig.canvas.manager.window.move(x, y)
                fig.canvas.manager.window.resize(w, h)
                fig.canvas.manager.window.setGeometry(x, y, w, h)
                # mngr.window.setGeometry(x,y,w,h)
            except Exception as e:
                print('problem with window manager: ', )
                print(be)
                print(e)
                pass
        if raisewindows:
            mngr.window.raise_()
        if tofront:
            plt.figure(f)


#%% Helper tools

def mkdirc(d):
    """ Similar to mkdir, but no warnings if the directory already exists """
    try:
        os.mkdir(d)
    except:
        pass
    return d

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
    print('python %s, ipython %s, notebook %s' % ( pversion, ipversion, in_ipynb() ))

#%%

try:
    import graphviz
except:
    pass
import matplotlib.pyplot as plt

def showDotGraph(dot, fig=10):
    dot.format = 'png'
    outfile = dot.render('dot-dummy', view=False)
    print(outfile)

    im = plt.imread(outfile)
    plt.figure(fig)
    plt.clf()
    plt.imshow(im)
    plt.tight_layout()
    plt.axis('off')

#%%
try:
    import win32com
    import win32com.client

    def addPPTslide(title=None, fig=None, txt=None, notes=None, show=False, verbose=1):
        ''' Add slide to current active Powerpoint presentation

        Arguments:
            title (string): title added to slide
            fig (matplotlib.figure.Figure or qcodes.plots.pyqtgraph.QtPlot): 
                figure added to slide
            txt (string): text in textbox added to slide
            notes (string): notes added to slide
            show (boolean): shows the powerpoint
            verbose (int): print additional information
        Returns:
            ppt: PowerPoint presentation
            slide: PowerPoint slide

        The interface to Powerpoint used is described here:
            https://msdn.microsoft.com/en-us/library/office/ff743968.aspx
            
        Example
        -------
        >>> title = 'An example title'
        >>> fig = plt.figure(10)
        >>> txt = 'Some comments on the figure'
        >>> notes = 'some additional information' 
        >>> addPPTslide(title,fig,txt,notes)
        '''
        Application = win32com.client.Dispatch("PowerPoint.Application")
        if show:
            Application.Visible = True # shows what's happening, not required, but helpful for now

        if verbose:
            print('num of open PPTs: %d' % Application.presentations.Count)

        # ppt = Application.Presentations.Add()
        ppt = Application.ActivePresentation
        if verbose:
            print('name: %s'  % ppt.Name)

        ppLayoutTitleOnly = 11
        layout = ppLayoutTitleOnly

        slide = ppt.Slides.Add(ppt.Slides.Count+1, layout)

        if title is not None:
            slide.shapes.title.textframe.textrange.text = title
        else:
            slide.shapes.title.textframe.textrange.text = 'QCoDeS measurement'

        if fig is not None:
            fname = tempfile.mktemp(prefix='qcodesimagetem', suffix='.png')
            if isinstance(fig, matplotlib.figure.Figure):
                fig.savefig(fname)
            elif isinstance(fig, QtWidgets.QWidget):
                figtemp = QtGui.QPixmap.grabWidget(fig)
                figtemp.save(fname)
            else:
                if verbose:                
                    print('figure is of an unknown type')
            slide.Shapes.AddPicture(FileName=fname, LinkToFile=False, SaveWithDocument=True, Left=100, Top=160, Width=560, Height=350)

        txtbox = slide.Shapes.AddTextbox(1, Left=100, Top=100, Width=500, Height=300)
        txtbox.Name = 'text'

        if txt is not None:
            txtbox.TextFrame.TextRange.Text = txt

        if notes is not None:
            slide.notespage.shapes.placeholders[2].textframe.textrange.insertafter(notes)

        # ActivePresentation.Slides(ActiveWindow.View.Slide.SlideNumber).
        # s=Application.ActiveWindow.Selection

        return ppt, slide

except:
    def addPPTslide(title=None, fig=None, txt=None, notes=None, show=False, verbose=1):
        ''' Dummy implementation '''
        pass

#%%

from qtt.parameterviewer import ParameterViewer
from qtt.dataviewer import DataViewer

def setupMeasurementWindows(station, ilist=None):
    ms=monitorSizes()
    vv=ms[-1]
    # create custom viewer which gathers data from a station object
    if ilist is None:
        ilist=[station.gates]
    w = ParameterViewer(ilist)
    w.setGeometry(vv[0]+vv[2]-400-300,vv[1],300,600)
    w.updatecallback()

    plotQ = QtPlot(windowtitle='Live plot', interval=.5)
    plotQ.setGeometry(vv[0]+vv[2]-300,vv[1]+vv[3]-400,600,400)
    plotQ.update()

    app=QtWidgets.QApplication.instance()
    app.processEvents()
    
    return dict({'parameterviewer': w, 'plotwindow': plotQ, 'dataviewer': None} )

import time
def updatePlotTitle(qplot, basetxt='Live plot'):
    txt = basetxt + ' (%s)' % time.asctime()
    qplot.win.setWindowTitle(txt)


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

def logistic(x, x0=0, alpha=1):
    """ Logistic function

    Arguments:
    x : array
        values
    x0, alpha : float
        parameters of function

    Example
    -------

    >>> y=logistic(0, 1, alpha=1)
    """
    f = 1 / (1 + np.exp(-2 * alpha * (x - x0)))
    return f

#%%

from itertools import chain
def flatten(lst):
    ''' Flatten a list
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
    y = .5*(1-np.sin(np.pi*(x-thr)/(2*omega)))
    y[x < thr-omega] = 1
    y[x > thr+omega] = 0
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
    x = x-(np.array(fs)/2 - .5)
    x = np.linalg.norm(x, axis=2)
    # showIm(x);

    F = cutoffFilter(x, thr, omega)

    if fig is not None:
        plt.figure(10)
        plt.clf()
        plt.imshow(F, interpolation='nearest')

    return F#, rr
F = smoothFourierFilter([36, 36])


#%%

def fourierHighPass(imx, nc=40, omega=4, fs=1024, fig=None):
    """ Implement simple high pass filter using the Fourier transform """
    f = np.fft.fft2(imx, s=[fs, fs])                  #do the fourier transform

    fx = np.fft.fftshift(f)

    if fig:
        plt.figure(fig)
        plt.clf()
        plt.imshow(np.log(np.abs(f)+1), interpolation='nearest')
        # plt.imshow(f.real, interpolation='nearest')
        plt.title('Fourier spectrum (real part)' )
        plt.figure(fig+1)
        plt.clf()
        # plt.imshow(fx.real, interpolation='nearest')
        # plt.imshow(np.sign(np.real(fx))*np.log(np.abs(fx)+1), interpolation='nearest')
        plt.imshow(np.log(np.abs(fx)+1), interpolation='nearest')
        plt.title('Fourier spectrum (real part)' )

    if nc > 0 and omega == 0:
        f[0:nc, 0:nc] = 0
        f[-nc:, -nc:] = 0
        f[-nc:, 0:nc] = 0
        f[0:nc, -nc:] = 0
        img_back = np.fft.ifft2(f)     #inverse fourier transform

    else:
        # smooth filtering

        F = 1-smoothFourierFilter(fx.shape, thr=nc, omega=omega)
        fx = F*fx
        ff = np.fft.ifftshift(fx)  #inverse shift
        img_back = np.fft.ifft2(ff)     #inverse fourier transform

    imf = img_back.real
    imf = imf[0:imx.shape[0], 0:imx.shape[1]]
    return imf
