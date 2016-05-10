import qtpy
#print(qtpy.API_NAME)

import numpy as np
import scipy
import sys
import logging
import qcodes

import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets

from pmatlab import tilefigs


#import pmatlab; pmatlab.qtmodules(verbose=1)

#%%

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
        ipversion='.'.join('%s' % x for x in IPython.version_info[:-1])
    except:
        ipversion='None'        
            
            
    pversion='.'.join('%s' % x for x in sys.version_info[0:3])
    print('python %s, ipython %s, notebook %s' %( pversion, ipversion, in_ipynb() ))

#%%

try:
    import graphviz
except:
    pass
import matplotlib.pyplot as plt

def showDotGraph(dot, fig=10):
    dot.format='png'
    outfile=dot.render('dot-dummy', view=False)    
    print(outfile)
    
    im=plt.imread(outfile)
    plt.figure(fig)
    plt.clf()    
    plt.imshow(im)
    plt.tight_layout()
    plt.axis('off')
    
#%%

from qtt.qtt_toymodel import ParameterViewer
from qtt.logviewer import LogViewer

def setupMeasurementWindows(station):
    ms=monitorSizes()
    vv=ms[-1]
    # create custom viewer which gathers data from a station object
    w = ParameterViewer(station)
    w.setGeometry(vv[0]+vv[2]-400-300,vv[1],300,600)
    w.updatecallback()

    plotQ = qcodes.QtPlot(windowTitle='Live plot', remote=False)
    plotQ.win.setGeometry(vv[0]+vv[2]-300,vv[1]+vv[3]-400,600,400)
    plotQ.update()


    logviewer = LogViewer()
    logviewer.setGeometry(vv[0]+vv[2]-400,vv[1],400,600)
    logviewer.qplot.win.setMaximumHeight(400)
    logviewer.show()

    return dict({'parameterviewer': w, 'plotwindow': plotQ, 'dataviewer': logviewer} )


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


def cutoffFilter(x, thr, omega):
    """ Smooth cutoff filter
    
    Filter definition from: http://paulbourke.net/miscellaneous/imagefilter/
    
    Example
    -------
    
    >>> plt.clf()
    >>> x=np.arange(0, 4, .01)
    >>> _=plt.plot(x, cutoffFilter(x, 2, .25), '-r')
    
    """
    y=.5*(1-np.sin(np.pi*(x-thr)/(2*omega)))
    y[x<thr-omega]=1
    y[x>thr+omega]=0
    return y

#%%    
def smoothFourierFilter(fs=100, thr=6, omega=2, fig=None):
    """ Create smooth ND filter for Fourier high or low-pass filtering

    >>> F=smoothFourierFilter([24,24], thr=6, omega=2)    
    >>> _=plt.figure(10); plt.clf(); _=plt.imshow(F, interpolation='nearest')
    
    """
    rr=np.meshgrid(*[range(f) for f in fs])
    
    x=np.dstack(rr)
    x=x-(np.array(fs)/2 - .5)
    x=np.linalg.norm(x, axis=2)
    #showIm(x);
    
    F=cutoffFilter(x, thr, omega)

    if fig is not None:
        plt.figure(10); plt.clf();
        plt.imshow(F, interpolation='nearest')
    
    return F#, rr
F=smoothFourierFilter([36,36])    


#%%
    
def fourierHighPass(imx, nc=40, omega=4, fs=1024, fig=None):
    """ Implement simple high pass filter using the Fourier transform """
    f = np.fft.fft2(imx, s=[fs,fs])                  #do the fourier transform
    
    fx=np.fft.fftshift(f)

    if fig:
        plt.figure(fig); plt.clf()
        plt.imshow(np.log(np.abs(f)+1), interpolation='nearest')
        #plt.imshow(f.real, interpolation='nearest')
        plt.title('Fourier spectrum (real part)' )
        plt.figure(fig+1); plt.clf()
        #plt.imshow(fx.real, interpolation='nearest')
        #plt.imshow(np.sign(np.real(fx))*np.log(np.abs(fx)+1), interpolation='nearest')
        plt.imshow(np.log(np.abs(fx)+1), interpolation='nearest')
        plt.title('Fourier spectrum (real part)' )

    if nc>0 and omega==0:
        f[0:nc,0:nc]=0
        f[-nc:,-nc:]=0
        f[-nc:,0:nc]=0
        f[0:nc,-nc:]=0
        img_back = np.fft.ifft2(f)     #inverse fourier transform

    else:
        # smooth filtering

        F=1-smoothFourierFilter(fx.shape, thr=nc, omega=omega)    
        fx=F*fx
        ff = np.fft.ifftshift(fx)  #inverse shift
        img_back = np.fft.ifft2(ff)     #inverse fourier transform
        
    imf=img_back.real
    imf=imf[0:imx.shape[0], 0:imx.shape[1]]
    return imf
    