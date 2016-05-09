import numpy as np
import scipy
import sys

from pmatlab import tilefigs

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
    >>> plt.plot(x, cutoffFilter(x, 2, .25), '-r')
    
    """
    y=.5*(1-np.sin(np.pi*(x-thr)/(2*omega)))
    y[x<thr-omega]=1
    y[x>thr+omega]=0
    return y

#%%    
def smoothFourierFilter(fs=100, thr=6, omega=2, fig=None):
    """ Create smooth ND filter for Fourier high or low-pass filtering

    >>> F, rr=smoothFourierFilter([24,24], thr=6, omega=2)    
    >>> plt.figure(10); plt.clf(); plt.imshow(F, interpolation='nearest')
    
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
    