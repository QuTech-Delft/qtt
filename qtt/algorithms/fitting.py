import numpy as np
import scipy
import matplotlib.pyplot as plt

from qtt.algorithms.functions import FermiLinear

def fitFermiLinear(xdata, ydata, verbose=1, fig=None):
    """ Fit data to a Fermi-Linear function 
    
    Arguments:
        xdata, ydata (array): independent and dependent variable data
        
    Returns:
        p (array): fitted function parameters
        pp (dict): extra fitting data
        
    .. seealso:: FermiLinear
    """
    xdata=np.array(xdata)
    ydata=np.array(ydata)
    
    # initial values
    h=int(ydata.size/2)
    amp = np.mean(ydata[h:])-np.mean(ydata[:h]) # np.std(ydata)
    p0=[0, np.mean(ydata), np.mean(xdata), amp, -.1]
    
    # fit
    pp = scipy.optimize.curve_fit(FermiLinear, xdata, ydata, p0=p0)
    p=pp[0]


    if fig is not None:
        y = FermiLinear(xdata, *list(p) )
        plt.figure(fig); plt.clf()
        plt.plot(xdata, ydata, '.b', label='data')
        plt.plot(xdata, y, 'm-', label='fitted FermiLinear')
        plt.legend(numpoints=1)
    return p, dict({'pp': pp, 'p0': p0})
    