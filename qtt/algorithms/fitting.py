""" Fitting of Fermi-Dirac distributions. """
import numpy as np
import scipy
import matplotlib.pyplot as plt

import qtt.pgeometry
from qcodes import DataArray
from qtt.algorithms.functions import FermiLinear, linear_function, Fermi


def initFermiLinear(xdata, ydata, fig=None):
    """ Initalization of fitting a FermiLinear function 

    First the linear part is estimated, then the Fermi part of the function    
    """
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    n = xdata.size
    nx = int(np.ceil(n / 5))

    if nx < 4:
        p1, _ = scipy.optimize.curve_fit(linear_function, np.array(xdata[0:100]), np.array(ydata[0:100]))

        a = p1[0]
        b = p1[1]
        ab = [a, b]
        y = linear_function(xdata, ab[0], ab[1])
        cc = np.mean(xdata)
        A = 0
        T = np.std(xdata) / 10
        ff = [cc, A, T]
    else:
        # guess initial linear part
        mx = np.mean(xdata)
        my = np.mean(ydata)
        dx = np.hstack((np.diff(xdata[0:nx]), np.diff(xdata[-nx:])))
        dx = np.mean(dx)
        dd = np.hstack((np.diff(ydata[0:nx]), np.diff(ydata[-nx:])))
        dd = np.convolve(dd, np.array([1., 1, 1]) / 3)  # smooth
        if dd.size > 15:
            dd = np.array(sorted(dd))
            w = int(dd.size / 10)
            a = np.mean(dd[w:-w]) / dx
        else:
            a = np.mean(dd) / dx
        b = my - a * mx
        xx = np.hstack((xdata[0:nx], xdata[-nx:]))
        ab = [a, b]
        y = linear_function(xdata, ab[0], ab[1])

        # subtract linear part
        yr = ydata - y

        cc = np.mean(xdata)
        h = int(xdata.size / 2)
        A = -(np.mean(yr[h:]) - np.mean(yr[:h]))
        T = np.std(xdata) / 10
        ab[1] = ab[1] - A / 2  # correction
        ylin = linear_function(xdata, ab[0], ab[1])

        # subtract linear part
        yr = ydata - ylin
        ff = [cc, A, T]
    if fig is not None:
        yf = FermiLinear(xdata, ab[0], ab[1], *ff)

        xx = np.hstack((xdata[0:nx], xdata[-nx:]))
        yy = np.hstack((ydata[0:nx], ydata[-nx:]))
        plt.figure(fig)
        plt.clf()
        plt.plot(xdata, ydata, '.b', label='raw data')
        plt.plot(xx, yy, 'ok')
        qtt.pgeometry.plot2Dline([-1,0,cc],':c', label='center')
        plt.plot(xdata, ylin, '-m', label='fitted linear function')
        plt.plot(xdata, yf, '-m', label='fitted FermiLinear function')

        plt.title('initFermiLinear', fontsize=12)
        plt.legend(numpoints=1)

        plt.figure(fig + 1)
        plt.clf()
        plt.plot(xdata, yr, '.b', label='Fermi part')
        f = Fermi(xdata, cc, A, T)
        plt.plot(xdata, f, '-m', label='estimated')
        plt.plot(xdata, f, '-m', label='estimated')
        #plt.plot(xdata, yr, '.b', label='Fermi part')

        plt.legend()
    return ab, ff


#%%


def fitFermiLinear(xdata, ydata, verbose=1, fig=None, l=1.16):
    """ Fit data to a Fermi-Linear function 

    Arguments:
        xdata, ydata (array): independent and dependent variable data

    Returns:
        p (array): fitted function parameters
        pp (dict): extra fitting data

    .. seealso:: FermiLinear
    """
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # initial values
    ab, ff = initFermiLinear(xdata, ydata, fig=None)
    p0 = ab + ff
    if 0:
        h = int(ydata.size / 2)
        amp = np.mean(ydata[h:]) - np.mean(ydata[:h])  # np.std(ydata)
        p0 = [0, np.mean(ydata), np.mean(xdata), amp, -.1]

    # fit
    func = lambda xdata, a, b, cc, A, T: FermiLinear(xdata, a, b, cc, A, T, l=l)
    pp = scipy.optimize.curve_fit(func, xdata, ydata, p0=p0)
    p = pp[0]

    if fig is not None:
        y = FermiLinear(xdata, *list(p))
        plt.figure(fig)
        plt.clf()
        plt.plot(xdata, ydata, '.b', label='data')
        plt.plot(xdata, y, 'm-', label='fitted FermiLinear')
        plt.legend(numpoints=1)
    return p, dict({'pp': pp, 'p0': p0})

#%%
def fit_addition_line(dataset, trimborder = True):
    """Fits a FermiLinear function to the addition line and finds the middle of the step.
    
    Args:
        dataset (qcodes dataset): 1d measured data of additionline
        trimborder (bool): determines if the edges of the data are taken into account for the fit
    
    Returns:
        m_addition_line (float): x value of the middle of the addition line
        pfit (array): fit parameters of the Fermi Linear function
        pguess (array): parameters of initial guess
        dataset_fit (qcodes dataset): dataset with fitted Fermi Linear function
        dataset_guess (qcodes dataset):dataset with guessed Fermi Linear function
        
    See also: FermiLinear and fitFermiLinear
    """
    y_array = dataset.default_parameter_array()
    setarray = y_array.set_arrays[0]
    xdata = np.array(setarray)
    ydata = np.array(y_array)
    
    if trimborder:
        ncut = max(min(int(xdata.size/40), 100),1)
        xdata, ydata, setarray =xdata[ncut:-ncut], ydata[ncut:-ncut], setarray[ncut:-ncut]
        
    # fitting of the FermiLinear function
    pp = fitFermiLinear(xdata, ydata, verbose=1, fig=None)
    pfit = pp[0] #fit parameters
    pguess = pp[1]['p0'] #initial guess parameters
    y0 = FermiLinear(xdata, *list(pguess) )
    dataset_guess = DataArray(name='fit', label='fit', preset_data=y0,  set_arrays=(setarray,) )
    y = FermiLinear(xdata, *list(pfit) )
    dataset_fit = DataArray(name='fit', label='fit', preset_data=y,  set_arrays=(setarray,) )
    m_addition_line = pfit[2]
    result_dict = {'fit parameters': pfit, 'parameters initial guess': pguess, 'dataset fit': dataset_fit, 'dataset initial guess': dataset_guess}
    return  m_addition_line, result_dict

def test_fitfermilinear(fig=None):
    xdata=np.arange(-20, 10, .1)
    p0 = [0.01000295,  0.51806569, -4.88800525,  0.12838861,  0.25382811]
    
    ydata = FermiLinear(xdata, *p0)
    ydata += .01*np.random.rand(ydata.size)
    
    p, results=fitFermiLinear(xdata, ydata, verbose=1, fig=fig)
    if fig:
        print('fitted: %s' % p0)
        print('fitted: %s' % p)
        print('max diff: %.2f' % (np.abs(p-p0).max() ))
    assert(np.all(np.abs(p-p0)<1e-2))

if __name__=='__main__':
    test_fitfermilinear(fig=100)         