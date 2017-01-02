import numpy as np
import scipy
import matplotlib.pyplot as plt

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
        if dd.size>15:
            dd = np.array(sorted(dd))
            w = int(dd.size / 10)
            a = np.mean(dd[w:-w]) / dx
        else:
            a=np.mean(dd)/dx
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

if __name__ == '__main__':
    ab, ff = initFermiLinear(xdata, ydata, fig=100)


#%%


def fitFermiLinear(xdata, ydata, verbose=1, fig=None):
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
    pp = scipy.optimize.curve_fit(FermiLinear, xdata, ydata, p0=p0)
    p = pp[0]

    if fig is not None:
        y = FermiLinear(xdata, *list(p))
        plt.figure(fig)
        plt.clf()
        plt.plot(xdata, ydata, '.b', label='data')
        plt.plot(xdata, y, 'm-', label='fitted FermiLinear')
        plt.legend(numpoints=1)
    return p, dict({'pp': pp, 'p0': p0})
