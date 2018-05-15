#%% Helper functions


import numpy as np
import matplotlib.pyplot as plt

import qcodes
import qtt
import scipy.optimize


def fitOhmic(ds, verbose=1, fig=None, gainy=1e-7, gainx=1e-6):
    """ Fit data to a linear function 

    Arguments:
        ds (dataset): independent and dependent variable data
        gainy (float): conversion factor from y-data to Ampere
        gainx (float): conversion factor from x-data to Volt
        verbose (int):
        fig (None or int): if an integer, plot the fitted data

    Returns:
        analysis_data (dict): fitted function parameters

    .. seealso:: linear_function
    """
    xdata = gainx * np.array(ds.default_parameter_array().set_arrays[0]) # [V]
    ydata = gainy * np.array(ds.default_parameter_array()) # [A]

    ydata = ydata
    # initial values: offset and slope
    slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    offset = np.mean(ydata) - slope * np.mean(xdata)
    p0 = [slope, offset]

    # fit
    offset = 2    # remove first data points
    func = lambda xdata, a, b: qtt.algorithms.functions.linear_function(xdata, a, b)
    pp = scipy.optimize.curve_fit(func, xdata[offset:], ydata[offset:], p0=p0)
    fitparam = pp[0]
    r = 1 / fitparam[0]
    biascurrent = qtt.algorithms.functions.linear_function(0, *list(fitparam))
    
    if fig is not None:
        plt.figure(fig)
        plt.clf()
        yf = 1e9

        plt.plot(1e6 * xdata, yf * ydata, '.b')
        plt.xlabel('Bias [uV]')
        plt.ylabel('Current [nA]')
        y = qtt.algorithms.functions.linear_function(xdata, *list(fitparam))
        plt.plot(1e6 * xdata, yf * y, 'm-', label='fitted linear function')
        ax=plt.gca()
        ax.axvline(x=0, linestyle=':')
        ax.axhline(y=yf*biascurrent, linestyle=':')
        plt.legend(numpoints=1)
        plt.title(('dataset: %s: resistance %.1f [kOhm]' % (ds.location, r * 1e-3)))
    return {'fitparam': fitparam, 'resistance': r, 'biascurrent': biascurrent, 'description': 'ohmic'}


def test_fitohmic():
    import qcodes.tests.data_mocks
    ds = qcodes.tests.data_mocks.DataSet1D()
    x=np.arange(-200, 200)
    y=1e-10*(x+50+20*np.random.rand(x.size) )
    ds = qtt.data.makeDataSet1Dplain('gate', x, xunit='mV', yname='current', y=y )
    
    r = fitOhmic(ds, fig=300, gainx=1e-6, gainy=1)

if __name__ == '__main__':
    test_fitohmic()