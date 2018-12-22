""" Fitting of Fermi-Dirac distributions. """

import matplotlib.pyplot as plt
import numpy as np
import scipy

from qcodes import DataArray
import qtt.pgeometry
from qtt.algorithms.functions import Fermi, FermiLinear, linear_function

def _estimate_fermi_model_center(xdata, yr):
    """ Estimate the location of the center and amplitude of the step of a 
    charge addition line.
    
    Args:
        xdata (1D array): independent data
        yr (1D array): dependent data with linear estimate subtracted
        
    Returns:
        cc (float): estimate of xdata value at the center
        A (float): estimate of the amplitude of the step
    """
    sigma = xdata.size/250
    dyr = scipy.ndimage.gaussian_filter(yr, sigma, order=1)
    
    # assume step is steeper than overall slope
    cc_idx = np.argmax(np.abs(dyr))            
    h = int(xdata.size / 2)
    
    # prevent guess to be at the edges
    if cc_idx < xdata.size/100 or cc_idx > (99/100)*xdata.size:
        cc = np.mean(xdata)
    else:
        cc = xdata[cc_idx]
    
    A = -1*np.sign(dyr[cc_idx])*(np.mean(yr[h:]) - np.mean(yr[:h]))

    return cc, A

def initFermiLinear(x_data, y_data, fig=None):
    """ Initalization of fitting a FermiLinear function.

    First the linear part is estimated, then the Fermi part of the function.
    """
    xdata = np.array(x_data)
    ydata = np.array(y_data)
    n = xdata.size
    nx = int(np.ceil(n / 5))

    if nx < 4:
        p1, _ = scipy.optimize.curve_fit(linear_function, np.array(xdata[0:100]),
                                         np.array(ydata[0:100]))

        a = p1[0]
        b = p1[1]
        ab = [a, b]
        ylin = linear_function(xdata, ab[0], ab[1])
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
        ylin = linear_function(xdata, ab[0], ab[1])

        # subtract linear part
        yr = ydata - ylin

        cc, A = _estimate_fermi_model_center(xdata, yr)
        
        T = np.std(xdata) / 100
        ab[1] = ab[1] - A / 2  # correction
        ff = [cc, A, T]
        
    if fig is not None:
        yf = FermiLinear(xdata, ab[0], ab[1], *ff)

        xx = np.hstack((xdata[0:nx], xdata[-nx:]))
        yy = np.hstack((ydata[0:nx], ydata[-nx:]))
        plt.figure(fig)
        plt.clf()
        plt.plot(xdata, ydata, '.b', label='raw data')
        plt.plot(xx, yy, 'ok')
        qtt.pgeometry.plot2Dline([-1, 0, cc], ':c', label='center')
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
        # plt.plot(xdata, yr, '.b', label='Fermi part')

        plt.legend()
    return ab, ff


# %%

def fitFermiLinear(x_data, y_data, verbose=1, fig=None, l=1.16, use_lmfit=0):
    """ Fit data to a Fermi-Linear function

    Arguments:
        x_data, y_data (array): independent and dependent variable data
        l (float): leverarm passed to FermiLinear function
        use_lmfit (bool): If True use lmfit for optimization, otherwise use scipy
        
    Returns:
        p (array): fitted function parameters
        results (dict): extra fitting data

    .. seealso:: FermiLinear
    """
    xdata = np.array(x_data)
    ydata = np.array(y_data)

    # initial values
    ab, ff = initFermiLinear(xdata, ydata, fig=None)
    initial_parameters = ab + ff

    # fit
    def fermi_linear_fitting_function(xdata, a, b, cc, A, T):
        return FermiLinear(xdata, a, b, cc, A, T, l=l)

    if use_lmfit:
        import lmfit

        gmodel = lmfit.Model(fermi_linear_fitting_function)
        param_init = dict(zip(gmodel.param_names, initial_parameters))
        gmodel.set_param_hint('T', min=0)

        params = gmodel.make_params(**param_init)
        lmfit_results = gmodel.fit(y_data, params, xdata=x_data)
        fitting_results = lmfit_results.fit_report()
        p = np.array([lmfit_results.best_values[p] for p in gmodel.param_names])
    else:
        fitting_results = scipy.optimize.curve_fit(fermi_linear_fitting_function, xdata, ydata, p0=initial_parameters)
        p = fitting_results[0]

    if fig is not None:
        y = FermiLinear(xdata, *list(p))
        plt.figure(fig)
        plt.clf()
        plt.plot(xdata, ydata, '.b', label='data')
        plt.plot(xdata, y, 'm-', label='fitted FermiLinear')
        plt.legend(numpoints=1)
    return p, dict({'pp': fitting_results, 'p0': initial_parameters, 'initial_parameters': initial_parameters, 'fitting_results': fitting_results})


# %%

def fit_add_line_array(xdata, ydata, trimborder=True):
    """ Fits a FermiLinear function to the addition line and finds the middle of the step.
    
    Note: Similar to fit_addition_line but directly works with arrays of data.
    
    Args:
        x_data, y_data (array): independent and dependent variable data
        trimborder (bool): determines if the edges of the data are taken into account for the fit

    Returns:
        m_addition_line (float): x value of the middle of the addition line
        pfit (array): fit parameters of the Fermi Linear function
        pguess (array): parameters of initial guess
    """
    if trimborder:
        ncut = max(min(int(xdata.size / 40), 100), 1)
        xdata, ydata = xdata[ncut: -ncut], ydata[ncut:-ncut]

    # fitting of the FermiLinear function
    pp = fitFermiLinear(xdata, ydata, verbose=1, fig=None)
    pfit = pp[0]
    pguess = pp[1]['p0']
    m_addition_line = pfit[2]
    result_dict = {'fit parameters': pfit, 'parameters initial guess': pguess}
    return m_addition_line, result_dict

def fit_addition_line(dataset, trimborder=True):
    """Fits a FermiLinear function to the addition line and finds the middle of the step.

    Args:
        dataset (qcodes dataset): 1d measured data of additionline
        trimborder (bool): determines if the edges of the data are taken into account for the fit

    Returns:
        m_addition_line (float): x value of the middle of the addition line
        fit parameters (array): fit parameters of the Fermi Linear function
        parameters initial guess (array): parameters of initial guess
        dataset_fit (qcodes dataset): dataset with fitted Fermi Linear function
        dataset_guess (qcodes dataset):dataset with guessed Fermi Linear function

    See also: FermiLinear and fitFermiLinear
    """
    y_array = dataset.default_parameter_array()
    setarray = y_array.set_arrays[0]
    xdata = np.array(setarray)
    ydata = np.array(y_array)

    if trimborder:
        ncut = max(min(int(xdata.size / 40), 100), 1)
        xdata, ydata, setarray = xdata[ncut: -ncut], ydata[ncut:-ncut], setarray[ncut:-ncut]

    m_addition_line, result_dict = fit_add_line_array(xdata, ydata, trimborder=False)
    
    y0 = FermiLinear(xdata, *list(result_dict['parameters initial guess']))
    dataset_guess = DataArray(name='fit', label='fit', preset_data=y0, set_arrays=(setarray,))
    y = FermiLinear(xdata, *list(result_dict['fit parameters']))
    dataset_fit = DataArray(name='fit', label='fit', preset_data=y, set_arrays=(setarray,))
    result_dict['dataset fit'] = dataset_fit
    result_dict['dataset initial guess'] = dataset_guess
    
    return m_addition_line, result_dict


def test_fitfermilinear(fig=None):
    expected_parameters = [0.01000295, 0.51806569, -4.88800525, 0.12838861, 0.25382811]
    x_data = np.arange(-20, 10, 0.1)
    y_data = FermiLinear(x_data, *expected_parameters)
    y_data += 0.005 * np.random.rand(y_data.size)

    actual_parameters, _ = fitFermiLinear(x_data, y_data, verbose=1, fig=fig, use_lmfit=False)
    absolute_difference = np.abs(actual_parameters - expected_parameters)

    if fig:
        print('expected: %s' % expected_parameters)
        print('fitted:   %s' % actual_parameters)
        print('max diff: %.2f' % (absolute_difference.max()))

    assert np.all(absolute_difference < 1e-1)

    try:
        import lmfit
        have_lmfit=1
    except ImportError:
        have_lmfit=0
    if have_lmfit:
        actual_parameters, _ = fitFermiLinear(x_data, y_data, verbose=1, fig=fig, use_lmfit=True)
        absolute_difference = np.abs(actual_parameters - expected_parameters)
        assert np.all(absolute_difference < 1e-1)


if __name__ == '__main__':
    test_fitfermilinear(fig=100)
