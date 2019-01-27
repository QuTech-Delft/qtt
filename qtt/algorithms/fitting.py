""" Fitting of Fermi-Dirac distributions. """

import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy

import qtt.pgeometry
from qcodes import DataArray
from qtt.algorithms.functions import Fermi, FermiLinear, linear_function


def _estimate_fermi_model_center_amplitude(x_data, y_data_linearized, fig=None):
    """ Estimates the following properties of a charge addition line; the center location
        of the addition line. The amplitude step size caused by the addition line.

    Args:
        x_data (1D array): The independent data.
        y_data_linerized (1D array): The dependent data with linear estimate subtracted.

    Returns:
        xdata_center_est (float): Estimate of x-data value at the center.
        amplitude_step (float): Estimate of the amplitude of the step.
    """
    sigma = x_data.size / 250
    y_derivative_filtered = scipy.ndimage.gaussian_filter(y_data_linearized, sigma, order=1)

    # assume step is steeper than overall slope
    estimated_index = np.argmax(np.abs(y_derivative_filtered))
    center_index = int(x_data.size / 2)

    # prevent guess to be at the edges
    if estimated_index < 0.01 * x_data.size or estimated_index > 0.99 * x_data.size:
        estimated_center_xdata = np.mean(x_data)
    else:
        estimated_center_xdata = x_data[estimated_index]

    split_offset = int(np.floor(x_data.size/10))
    mean_right = np.mean(y_data_linearized[(center_index+split_offset):])
    mean_left = np.mean(y_data_linearized[:(center_index-split_offset)])
    step_size = -(mean_right - mean_left)
    
    amplitude_step = step_size
    if np.sign(-y_derivative_filtered[estimated_index]) != np.sign(step_size):
        warnings.warn('step size might be incorrect')
        
    if fig is not None:
        T = np.std(x_data) / 100
        fermi_parameters = [estimated_center_xdata, amplitude_step, T]

        plt.figure(fig); plt.clf()
        plt.plot(x_data, y_data_linearized, '.b', label='y_data_linearized')
        plt.plot(x_data, Fermi(x_data, *fermi_parameters), '-c', label='estimated model')
        plt.plot(x_data[estimated_index], y_data_linearized[estimated_index], '.g', label='max slope')
        plt.legend()
        
        plt.figure(fig+1); plt.clf()
        plt.plot(x_data, y_derivative_filtered, '-r')
        plt.plot(x_data[estimated_index], y_derivative_filtered[estimated_index], '.g', label='max slope')


    return estimated_center_xdata, amplitude_step


    
def initFermiLinear(x_data, y_data, fig=None):
    """ Initialization of fitting a FermiLinear function.

    First the linear part is estimated, then the Fermi part of the function.

    Args:
        x_data (array): data for independent variable
        y_data (array): dependent variable

    Returns:
        linear_part (array)
        fermi_part (array)
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
        linear_part = [a, b]
        ylin = linear_function(xdata, linear_part[0], linear_part[1])
        cc = np.mean(xdata)
        A = 0
        T = np.std(xdata) / 10
        fermi_part = [cc, A, T]
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
        linear_part = [a, b]
        ylin = linear_function(xdata, *linear_part)

        # subtract linear part
        yr = ydata - ylin

        cc, A = _estimate_fermi_model_center_amplitude(xdata, yr)

        T = np.std(xdata) / 100
        linear_part[1] = linear_part[1] - A / 2  # correction
        fermi_part = [cc, A, T]

        yr = ydata - linear_function(xdata, *linear_part)

    if fig is not None:
        yf = FermiLinear(xdata, *linear_part, *fermi_part)

        xx = np.hstack((xdata[0:nx], xdata[-nx:]))
        yy = np.hstack((ydata[0:nx], ydata[-nx:]))
        plt.figure(fig)
        plt.clf()
        plt.plot(xdata, ydata, '.b', label='raw data')
        plt.plot(xx, yy, 'ok')
        qtt.pgeometry.plot2Dline([-1, 0, cc], ':c', label='center')
        plt.plot(xdata, ylin, '-c', alpha=.5, label='fitted linear function')
        plt.plot(xdata, yf, '-m', label='fitted FermiLinear function')

        plt.title('initFermiLinear', fontsize=12)
        plt.legend(numpoints=1)

        plt.figure(fig + 1)
        plt.clf()
        plt.plot(xdata, yr, '.b', label='Fermi part')
        fermi_part_values = Fermi(xdata, cc, A, T)
        plt.plot(xdata, fermi_part_values, '-m', label='initial estimate')

        plt.legend()

        plt.legend()
    return linear_part, fermi_part


# %%

def fitFermiLinear(x_data, y_data, verbose=1, fig=None, l=1.16, use_lmfit=0):
    """ Fit data to a Fermi-Linear function

    Args:
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
    linear_part, fermi_part = initFermiLinear(xdata, ydata, fig=None)
    initial_parameters = linear_part + fermi_part

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

def fit_addition_line_array(x_data, y_data, trimborder=True):
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
        cut_index = max(min(int(x_data.size / 40), 100), 1)
        x_data = x_data[cut_index: -cut_index]
        y_data = y_data[cut_index: -cut_index]

    # fitting of the FermiLinear function
    fit_parameters, extra_data = fitFermiLinear(x_data, y_data, verbose=1, fig=None)
    initial_parameters = extra_data['p0']
    m_addition_line = fit_parameters[2]

    return m_addition_line, {'fit parameters': fit_parameters, 'parameters initial guess': initial_parameters}


def fit_addition_line(dataset, trimborder=True):
    """Fits a FermiLinear function to the addition line and finds the middle of the step.

    Args:
        dataset (qcodes dataset): The 1d measured data of addition line.
        trimborder (bool): determines if the edges of the data are taken into account for the fit.

    Returns:
        m_addition_line (float): x value of the middle of the addition line
        result_dict (dict): dictionary with the following results
            fit parameters (array): fit parameters of the Fermi Linear function
            parameters initial guess (array): parameters of initial guess
            dataset fit (qcodes dataset): dataset with fitted Fermi Linear function
            dataset initial guess (qcodes dataset):dataset with guessed Fermi Linear function

    See also: FermiLinear and fitFermiLinear
    """
    y_array = dataset.default_parameter_array()
    setarray = y_array.set_arrays[0]
    x_data = np.array(setarray)
    y_data = np.array(y_array)

    if trimborder:
        cut_index = max(min(int(x_data.size / 40), 100), 1)
        x_data = x_data[cut_index: -cut_index]
        y_data = y_data[cut_index: -cut_index]
        setarray = setarray[cut_index: -cut_index]

    m_addition_line, result_dict = fit_addition_line_array(x_data, y_data, trimborder=False)

    y_initial_guess = FermiLinear(x_data, *list(result_dict['parameters initial guess']))
    dataset_guess = DataArray(name='fit', label='fit', preset_data=y_initial_guess, set_arrays=(setarray,))

    y_fit = FermiLinear(x_data, *list(result_dict['fit parameters']))
    dataset_fit = DataArray(name='fit', label='fit', preset_data=y_fit, set_arrays=(setarray,))

    return m_addition_line, {'dataset fit': dataset_fit, 'dataset initial guess': dataset_guess}


def test_fit_fermi_linear(fig=100, verbose=0):
    expected_parameters = [0.01000295, 0.51806569, -4.88800525, 0.12838861, 0.25382811]
    x_data = np.arange(-20, 10, 0.1)
    y_data = FermiLinear(x_data, *expected_parameters)
    y_data += 0.005 * np.random.rand(y_data.size)

    actual_parameters, _ = fitFermiLinear(x_data, y_data, verbose=verbose, fig=fig, use_lmfit=False)
    absolute_difference_parameters = np.abs(actual_parameters - expected_parameters)

    y_data_fitted = FermiLinear(x_data, *actual_parameters)
    max_difference = np.max(np.abs(y_data_fitted - y_data))

    if verbose:
        print('expected: %s' % expected_parameters)
        print('fitted:   %s' % actual_parameters)
        print('temperature: %.2f' % (actual_parameters[-1]))
        print('max diff parameters: %.2f' % (absolute_difference_parameters.max()))
        print('max diff values: %.4f' % (max_difference.max()))

    assert np.all(max_difference < 1.0e-2)
    assert np.all(absolute_difference_parameters < 0.6)

    try:
        import lmfit
        have_lmfit = True
    except ImportError:
        have_lmfit = False

    if have_lmfit:
        actual_parameters, _ = fitFermiLinear(x_data, y_data, verbose=1, fig=fig, use_lmfit=True)
        absolute_difference_parameters = np.abs(actual_parameters - expected_parameters)
        assert np.all(absolute_difference_parameters < 0.1)



    # test with data from F006
    x_data = np.array([-20. , -19.9, -19.8, -19.7, -19.6, -19.5, -19.4, -19.3, -19.2,
       -19.1, -19. , -18.9, -18.8, -18.7, -18.6, -18.5, -18.4, -18.3,
       -18.2, -18.1, -18. , -17.9, -17.8, -17.7, -17.6, -17.5, -17.4,
       -17.3, -17.2, -17.1, -17. , -16.9, -16.8, -16.7, -16.6, -16.5,
       -16.4, -16.3, -16.2, -16.1, -16. , -15.9, -15.8, -15.7, -15.6,
       -15.5, -15.4, -15.3, -15.2, -15.1, -15. , -14.9, -14.8, -14.7,
       -14.6, -14.5, -14.4, -14.3, -14.2, -14.1])
    y_data = np.array([0.03055045, 0.0311075 , 0.03098561, 0.03033496, 0.03006341,
       0.03072266, 0.03183486, 0.03170599, 0.03199145, 0.03224666,
       0.03164276, 0.03156053, 0.03133487, 0.03184649, 0.03224385,
       0.03207413, 0.03196082, 0.03229934, 0.03158735, 0.03120681,
       0.03119833, 0.03220412, 0.03185901, 0.03124884, 0.03129008,
       0.0314923 , 0.0315841 , 0.0313667 , 0.03115382, 0.03069049,
       0.03058055, 0.02923863, 0.02789339, 0.02437544, 0.01896179,
       0.01776424, 0.01175409, 0.01074043, 0.00950811, 0.0074723 ,
       0.0060949 , 0.00575982, 0.00501728, 0.00490061, 0.00465821,
       0.00440039, 0.00434098, 0.00429608, 0.00421024, 0.0042945 ,
       0.0042552 , 0.00433429, 0.00440945, 0.00446915, 0.00446351,
       0.00439317, 0.00447768, 0.0044295 , 0.00450926, 0.0045605 ])
    
    figx = fig if fig is None else fig+100
    linear_part, fermi_part = initFermiLinear(x_data, y_data, fig=figx)
    np.testing.assert_almost_equal(linear_part,[0, 0], decimal=1)
    np.testing.assert_almost_equal(fermi_part,[-16.7, 0.02755, 0.01731], decimal=2)
    
if __name__ == '__main__':
    test_fit_fermi_linear(verbose=1)
