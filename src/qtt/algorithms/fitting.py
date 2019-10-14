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
        y_data_linearized (1D array): The dependent data with linear estimate subtracted.

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

    split_offset = int(np.floor(x_data.size / 10))
    mean_right = np.mean(y_data_linearized[(center_index + split_offset):])
    mean_left = np.mean(y_data_linearized[:(center_index - split_offset)])
    amplitude_step = -(mean_right - mean_left)

    if np.sign(-y_derivative_filtered[estimated_index]) != np.sign(amplitude_step):
        warnings.warn('step size might be incorrect')

    if fig is not None:
        _plot_fermi_model_estimate(x_data, y_data_linearized, estimated_center_xdata,
                                   amplitude_step, estimated_index, fig=fig)

    return estimated_center_xdata, amplitude_step


def _plot_fermi_model_estimate(x_data, y_data_linearized, estimated_center_xdata, amplitude_step, estimated_index, fig):
    T = np.std(x_data) / 100
    fermi_parameters = [estimated_center_xdata, amplitude_step, T]

    plt.figure(fig)
    plt.clf()
    plt.plot(x_data, y_data_linearized, '.b', label='y_data_linearized')
    plt.plot(x_data, Fermi(x_data, *fermi_parameters), '-c', label='estimated model')
    plt.plot(x_data[estimated_index], y_data_linearized[estimated_index], '.g', label='max slope')
    vline = plt.axvline(estimated_center_xdata, label='estimated_center_xdata')
    vline.set_color('c')
    vline.set_alpha(.5)
    plt.legend()


def initFermiLinear(x_data, y_data, fig=None):
    """ Initialization of fitting a FermiLinear function.

    First the linear part is estimated, then the Fermi part of the function.

    Args:
        x_data (array): data for independent variable
        y_data (array): dependent variable
        fig (int) : figure number

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
        # TODO: When nx < 4 and fig is not None -> yr is undefined
        plt.plot(xdata, yr, '.b', label='Fermi part')
        fermi_part_values = Fermi(xdata, cc, A, T)
        plt.plot(xdata, fermi_part_values, '-m', label='initial estimate')
        plt.legend()
    return linear_part, fermi_part


# %%

def fitFermiLinear(x_data, y_data, verbose=0, fig=None, lever_arm = 1.16, l=None, use_lmfit=False):
    """ Fit data to a Fermi-Linear function

    Args:
        x_data (array): independent variable data
        y_data (array): dependent variable data
        verbose (int) : verbosity (0 == silent). Not used
        fig (int) : figure number
        lever_arm (float): leverarm passed to FermiLinear function
        use_lmfit (bool): If True use lmfit for optimization, otherwise use scipy

    Returns:
        p (array): fitted function parameters
        results (dict): extra fitting data

    .. seealso:: FermiLinear
    """
    xdata = np.array(x_data)
    ydata = np.array(y_data)

    if l is not None:
        warnings.warn('use argument lever_arm instead of l')
        lever_arm = l
        
    # initial values
    linear_part, fermi_part = initFermiLinear(xdata, ydata, fig=None)
    initial_parameters = linear_part + fermi_part

    # fit
    def fermi_linear_fitting_function(xdata, a, b, cc, A, T):
        return FermiLinear(xdata, a, b, cc, A, T, l=lever_arm)

    if use_lmfit:
        import lmfit

        gmodel = lmfit.Model(fermi_linear_fitting_function)
        param_init = dict(zip(gmodel.param_names, initial_parameters))
        gmodel.set_param_hint('T', min=0)

        params = gmodel.make_params(**param_init)
        lmfit_results = gmodel.fit(y_data, params, xdata=x_data)
        fitting_results = lmfit_results.fit_report()
        fitted_parameters = np.array([lmfit_results.best_values[p] for p in gmodel.param_names])
    else:
        fitting_results = scipy.optimize.curve_fit(fermi_linear_fitting_function, xdata, ydata, p0=initial_parameters)
        fitted_parameters = fitting_results[0]

    results = dict({'fitted_parameters': fitted_parameters, 'pp': fitting_results,
                                    'centre': fitted_parameters[2], 'initial_parameters': initial_parameters, 'lever_arm': lever_arm,
                                    'fitting_results': fitting_results})

    if fig is not None:
        plot_FermiLinear(xdata, ydata, results, fig=fig)

    return fitted_parameters, results


def plot_FermiLinear(x_data, y_data, results, fig=10):
        """ Plot results for fitFermiLinear 

        Args:
            x_data (np.array): Independant variable
            y_data (np.array): Dependant variable
            results (dict): Output of fitFermiLinear
            
        """
        fitted_parameters = results['fitted_parameters']
        lever_arm = results['lever_arm']
        y = FermiLinear(x_data, *list(fitted_parameters), lever_arm = lever_arm)

        plt.figure(fig)
        plt.clf()
        plt.plot(x_data, y_data, '.b', label='data')
        plt.plot(x_data, y, 'm-', label='fitted FermiLinear')
        plt.legend(numpoints=1)

# %%

def fit_addition_line_array(x_data, y_data, trimborder=True):
    """ Fits a FermiLinear function to the addition line and finds the middle of the step.

    Note: Similar to fit_addition_line but directly works with arrays of data.

    Args:
        x_data (array): independent variable data
        y_data (array): dependent variable data
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
