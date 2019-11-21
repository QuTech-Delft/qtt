""" Mathematical functions and models """

import warnings
import operator
import numpy as np
import scipy
import scipy.constants
import matplotlib.pyplot as plt
from lmfit import Model

import qtt.pgeometry
from qtt.utilities.visualization import plot_vertical_line
import qtt.algorithms.fitting.extract_lmfit_parameters


def gaussian(x, mean, std, amplitude=1, offset=0):
    """ Model for Gaussian function

       $$y = offset + amplitude * np.exp(-(1/2)*(x-mean)^2/s^2)$$

    Args:
        x (array): data points
        mean, std, amplitude, offset: parameters
    Returns:
        y (array)

    """
    y = offset + amplitude * np.exp(- (x - mean) * (x - mean) / (2 * std * std))
    return y


def _cost_gaussian(x_data, y_data, params):
    """Cost function for fitting a gaussian

    Args:
        x_data (array): x values of the data
        y_data (array): y values of the data
        params (array): parameters of a gaussian, [mean, s, amplitude, offset]
    Returns:
        cost (float): value which indicates the difference between the data and the fit
    """

    [mean, std, amplitude, offset] = params
    model_y_data = gaussian(x_data, mean, std, amplitude, offset)
    cost = np.linalg.norm(y_data - model_y_data)
    return cost


def _estimate_initial_parameters_gaussian(x_data, y_data, include_offset):
    maxsignal = np.percentile(x_data, 98)
    minsignal = np.percentile(x_data, 2)
    amplitude = np.max(y_data) - np.min(y_data)
    s = (maxsignal - minsignal) * 1 / 20
    mean = x_data[int(np.where(y_data == np.max(y_data))[0][0])]
    offset = np.min(y_data)
    if include_offset:
        initial_parameters = np.array([mean, s, amplitude, offset])
    else:
        initial_parameters = np.array([mean, s, amplitude])
    return initial_parameters


def fit_gaussian(x_data, y_data, maxiter=None, maxfun=None, verbose=0, initial_parameters=None, initial_params=None, estimate_offset=True):
    """ Fitting of a gaussian, see function 'gaussian' for the model that is fitted

    Args:
        x_data (array): x values of the data
        y_data (array): y values of the data
        verbose (int): set positive for verbose fit
        initial_parameters (None or array): optional, initial guess for the
            fit parameters: [mean, s, amplitude, offset]
        estimate_offset (bool): If True then include offset in the Gaussian parameters

        maxiter (int): Legacy argument, not used
        maxfun (int): Legacy argument, not used

    Returns:
        par_fit (array): fit parameters of the gaussian: [mean, s, amplitude, offset]
        result_dict (dict): result dictonary containging the fitparameters and the initial guess parameters
    """

    if initial_params is not None:
        warnings.warn('use initial_parameters instead of initial_params')
        initial_parameters = initial_params
    if maxiter is not None:
        warnings.warn('argument maxiter is not used any more')
    if maxfun is not None:
        warnings.warn('argument maxfun is not used any more')

    if initial_parameters is None:
        initial_parameters = _estimate_initial_parameters_gaussian(x_data, y_data, include_offset=estimate_offset)

    if estimate_offset:
        def gaussian_model(x, mean, sigma, amplitude, offset):
            """ Gaussian helper function for lmfit """
            y = gaussian(x, mean, sigma, amplitude, offset)
            return y
    else:
        def gaussian_model(x, mean, sigma, amplitude):
            """ Gaussian helper function for lmfit """
            y = gaussian(x, mean, sigma, amplitude)
            return y

    lmfit_model = Model(gaussian_model)
    lmfit_model.set_param_hint('amplitude', min=2)
    lmfit_result = lmfit_model.fit(
        y_data, x=x_data, **dict(zip(lmfit_model.param_names, initial_parameters)), verbose=verbose)
    result_dict = extract_lmfit_parameters(lmfit_model, lmfit_result)

    result_dict['parameters fitted gaussian'] = result_dict['fitted_parameters']
    result_dict['parameters initial guess'] = result_dict['initial_parameters']

    return result_dict['fitted_parameters'], result_dict


def double_gaussian(x_data, params):
    """ A model for the sum of two Gaussian distributions.

    Args:
        x_data (array): x values of the data
        params (array): parameters of the two gaussians, [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]
            amplitude of first (second) gaussian = A_dn (A_up)
            standard deviation of first (second) gaussian = sigma_dn (sigma_up)
            average value of the first (second) gaussian = mean_dn (mean_up)

    Returns:
        double_gauss (np.array): model of a double gaussian
    """
    [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up] = params
    gauss_dn = gaussian(x_data, mean_dn, sigma_dn, A_dn)
    gauss_up = gaussian(x_data, mean_up, sigma_up, A_up)
    double_gauss = gauss_dn + gauss_up
    return double_gauss


def _cost_double_gaussian(x_data, y_data, params):
    """ Cost function for fitting of double Gaussian.

    Args:
        x_data (array): x values of the data
        y_data (array): y values of the data
        params (array): parameters of the two gaussians, [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]
            amplitude of first (second) gaussian = A_dn (A_up)
            standard deviation of first (second) gaussian = sigma_dn (sigma_up)
            average value of the first (second) gaussian = mean_dn (mean_up)

    Returns:
        cost (float): value which indicates the difference between the data and the fit
    """
    model = double_gaussian(x_data, params)
    cost = np.linalg.norm(y_data - model)
    return cost


def _integral(x_data, y_data):
    """ Calculate integral of function """
    d_xdata = np.diff(x_data)
    d_xdata = np.hstack([d_xdata, d_xdata[-1]])
    data_integral = np.sum(d_xdata * y_data)
    return data_integral


def _estimate_double_gaussian_parameters(x_data, y_data, fast_estimate=False):
    """ Estimate of double gaussian model parameters."""
    maxsignal = np.percentile(x_data, 98)
    minsignal = np.percentile(x_data, 2)

    data_left = y_data[:int((len(y_data) / 2))]
    data_right = y_data[int((len(y_data) / 2)):]

    amplitude_left = np.max(data_left)
    amplitude_right = np.max(data_right)
    sigma_left = (maxsignal - minsignal) * 1 / 20
    sigma_right = (maxsignal - minsignal) * 1 / 20

    if fast_estimate:
        alpha = .1
        mean_left = minsignal + (alpha) * (maxsignal - minsignal)
        mean_right = minsignal + (1 - alpha) * (maxsignal - minsignal)
    else:
        x_data_left = x_data[:int((len(y_data) / 2))]
        x_data_right = x_data[int((len(y_data) / 2)):]

        data_integral_left = _integral(x_data_left, data_left)
        data_integral_right = _integral(x_data_right, data_right)

        sigma_left = data_integral_left / (np.sqrt(2 * np.pi) * amplitude_left)
        sigma_right = data_integral_right / (np.sqrt(2 * np.pi) * amplitude_right)

        mean_left = np.sum(x_data_left * data_left) / np.sum(data_left)
        mean_right = np.sum(x_data_right * data_right) / np.sum(data_right)
    initial_params = np.array([amplitude_left, amplitude_right, sigma_left, sigma_right, mean_left, mean_right])
    return initial_params


def fit_double_gaussian(x_data, y_data, maxiter=None, maxfun=5000, verbose=1, initial_params=None):
    """ Fitting of double gaussian

    Fitting the Gaussians and finding the split between the up and the down state,
    separation between the max of the two gaussians measured in the sum of the std.

    Args:
        x_data (array): x values of the data
        y_data (array): y values of the data
        maxiter (int): maximum number of iterations to perform
        maxfun (int): maximum number of function evaluations to make
        verbose (int): set to >0 to print convergence messages
        initial_params (None or array): optional, initial guess for the fit parameters:
            [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]

    Returns:
        par_fit (array): fit parameters of the double gaussian: [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]
        result_dict (dict): dictionary with results of the fit. Fields guaranteed in the dictionary:
            parameters (array): Fitted parameters
            parameters initial guess (array): initial guess for the fit parameters, either the ones give to the
                 function, or generated by the function: [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]
            reduced_chi_squared (float): Reduced chi squared value of the fit
            separation (float): separation between the max of the two gaussians measured in the sum of the std
            split (float): value that separates the up and the down level
            left (array), right (array): Parameters of the left and right fitted Gaussian

    """

    if initial_params is None:
        initial_params = _estimate_double_gaussian_parameters(x_data, y_data)

    def _double_gaussian(x, A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up):
        """ Double Gaussian helper function for lmfit """
        gauss_dn = gaussian(x, mean_dn, sigma_dn, A_dn)
        gauss_up = gaussian(x, mean_up, sigma_up, A_up)
        double_gauss = gauss_dn + gauss_up
        return double_gauss

    double_gaussian_model = Model(_double_gaussian)
    delta_x = x_data.max() - x_data.min()
    bounds = [x_data.min() - .1 * delta_x, x_data.max() + .1 * delta_x]
    double_gaussian_model.set_param_hint('mean_up', min=bounds[0], max=bounds[1])
    double_gaussian_model.set_param_hint('mean_dn', min=bounds[0], max=bounds[1])
    double_gaussian_model.set_param_hint('A_up', min=0)
    double_gaussian_model.set_param_hint('A_dn', min=0)

    param_names = double_gaussian_model.param_names
    result = double_gaussian_model.fit(y_data, x=x_data, **dict(zip(param_names, initial_params)), verbose=0)

    par_fit = np.array([result.best_values[p] for p in param_names])

    if par_fit[4] > par_fit[5]:
        par_fit = np.take(par_fit, [1, 0, 3, 2, 5, 4])
    # separation is the difference between the max of the gaussians divided by the sum of the std of both gaussians
    separation = (par_fit[5] - par_fit[4]) / (abs(par_fit[2]) + abs(par_fit[3]))
    # split equal distant to both peaks measured in std from the peak
    weigthed_distance_split = par_fit[4] + separation * abs(par_fit[2])

    result_dict = {'parameters': par_fit, 'parameters initial guess': initial_params, 'separation': separation,
                   'split': weigthed_distance_split, 'reduced_chi_squared': result.redchi,
                   'left': np.take(par_fit, [4, 2, 0]), 'right': np.take(par_fit, [5, 3, 1]),
                   'type': 'fitted double gaussian'}

    return par_fit, result_dict


def _double_gaussian_parameters(gauss_left, gauss_right):
    return np.vstack((gauss_left[::-1], gauss_right[::-1])).T.flatten()


def refit_double_gaussian(result_dict, x_data, y_data, gaussian_amplitude_ratio_threshold=8):
    """ Improve fit of double Gaussian by estimating the initial parameters based on an existing fit

    Args:
        result_dict(dict): Result dictionary from fit_double_gaussian
        x_data (array): Independent data
        y_data (array): Signal data
        gaussian_amplitude_ratio_threshold (float): If ratio between amplitudes of Gaussian peaks is larger than
                            this fit, re-estimate
    Returns:
        Dictionary with improved fitting results
    """

    mean = operator.itemgetter(0)
    std = operator.itemgetter(1)
    amplitude = operator.itemgetter(2)

    if amplitude(result_dict['left']) > amplitude(result_dict['right']):
        large_gaussian_parameters = result_dict['left']
        small_gaussian_parameters = result_dict['right']
    else:
        large_gaussian_parameters = result_dict['right']
        small_gaussian_parameters = result_dict['left']
    gaussian_ratio = amplitude(large_gaussian_parameters) / amplitude(small_gaussian_parameters)

    if gaussian_ratio > gaussian_amplitude_ratio_threshold:
        # re-estimate by fitting a single gaussian to the data remaining after removing the main gaussian
        y_residual = y_data - gaussian(x_data, *large_gaussian_parameters)
        idx = np.logical_and(x_data > mean(large_gaussian_parameters) - 1.5 * std(large_gaussian_parameters),
                             x_data < mean(large_gaussian_parameters) + 1.5 * std(large_gaussian_parameters))
        y_residual[idx] = 0
        gauss_fit, _ = fit_gaussian(x_data, y_residual)

        initial_parameters = _double_gaussian_parameters(large_gaussian_parameters, gauss_fit[:3])
        _, result_dict_refit = fit_double_gaussian(x_data, y_data, initial_params=initial_parameters)
        if result_dict_refit['reduced_chi_squared'] < result_dict['reduced_chi_squared']:
            result_dict = result_dict_refit
    return result_dict


def exp_function(x, a, b, c):
    """ Model for exponential function

       $$y = a + b * np.exp(-c * x)$$

    Args:
        x (array): x values of the data
        a = offset
        b = starting value
        c = 1/typical decay time
    Returns:
        y (array): model for exponantial decay

    """
    y = a + b * np.exp(-c * x)
    return y


def cost_exp_decay(x_data, y_data, params, threshold=None):
    """ Cost function for exponential decay.

    Args:
        x_data (array): the data for the input variable
        y_data (array): the data for the measured variable
        params (array): parameters of the exponential decay function, [A,B, gamma]
        threshold (float or None or 'auto'): if the difference between data and model is larger then the threshold,
                                             then the cost penalty is reduced.
            If None use normal cost function. If 'auto' use automatic detection (at 95th percentile)
    Returns:
        cost (float): value which indicates the difference between the data and the fit
    """
    model = exp_function(x_data, *params)
    cost = qtt.pgeometry.robustCost(y_data - model, thr=threshold)
    cost = np.linalg.norm(cost)
    return cost


def _estimate_exp_decay_initial_parameters(x_data, y_data, offset_parameter):
    """ Estimate parameters for exponential decay function

    Args:
        x_data (array): Independent data
        y_data (array): Dependent data
        offset_parameter (None or float): If None, then estimate the offset, otherwise fix the offset to the
        specified value
    Returns:
        Array with initial parameters
    """
    maxsignal = np.percentile(y_data, 98)
    minsignal = np.percentile(y_data, 2)
    midpoint = int(len(x_data) / 2)
    gamma = 1 / (x_data[midpoint] - x_data[0])

    mean_left = np.mean(y_data[:midpoint])
    mean_right = np.mean(y_data[midpoint:])
    increasing_exponential = mean_left < mean_right
    alpha = np.exp(gamma * x_data[0])

    if offset_parameter is None:
        if increasing_exponential:
            A = maxsignal
            B = -(maxsignal - minsignal) * alpha
        else:
            A = minsignal
            B = (maxsignal - minsignal) * alpha
        initial_params = np.array([A, B, gamma])
    else:
        if increasing_exponential:
            B = -(offset_parameter - minsignal) * alpha
        else:
            B = (maxsignal - offset_parameter) * alpha
        initial_params = np.array([B, gamma])
    return initial_params


def fit_exp_decay(x_data, y_data, maxiter=None, maxfun=5000, verbose=1, initial_params=None, threshold=None,
                  offset_parameter=None):
    """ Fit a exponential decay.

    Args:
        x_data (array): the data for the input variable
        y_data (array): the data for the measured variable
        maxiter (int): maximum number of iterations to perform
        maxfun (int): maximum number of function evaluations to make
        verbose (int): set to >0 to print convergence messages
        initial_params (None or array): optional, initial guess for the fit parameters: [A,B, gamma]
        threshold (float or None): threshold for the cost function.
            If the difference between data and model is larger then the threshold, these data are not taken into
            account for the fit.
            If None use automatic detection (at 95th percentile)
        offset_parameter (None or float): if None, then estimate the offset, otherwise fix the offset to the
            specified value
    Returns:
        fitted_parameters (array): fit parameters of the exponential decay, [A, B, gamma]

    See: :func:`exp_function`

    """

    if initial_params is None:
        initial_params = _estimate_exp_decay_initial_parameters(x_data, y_data, offset_parameter)

    if offset_parameter is None:
        def cost_function(params):
            return cost_exp_decay(x_data, y_data, params, threshold)
    else:
        def cost_function(params):
            return cost_exp_decay(
                x_data, y_data, np.hstack((offset_parameter, params)), threshold)

    fitted_parameters = scipy.optimize.fmin(cost_function, initial_params,
                                            maxiter=maxiter, maxfun=maxfun, disp=verbose >= 2)
    if offset_parameter is not None:
        fitted_parameters = np.hstack(([offset_parameter], fitted_parameters))

    return fitted_parameters


def gauss_ramsey(x_data, params):
    """ Model for the measurement result of a pulse Ramsey sequence while varying the free evolution time, the phase
    of the second pulse is made dependent on the free evolution time. This results in a gaussian decay multiplied
    by a sinus.
    Function as used by T.F. Watson et all., example in qtt/docs/notebooks/example_fit_ramsey.ipynb

    $$ gauss_ramsey = A * exp(-(x_data/t2s)**2) * sin(2pi*ramseyfreq * x_data - angle) +B  $$

    Args:
        x_data (array): the data for the input variable
        params (array): parameters of the gauss_ramsey function, [A,t2s,ramseyfreq,angle,B]

    Result:
        gauss_ramsey (array): model for the gauss_ramsey
    """
    [A, t2s, ramseyfreq, angle, B] = params
    gauss_ramsey = A * np.exp(-(x_data / t2s) ** 2) * np.sin(2 * np.pi * ramseyfreq * x_data - angle) + B
    return gauss_ramsey


def cost_gauss_ramsey(x_data, y_data, params, weight_power=0):
    """ Cost function for gauss_ramsey.

    Args:
        x_data (array): the data for the input variable
        y_data (array): the data for the measured variable
        params (array): parameters of the gauss_ramsey function, [A,C,ramseyfreq,angle,B]
        weight_power (float)

    Returns:
        cost (float): value which indicates the difference between the data and the fit
    """
    model = gauss_ramsey(x_data, params)
    cost = np.sum([(np.array(y_data)[1:] - np.array(model)[1:]) ** 2 * (np.diff(x_data)) ** weight_power])
    return cost


def estimate_dominant_frequency(signal, sample_rate=1, remove_dc=True, fig=None):
    """ Estimate dominant frequency in a signal

    Args:
        signal (array): Input data
        sample_rate (float): Sample rate of the data
        remove_dc (bool): If True, then do not estimate the DC component
        fig (int or None): Optionally plot the estimated frequency
    """
    w = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1. / sample_rate)

    if remove_dc:
        w[0] = 0

    ff = freqs[np.argmax(np.abs(w))]

    if fig:
        plt.figure(fig)
        plt.clf()
        plt.plot(freqs, np.abs(w), '.b')
        plt.xlabel('Frequency')
        plt.ylabel('Abs of fft')
        plot_vertical_line(ff)
    return ff

# %%
#


def estimate_parameters_damped_sine_wave(x_data, y_data, exponent=2):
    """ Estimate initial parameters of a damped sine wave

    Also see: https://en.wikipedia.org/wiki/Damped_sine_wave

    Args:
        exponent: Exponent from the exponential decay factor

    Returns:
        Estimated parameters for gauss_ramsey method
    """
    A = (np.max(y_data) - np.min(y_data)) / 2
    B = (np.min(y_data) + A)

    n = int(x_data.size / 2)
    mean_left = np.mean(np.abs(y_data[:n] - B))
    mean_right = np.mean(np.abs(y_data[n:] - B))

    laplace_factor = 1e-16
    decay_factor = (mean_left + laplace_factor) / (mean_right + laplace_factor)

    duration = x_data[-1] - x_data[0]
    sample_rate = x_data.size / duration
    ramseyfreq = estimate_dominant_frequency(y_data, sample_rate=sample_rate)
    if A == 0:
        angle = 0
    else:
        n_start = max(min((y_data[0] - B) / A, 1), -1)
        angle = -np.arcsin(n_start)
    t2s = 2 * duration / decay_factor

    initial_params = np.array([A, t2s, ramseyfreq, angle, B])
    return initial_params


def fit_gauss_ramsey(x_data, y_data, weight_power=0, maxiter=None, maxfun=5000, verbose=1, initial_params=None):
    """ Fit a gauss_ramsey. The function gauss_ramsey gives a model for the measurement result of a pulse Ramsey
    sequence while varying the free evolution time, the phase of the second pulse is made dependent on the free
    evolution time.
    This results in a gaussian decay multiplied by a sinus. Function as used by T.F. Watson et all.,
    see function 'gauss_ramsey' and example in qtt/docs/notebooks/example_fit_ramsey.ipynb

    Args:
        x_data (array): the data for the independent variable
        y_data (array): the data for the measured variable
        weight_power (float)
        maxiter (int): maximum number of iterations to perform
        maxfun (int): maximum number of function evaluations to make
        verbose (int): set to >0 to print convergence messages
        initial_params (None or array): optional, initial guess for the fit parameters: [A,C,ramseyfreq,angle,B]

    Returns:
        par_fit (array): array with the fit parameters: [A,t2s,ramseyfreq,angle,B]
        result_dict (dict): dictionary containing a description, the par_fit and initial_params

    """

    def func(params): return cost_gauss_ramsey(x_data, y_data, params, weight_power=weight_power)

    if initial_params is None:
        initial_params = estimate_parameters_damped_sine_wave(x_data, y_data, exponent=2)

    fit_parameters = scipy.optimize.fmin(func, initial_params, maxiter=maxiter, maxfun=maxfun, disp=verbose >= 2)

    result_dict = {
        'description': 'Function to analyse the results of a Ramsey experiment, fitted function: gauss_ramsey = '
                       'A * exp(-(x_data/t2s)**2) * sin(2pi*ramseyfreq * x_data - angle) +B',
        'parameters fit': fit_parameters,
        'parameters initial guess': initial_params}

    return fit_parameters, result_dict


def plot_gauss_ramsey_fit(x_data, y_data, fit_parameters, fig):
    """ Plot Gauss Ramset fit

    Args:
        x_data: Input array with time variable
        y_data: Input array with signal
        fit_parameters: Result of fit_gauss_ramsey
    """
    test_x = np.linspace(0, np.max(x_data), 200)
    freq_fit = abs(fit_parameters[2] * 1e-6)
    t2star_fit = fit_parameters[1] * 1e6

    plt.figure(fig)
    plt.clf()
    plt.plot(x_data * 1e6, y_data, 'o', label='Data')
    plt.plot(test_x * 1e6, gauss_ramsey(test_x, fit_parameters), label='Fit')
    plt.title('Gauss Ramsey fit: %.2f MHz / $T_2^*$: %.1f $\mu$s' % (freq_fit, t2star_fit))
    plt.xlabel('time ($\mu$s)')
    plt.ylabel('Spin-up probability')
    plt.legend()


def linear_function(x, a, b):
    """ Linear function with offset"""
    return a * x + b


def Fermi(x, cc, A, T, kb=1):
    r""" Fermi distribution

    Arguments:
        x (numpy array): independent variable
        cc (float): center of Fermi distribution
        A (float): amplitude of Fermi distribution
        T (float): temperature Fermi distribution
        kb (float, default: 1): temperature scaling factor

    Returns:
        y (numpy array): value of the function

    .. math::

        y =  A*(1/ (1+\exp( (x-cc)/(kb*T) ) ) )
    """
    y = A * 1. / (1 + np.exp((x - cc) / (kb * T)))
    return y


def FermiLinear(x, a, b, cc, A, T, l=1.16):
    r""" Fermi distribution with linear function added

    Arguments:
        x (numpy array): independent variable 
        a, b (float): coefficients of linear part
        cc (float): center of Fermi distribution
        A (float): amplitude of Fermi distribution
        T (float): temperature Fermi distribution in Kelvin
        l (float): leverarm divided by kb

    The default value of the leverarm is
        (100 ueV/mV)/kb = (100*1e-6*scipy.constants.eV )/kb = 1.16.

    For this value the input variable x should be in mV, the
    temperature T in K. We input the leverarm divided by kb for numerical stability.

    Returns:
        y (numpy array): value of the function

    .. math::

        y = a*x + b + A*(1/ (1+\exp( l* (x-cc)/(T) ) ) )

    """
    y = a * x + b + A * 1. / (1 + np.exp(l * (x - cc) / (T)))
    return y


def logistic(x, x0=0, alpha=1):
    """ Logistic function

    Defines the logistic function

    .. math::

        y = 1 / (1 + \exp(-2 * alpha * (x - x0)))

    Args:
        x (array): Independent data
        x0 (float): Midpoint of the logistic function
        alpha (float): Growth rate

    Example:
        y = logistic(0, 1, alpha=1)
    """
    f = 1 / (1 + np.exp(-2 * alpha * (x - x0)))
    return f
