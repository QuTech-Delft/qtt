""" Mathematical functions and models """

import numpy as np
import scipy
import scipy.constants
import qtt.pgeometry


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


def fit_gaussian(x_data, y_data, maxiter=None, maxfun=5000, verbose=1, initial_params=None):
    """ Fitting of a gaussian, see function 'gaussian' for the model that is fitted

    Args:
        x_data (array): x values of the data
        y_data (array): y values of the data
        maxiter (int): maximum number of iterations to perform
        maxfun (int): maximum number of function evaluations to make
        verbose (int): set to >0 to print convergence messages
        initial_params (None or array): optional, initial guess for the fit parameters: 
            [mean, s, amplitude, offset]

    Returns:
        par_fit (array): fit parameters of the gaussian: [mean, s, amplitude, offset]
        result_dict (dict): result dictonary containging the fitparameters and the initial guess parameters
    """
    def func(params): return _cost_gaussian(x_data, y_data, params)
    maxsignal = np.percentile(x_data, 98)
    minsignal = np.percentile(x_data, 2)
    if initial_params is None:
        amplitude = np.max(y_data)
        s = (maxsignal - minsignal) * 1 / 20
        mean = x_data[int(np.where(y_data == np.max(y_data))[0][0])]
        offset = np.min(y_data)
        initial_params = np.array([mean, s, amplitude, offset])
    par_fit = scipy.optimize.fmin(func, initial_params, maxiter=maxiter, maxfun=maxfun, disp=verbose >= 2)

    result_dict = {'parameters fitted gaussian': par_fit, 'parameters initial guess': initial_params}

    return par_fit, result_dict


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
        result_dict (dict): dictionary with results of the fit. Fields in the dictionary:
            parameters initial guess (array): initial guess for the fit parameters, either the ones give to the
            function, or generated by the function: [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]
            separation (float): separation between the max of the two gaussians measured in the sum of the std
            split (float): value that seperates the up and the down level

    """
    def func(params): return _cost_double_gaussian(x_data, y_data, params)
    if initial_params is None:
        initial_params = _estimate_double_gaussian_parameters(x_data, y_data)
    par_fit = scipy.optimize.fmin(func, initial_params, maxiter=maxiter, maxfun=maxfun, disp=verbose >= 2)

    if par_fit[4] > par_fit[5]:
        par_fit = np.take(par_fit, [1, 0, 3, 2, 5, 4])
    # separation is the difference between the max of the gaussians devided by the sum of the std of both gaussians
    separation = (par_fit[5] - par_fit[4]) / (abs(par_fit[2]) + abs(par_fit[3]))
    # split equal distant to both peaks measured in std from the peak
    split = par_fit[4] + separation * abs(par_fit[2])

    result_dict = {'parameters initial guess': initial_params, 'separation': separation, 'split': split}
    result_dict['parameters'] = par_fit
    result_dict['left'] = np.take(par_fit, [4, 2, 0])
    result_dict['right'] = np.take(par_fit, [5, 3, 1])
    result_dict['type'] = 'fitted double gaussian'

    return par_fit, result_dict


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
        par_fit (array): fit parameters of the exponential decay, [A, B, gamma]

    See: :func:`exp_function`

    """

    if initial_params is None:
        maxsignal = np.percentile(y_data, 98)
        minsignal = np.percentile(y_data, 2)
        gamma = 1 / (x_data[int(len(x_data) / 2)])
        B = maxsignal
        if offset_parameter is None:
            A = minsignal
            initial_params = np.array([A, B, gamma])

        else:
            initial_params = np.array([B, gamma])

    if offset_parameter is None:
        def func(params): return cost_exp_decay(x_data, y_data, params, threshold)
    else:
        def func(params): return cost_exp_decay(x_data, y_data, np.hstack((offset_parameter, params)), threshold)

    par_fit = scipy.optimize.fmin(func, initial_params, maxiter=maxiter, maxfun=maxfun, disp=verbose >= 2)
    if offset_parameter is not None:
        par_fit = np.hstack(([offset_parameter], par_fit))

    return par_fit


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
    gauss_ramsey = A * np.exp(-(x_data / t2s)**2) * np.sin(2 * np.pi * ramseyfreq * x_data - angle) + B
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
    # cost = np.sum([(np.array(y_data)[1:] - np.array(model)[1:])**2*(np.array(x_data)[1:] -
    #                                                                 np.array(x_data)[:-1])**weight_power])
    cost = np.sum([(np.array(y_data)[1:] - np.array(model)[1:])**2 * (np.diff(x_data))**weight_power])
    return cost


def fit_gauss_ramsey(x_data, y_data, weight_power=0, maxiter=None, maxfun=5000, verbose=1, initial_params=None):
    """ Fit a gauss_ramsey. The function gauss_ramsey gives a model for the measurement result of a pulse Ramsey
    sequence while varying the free evolution time, the phase of the second pulse is made dependent on the free
    evolution time.
    This results in a gaussian decay multiplied by a sinus. Function as used by T.F. Watson et all.,
    see function 'gauss_ramsey' and example in qtt/docs/notebooks/example_fit_ramsey.ipynb

    Args:
        x_data (array): the data for the independant variable
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
        A = (np.max(y_data) - np.min(y_data)) / 2
        t2s = 1e-6
        ramseyfreq = 1 / (1e-6)
        angle = 0
        B = (np.min(y_data) + (np.max(y_data) - np.min(y_data)) / 2)
        initial_params = np.array([A, t2s, ramseyfreq, angle, B])

    par_fit = scipy.optimize.fmin(func, initial_params, maxiter=maxiter, maxfun=maxfun, disp=verbose >= 2)

    result_dict = {
        'description': 'Function to analyse the results of a Ramsey experiment, fitted function: gauss_ramsey = '
                       'A * exp(-(x_data/t2s)**2) * sin(2pi*ramseyfreq * x_data - angle) +B',
        'parameters fit': par_fit,
        'parameters initial guess': initial_params}

    return par_fit, result_dict


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

    Args:
        x (array): TODO
        x0 (float): TODO
        alpha (float): TODO

    Example:
        y = logistic(0, 1, alpha=1)
    """
    f = 1 / (1 + np.exp(-2 * alpha * (x - x0)))
    return f
