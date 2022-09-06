""" Mathematical functions and models """

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.constants
from lmfit import Model
from matplotlib.axes import Axes

import qtt.pgeometry
import qtt.utilities.tools
from qtt.algorithms.generic import subpixelmax
from qtt.utilities.visualization import get_axis, plot_vertical_line


def gaussian(x, mean, std, amplitude=1, offset=0):
    """ Model for Gaussian function

       $$y = offset + amplitude * np.exp(-(1/2)*(x-mean)^2/s^2)$$

    Args:
        x (array): data points
        mean, std, amplitude, offset: parameters
    Returns:
        y (array)

    """
    x0 = (x - mean)
    y = amplitude * np.exp(- x0 * x0 / (2 * std * std))
    if offset:
        return offset + y
    return y


def sine(x: Union[float, np.ndarray], amplitude: float, frequency: float,
         phase: float, offset: float) -> Union[float, np.ndarray]:
    """ Model for sine function

        y = offset + amplitude * np.sin(2 * np.pi * frequency * x + phase)

    Args:
        x : Independent data points
        amplitude, frequency, phase, offset: Arguments for the sine model
    Returns:
        Calculated data points

    """
    y = amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset
    return y


def fit_gaussian(x_data, y_data, maxiter=None, maxfun=None, verbose=0, initial_parameters=None, initial_params=None,
                 estimate_offset=True):
    raise Exception('The fit_gaussian method has moved to qtt.algorithms.fitting')


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


def fit_double_gaussian(x_data, y_data, maxiter=None, maxfun=5000, verbose=1, initial_params=None):
    raise Exception('fit_double_gaussian was moved to qtt.algorithms.fitting')


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
    minsignal, maxsignal = np.percentile(y_data, [2, 98])
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
    cost = np.sum([(np.asarray(y_data)[1:] - np.asarray(model)[1:]) ** 2 * (np.diff(x_data)) ** weight_power])
    return cost


def estimate_dominant_frequency(signal, sample_rate=1, remove_dc=True, fig=None):
    """ Estimate dominant frequency in a signal

    Args:
        signal (array): Input data
        sample_rate (float): Sample rate of the data
        remove_dc (bool): If True, then do not estimate the DC component
        fig (int or None): Optionally plot the estimated frequency
    Returns:
        Estimated dominant frequency
    """
    w = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1. / sample_rate)

    if remove_dc:
        w[0] = 0
    w[freqs < 0] = 0

    dominant_idx = np.argmax(np.abs(w))
    dominant_frequency = freqs[dominant_idx]

    if 0 < dominant_idx < freqs.size / 2 - 1:
        dominant_idx_subpixel, _ = subpixelmax(np.abs(w), [dominant_idx])
        dominant_frequency = np.interp(dominant_idx_subpixel, np.arange(freqs.size), freqs)[0]

    if fig:
        plt.figure(fig)
        plt.clf()
        plt.plot(freqs, np.abs(w), '.b')
        plt.xlabel('Frequency')
        plt.ylabel('Abs of fft')
        plot_vertical_line(dominant_frequency, label='Dominant frequency')
    return dominant_frequency

# %%


def estimate_parameters_damped_sine_wave(x_data, y_data, exponent=2):
    """ Estimate initial parameters of a damped sine wave

    The damped sine wave is described in https://en.wikipedia.org/wiki/Damped_sine_wave.
    This is a first estimate of the parameters, no numerical optimization is performed.

    The amplitude is estimated from the minimum and maximum values of the data. The osciallation frequency using
    the dominant frequency in the FFT of the signal. The phase of the signal is calculated based on the first
    datapoint in the sequences and the other parameter estimates. Finally, the decay factor of the damped sine wave is
    determined by a heuristic rule.

    Example:
        >>> estimate_parameters_damped_sine_wave(np.arange(10), np.sin(np.arange(10)))

    Args:
        x_data (float): Independent data
        y_data (float): Dependent data
        exponent (float): Exponent from the exponential decay factor

    Returns:
        Estimated parameters for damped sine wave (see the gauss_ramsey method)
    """
    y_data = np.asarray(y_data)
    A = (y_data.max() - y_data.min()) / 2
    B = y_data.min() + A

    n = int(x_data.size / 2)
    mean_left = np.mean(np.abs(y_data[:n] - B))
    mean_right = np.mean(np.abs(y_data[n:] - B))

    laplace_factor = 1e-16
    decay_factor = (mean_left + laplace_factor) / (mean_right + laplace_factor)

    duration = x_data[-1] - x_data[0]
    sample_rate = (x_data.size - 1) / duration
    frequency = estimate_dominant_frequency(y_data, sample_rate=sample_rate)

    if A == 0:
        angle = 0
    else:
        n_start = max(min((y_data[0] - B) / A, 1), -1)
        angle_first_datapoint = -np.arcsin(n_start)
        angle = angle_first_datapoint + 2 * np.pi * frequency * x_data[0]
        angle = np.mod(np.pi + angle, 2 * np.pi) - np.pi
    t2s = 2 * duration / decay_factor

    initial_params = np.array([A, t2s, frequency, angle, B])
    return initial_params


def fit_gauss_ramsey(x_data, y_data, weight_power=None, maxiter=None, maxfun=5000, verbose=1, initial_params=None):
    """ Fit a gauss_ramsey. The function gauss_ramsey gives a model for the measurement result of a pulse Ramsey
    sequence while varying the free evolution time, the phase of the second pulse is made dependent on the free
    evolution time.
    This results in a gaussian decay multiplied by a sinus. Function as used by T.F. Watson et all.,
    see function 'gauss_ramsey' and example in qtt/docs/notebooks/example_fit_ramsey.ipynb

    Args:
        x_data (array): the data for the independent variable
        y_data (array): the data for the measured variable
        weight_power (float or None): If a float then weight all the residual errors with a scale factor
        maxiter (int): maximum number of iterations to perform
        maxfun (int): maximum number of function evaluations to make
        verbose (int): set to >0 to print convergence messages
        initial_params (None or array): optional, initial guess for the fit parameters: [A,C,ramseyfreq,angle,B]

    Returns:
        par_fit (array): array with the fit parameters: [A,t2s,ramseyfreq,angle,B]
        result_dict (dict): dictionary containing a description, the par_fit and initial_params

    """

    def gauss_ramsey_model(x, amplitude, decay_time, frequency, phase, offset):
        """  """
        y = gauss_ramsey(x, [amplitude, decay_time, frequency, phase, offset])
        return y

    if weight_power is None:
        weights = None
    else:
        diff_x = np.diff(x_data)
        weights = np.hstack((diff_x[0], diff_x)) ** weight_power

    if initial_params is None:
        initial_parameters = estimate_parameters_damped_sine_wave(x_data, y_data, exponent=2)
    else:
        initial_parameters = initial_params
    lmfit_model = Model(gauss_ramsey_model)
    lmfit_model.set_param_hint('amplitude', min=0)
    lmfit_model.set_param_hint('decay_time', min=0)
    lmfit_result = lmfit_model.fit(y_data, x=x_data, **dict(zip(lmfit_model.param_names, initial_parameters)),
                                   verbose=verbose >= 2, weights=weights, method='least_squares')

    import qtt.algorithms.fitting
    result_dict = qtt.algorithms.fitting.extract_lmfit_parameters(lmfit_model, lmfit_result)

    result_dict['description'] = 'Function to analyse the results of a Ramsey experiment, ' + \
        'fitted function: gauss_ramsey = A * exp(-(x_data/t2s)**2) * sin(2*pi*ramseyfreq * x_data - angle) + B'

    # backwards compatibility
    result_dict['parameters fit'] = result_dict['fitted_parameters']
    result_dict['parameters initial guess'] = initial_parameters

    return result_dict['fitted_parameters'], result_dict


def plot_gauss_ramsey_fit(x_data, y_data, fit_parameters, fig: Union[int, Axes, None]):
    """ Plot Gauss Ramsey fit

    Args:
        x_data: Input array with time variable (in seconds)
        y_data: Input array with signal
        fit_parameters: Result of fit_gauss_ramsey (fitting units in seconds)
        fig: Figure or axis handle. Is passed to `get_axis`
    """
    test_x = np.linspace(0, np.max(x_data), 200)
    freq_fit = abs(fit_parameters[2] * 1e-6)
    t2star_fit = fit_parameters[1] * 1e6

    ax = get_axis(fig)
    ax.plot(x_data * 1e6, y_data, 'o', label='Data')
    ax.plot(test_x * 1e6, gauss_ramsey(test_x, fit_parameters), label='Fit')
    ax.set_title(r'Gauss Ramsey fit: %.2f MHz / $T_2^*$: %.1f $\mu$s' % (freq_fit, t2star_fit))
    ax.set_xlabel(r'time ($\mu$s)')
    ax.set_ylabel('Spin-up probability')
    ax.legend()


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
    r""" Logistic function

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


def raised_cosine_frequency_domain(frequency: np.ndarray, roll_off_factor: float, symbol_period: float) -> np.ndarray:
    """ Raised cosine frequency domain function

    See https://en.wikipedia.org/wiki/Raised-cosine_filter

    Args:
        frequency: Frequency
        roll_off_factor: Roll-off factor
        symbol_period: Symbol period

    Returns:
        Calculated values of the raised cosine filter in frequency domain
    """
    f = np.asarray(frequency)
    if symbol_period == 0:
        raise ValueError('argument symbol_period should be positive')
    two_symbol_period = 2*symbol_period
    if roll_off_factor == 0:
        return np.abs(f) < 1/(two_symbol_period)
    f_abs = np.abs(f)
    ww = f_abs-(1-roll_off_factor)/(two_symbol_period)
    value = .5*(1+np.cos((np.pi*symbol_period/roll_off_factor)*ww))
    idx = f_abs <= (1-roll_off_factor)/two_symbol_period
    value[idx] = 1
    idx = f_abs >= (1+roll_off_factor)/two_symbol_period
    value[idx] = 0
    return value


def raised_cosine(t: np.ndarray, roll_off_factor: float, symbol_period: float) -> np.ndarray:
    """ Raised cosine impulse response function

    See https://en.wikipedia.org/wiki/Raised-cosine_filter

    Args:
        t: Independent variable
        roll_off_factor: Roll-off factor
        symbol_period: Symbol period

    Returns:
        Calculated values of the raised cosine
    """
    t = np.asarray(t)
    if symbol_period == 0:
        raise ValueError('argument symbol_period should be positive')
    t_div_T = t / symbol_period
    if roll_off_factor == 0:
        return (1 / symbol_period) * np.sinc(t_div_T)

    # special points where the usual formula has a division by zero
    idx_limit = np.abs(t) == symbol_period / (2 * roll_off_factor)
    t_div_T[idx_limit] = 0  # exclude in calculation

    def sinc(x):
        pi_x = np.pi*x
        return np.sin(pi_x)/(pi_x)

    rc = (1 / symbol_period) * np.sinc(t_div_T) * np.cos(np.pi *
                                                         roll_off_factor * t_div_T) / (1 - (2 * roll_off_factor * t_div_T)**2)

    rc[idx_limit] = (np.pi / (4 * symbol_period)) * np.sinc(1 / (2 * roll_off_factor))
    return rc
