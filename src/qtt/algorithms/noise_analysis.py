"""

@author: eendebakpt
"""

# %%
from typing import Tuple, Union, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import logging
from qtt.algorithms.fitting import extract_lmfit_parameters
from lmfit.models import LinearModel, Model
from numpy.fft import irfft, rfftfreq
from numpy.random import normal


def power_law_model(frequency: Union[float, np.ndarray], A: float, alpha: float) -> np.ndarray:
    """ Model for power law

    This function implements the function y = A/x^alpha

    Also see: https://en.wikipedia.org/wiki/Power_law
    For alpha=1 we have the model for pink noise, see https://en.wikipedia.org/wiki/Pink_noise

    Args:
        frequency: Independent variable
        A: Scalar factor of the model
        alpha: Exponent of the model
    Returns:
        Calculated value of the model
    """
    return A / frequency**alpha


def calculate_psd_welch(signal: np.ndarray, sample_rate: float, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculate the PSD using the Welch method

    Args:
        signal: Data for which to calculate the PSD
        sample_rate: Sample rate of the data in Hz
        nperseg: Passed to the scipy welch method
    Returns:
        Tuple of frequencies and PSD

    """
    frequencies, psd_welch = scipy.signal.welch(signal, sample_rate, nperseg=nperseg)
    return frequencies, psd_welch


def plot_psd(frequencies: np.ndarray, psd: np.ndarray, frequency_unit: str = 'Hz', signal_unit: str = 'V',
             label: Optional[str] = None, fig: int = 1):
    """ Plot calculated PSD """
    if fig is not None:
        Fig = plt.figure(fig)
        plt.clf()
        ax = Fig.add_subplot(111)
    else:
        ax = plt.gca()
    ax.loglog(frequencies, psd, label=label, color='navy')
    plt.xlabel(f'Frequency ({frequency_unit})')
    plt.ylabel(f'Power spectral density ({signal_unit}$^2$/{frequency_unit})')


def generate_powerlaw_noise(sample_rate: float, number_of_samples: int, A: float = 1, alpha: float = 1):
    """ Generate sampled data with 1/f noise

    The method is based on "On generating power law noise.", J. Timmer and M. Konig, Astron. Astrophys. 300,
    pp. 707-710 (1995). For alpha = 1 this method generates pink noise, for alpha = 2 brown noise and for
    alpha = 0 white noise.

    Args:
        sample_rate: Rate of sampling
        number_of_samples: Number of samples to generate
        A: Parameter of noise model
        alpha: Parameter of noise model

    Returns:
        Array with sampled data
    """
    frequencies = rfftfreq(number_of_samples, 1 / sample_rate)

    s_scale = frequencies
    s_scale[0] = s_scale[1]
    s_scale = s_scale**(-alpha / 2.)
    size = s_scale.size

    signal_fft = normal(scale=s_scale, size=size) + 1J * normal(scale=s_scale, size=size)
    signal_fft[0] = 0

    w = s_scale[1:].copy()
    w[-1] *= (1 + (number_of_samples % 2)) / 2.
    sigma = 2 * np.sqrt(np.sum(w**2)) / number_of_samples
    signal = 3.5 * np.sqrt(A) * irfft(signal_fft) / sigma

    return signal


def get_outlier_mask(data: np.ndarray, threshold: Optional[float] = None,
                      percentile: float = 90):
    """ Detect outliers in data using a threshold

    Args:
        data: Data from which to determine the outliers
        threshold: Threshold to use for outlier detection. If None, then use the percentile argument to automatically determine the threshold
        percentile: Determine the threshold by setting it to the percentile specified
    Returns:
        Boolean array with value True for the inliers
    """
    residuals = np.abs(data)
    if threshold is None:
        threshold = np.percentile(residuals, percentile)
    inlier_mask = residuals <= threshold
    return inlier_mask


def fit_power_law(frequencies: np.ndarray, signal_data: np.ndarray, initial_parameters: Optional[np.ndarray] = None,
                  remove_outliers: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Fit a model to data with a power law distribution

    The model fitted is signal = A/f^alpha

    Args:
        frequencies: Independent variable
        signal_data: Dependent variable
        initial_parameters: Optional list with estimate of model parameters
        remove_outliers: If True, then remove outliers in the fitting

    Returns:
        Tuple of fitted parameters and results dictionary
    """
    if initial_parameters is None:
        A0 = signal_data[0]
        alpha0 = 1
        initial_parameters = [A0, alpha0]

    if np.any(frequencies == 0):
        raise Exception('input data cannot contain 0')

    lmfit_model = Model(power_law_model, name='Power law model')
    lmfit_result = lmfit_model.fit(signal_data, frequency=frequencies, **
                                   dict(zip(lmfit_model.param_names, initial_parameters)))

    inliers = None
    if remove_outliers:
        inliers = get_outlier_mask(lmfit_result.residual)

        logging.info(f'fit_power_law: outlier detection: number of outliers: {(inliers==False).sum()}')
        lmfit_result = lmfit_model.fit(signal_data[inliers], frequency=frequencies[inliers], **lmfit_result.best_values)

    result_dict = extract_lmfit_parameters(lmfit_model, lmfit_result)
    result_dict['description'] = 'fit of power law model'
    result_dict['inliers'] = inliers

    return result_dict['fitted_parameters'], result_dict


def fit_power_law_loglog(frequencies: np.ndarray, signal_data: np.ndarray, initial_parameters: Optional[np.ndarray] = None,
                         remove_outliers: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Fit a model to data with a power law distribution

    The model fitted is signal = A/f^alpha

    The fitting is performed in loglog coordinates.
    The data points are weighted to have an equal contribution in the frequency log-space.

    Args:
        frequencies: Independent variable
        signal_data: Dependent variable
        initial_parameters: Optional list with estimate of model parameters
        remove_outliers: If True, then remove outliers in the fitting

    Returns:
        Tuple of fitted parameters and results dictionary
    """

    def transform_from_loglog(x):
        slope = x[0]
        intercept = x[1]
        A = np.exp(intercept)
        alpha = -slope
        return [A, alpha]

    def transform_to_loglog(x):
        intercept = np.log(x[0])
        slope = -x[1]
        return [slope, intercept]

    if initial_parameters is None:
        A0 = signal_data[0]
        alpha0 = 1
        initial_parameters = [A0, alpha0]

    initial_parameters_log_log = transform_to_loglog(initial_parameters)

    if np.any(frequencies == 0):
        raise Exception('input frequencies cannot contain 0')
    else:
        signal_data_log = np.log(signal_data)
        frequencies_log = np.log(frequencies)

    weights = np.sqrt(1 / frequencies)

    lmfit_model = LinearModel(independent_vars=['x'], name='Power law model in loglog coordinates')
    lmfit_result = lmfit_model.fit(signal_data_log, x=frequencies_log, **
                                   dict(zip(lmfit_model.param_names, initial_parameters_log_log)), weights=weights)

    inliers = None
    if remove_outliers:
        inliers = get_outlier_mask(lmfit_result.residual)

        logging.info(f'fit_power_law: outlier detection: number of outliers: {(inliers==False).sum()}')
        lmfit_result = lmfit_model.fit(signal_data_log[inliers], x=frequencies_log[inliers], **lmfit_result.best_values)

    loglog_result_dict = extract_lmfit_parameters(lmfit_model, lmfit_result)

    fitted_parameters = transform_from_loglog(loglog_result_dict['fitted_parameters'])
    result_dict = {'fitted_parameters': fitted_parameters, 'initial_parameters': initial_parameters,
                   'inliers': inliers,
                   'fitted_parameter_dictionary': dict(zip(['A', 'alpha'], fitted_parameters)),
                   'results_loglog_fit': loglog_result_dict,
                   'description': 'fit of power law model in loglog coordinates'}
    return result_dict['fitted_parameters'], result_dict
