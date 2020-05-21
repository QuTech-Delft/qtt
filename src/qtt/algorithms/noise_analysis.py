#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from lmfit.models import LinearModel
from numpy.fft import irfft, rfftfreq
from numpy.random import normal


def pink_noise_model(frequency: Union[float, np.ndarray], A: float, alpha: float) -> np.ndarray:
    """ Model for pink noise

    See https://en.wikipedia.org/wiki/Pink_noise

    Args:
        frequency: Independent variable
        A: Offset of the model
        alpha: Exponent of the model
    Returns:
        PSD of the noise model
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
        fig = plt.figure(fig)
        plt.clf()
        ax = fig.add_subplot(111)
    else:
        ax = plt.gca()
    ax.loglog(frequencies, psd, label=label, color='navy')
    plt.xlabel(f'Frequency ({frequency_unit})')
    plt.ylabel(f'Power spectral density ({signal_unit}$^2$/{frequency_unit})')


def generate_pink_noise(sample_rate: float, number_of_samples: int, A: float = 1, alpha: float = 1):
    """ Generate sampled data with 1/f noise

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

    fft = normal(scale=s_scale, size=size) + 1J * normal(scale=s_scale, size=size)
    fft[0] = 0

    w = s_scale[1:].copy()
    w[-1] *= (1 + (number_of_samples % 2)) / 2.
    sigma = 2 * np.sqrt(np.sum(w**2)) / number_of_samples
    signal = 3.5 * np.sqrt(A) * irfft(fft) / sigma

    return signal


def outlier_detection(data: np.ndarray, threshold: Optional[float] = None,
                      percentile: float = 90):
    """ Detect outliers in data using a threshold

    Args:
        data: Data from which to determine the outliers
        threshold: Threshold to use for outlier detection. If None, then use the percentile argument to automatically determine the threshold
        percentile: Determine the threshold by setting it the the percentile specified
    Returns:
        Boolean array with value True for the inliers
    """
    residuals = np.abs(data)
    if threshold is None:
        threshold = np.percentile(residuals, percentile)
    inliers = residuals <= threshold
    return inliers


def fit_pink_noise(x_data: np.ndarray, y_data: np.ndarray, initial_parameters: Optional[np.ndarray] = None,
                   remove_outliers: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ Fit a model to data with 1/f noise

    The model fitted is = A/f^alpha

    The fitting is performed in loglog coordinates.

    Args:
        x_data: Independent variable
        y_data: Dependent variable
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
        A0 = y_data[0]
        alpha0 = 1
        initial_parameters = [A0, alpha0]

    initial_parameters_log_log = transform_to_loglog(initial_parameters)

    if x_data[0] == 0:
        y_data_log = np.log(y_data[1:])
        x_data_log = np.log(x_data[1:])
    else:
        y_data_log = np.log(y_data)
        x_data_log = np.log(x_data)

    lmfit_model = LinearModel(independent_vars=['x'], name='Pink noise in loglog')
    lmfit_result = lmfit_model.fit(y_data_log, x=x_data_log, **
                                   dict(zip(lmfit_model.param_names, initial_parameters_log_log)))

    inliers = None
    if remove_outliers:
        inliers = outlier_detection(lmfit_result.residual)

        logging.info(f'fit_pink_noise: outlier detection: number of outliers: {(inliers==False).sum()}')
        lmfit_result = lmfit_model.fit(y_data_log[inliers], x=x_data_log[inliers], **lmfit_result.best_values)

    loglog_result_dict = extract_lmfit_parameters(lmfit_model, lmfit_result)

    fitted_parameters = transform_from_loglog(loglog_result_dict['fitted_parameters'])
    result_dict = {'fitted_parameters': fitted_parameters, 'initial_parameters': initial_parameters,
                   'inliers': inliers,
                   'fitted_parameter_dictionary': dict(zip(['A', 'alpha'], fitted_parameters)),
                   'results_loglog_fit': loglog_result_dict}
    return result_dict['fitted_parameters'], result_dict
