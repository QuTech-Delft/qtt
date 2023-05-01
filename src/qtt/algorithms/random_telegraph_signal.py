""" Functionality to analyse random telegraph signals

Created on Wed Feb 28 10:20:46 2018

@author: riggelenfv /eendebakpt
"""

import operator
import warnings
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import qcodes_loop

from qtt.algorithms.fitting import fit_double_gaussian, refit_double_gaussian
from qtt.algorithms.functions import double_gaussian, exp_function, fit_exp_decay, gaussian
from qtt.algorithms.markov_chain import ContinuousTimeMarkovModel
from qtt.utilities.tools import addPPTslide
from qtt.utilities.visualization import get_axis, plot_double_gaussian_fit, plot_vertical_line


def rts2tunnel_ratio(binary_signal: np.ndarray) -> float:
    """ Calculate ratio between tunnelrate down and up

    From the mean and standard deviation of the RTS data we can determine the ratio between
    the two tunnel rates. See equations on https://en.wikipedia.org/wiki/Telegraph_process

    Args:
        binary_signal: RTS signal with two levels 0 and 1

    Returns:
        Ratio of tunnelrate up to down (l2) and down to up (l1)
    """

    binary_signal = np.asarray(binary_signal)
    c1 = binary_signal.min()
    c2 = binary_signal.max()

    number_of_transitions = np.abs(np.diff(binary_signal)).sum()

    if number_of_transitions < 40:
        warnings.warn(f'number of transitions {number_of_transitions} is low, estimate can be inaccurate')

    if c1 == c2:
        raise ValueError(f'binary signal contains only a single value {c1}')

    if c1 != 0 or c2 != 1:
        raise ValueError('signal must only contain 0 and 1')
    m = binary_signal.mean()
    var = binary_signal.var()

    ratio_l2_over_l1 = var/m**2

    return ratio_l2_over_l1


def transitions_durations(data: np.ndarray, split: float, add_start: bool = False,
                          add_end: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """ For data of a two level system (up and down) determine durations of segments

    This function determines which datapoints belong to which
    level and finds the transitions, in order to determines
    how long the system stays in these levels.

    Args:
        data : data from the two level system
        split: value that separates the up and down level
        add_start: If True, then include the segments at the start of the data
        add_end:: If True, then include the segments at the end of the data

    Returns:
        duration_dn:  array of the durations (unit: data points) in the down level
        duration_up: array of durations (unit: data points) in the up level
    """

    size = len(data)
    if size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # split the data and find the index of the transitions, transitions from
    # up to down are marked with -1 and from down to up with 1
    b = np.asarray(data) > split
    d = np.diff(b.astype(int))
    transitions_down_to_up = (d == 1).nonzero()[0]
    transitions_up_to_down = (d == -1).nonzero()[0]

    # durations are calculated by taking the difference in data points between
    # the transitions
    endpoints_dn = []
    endpoints_up = []
    if data[0] <= split and data[-1] <= split:
        duration_up = transitions_up_to_down - transitions_down_to_up
        duration_dn = transitions_down_to_up[1:] - transitions_up_to_down[:-1]

        if len(transitions_up_to_down) == 0:
            if add_start or add_end:
                endpoints_dn.append(size)
        else:
            if add_start:
                endpoints_dn.append(transitions_down_to_up[0]+1)
            if add_end:
                endpoints_dn.append(size - transitions_up_to_down[-1]-1)

    elif data[0] <= split < data[-1]:
        duration_up = transitions_up_to_down - transitions_down_to_up[:-1]
        duration_dn = transitions_down_to_up[1:]-transitions_up_to_down

        if add_start:
            endpoints_dn.append(transitions_down_to_up[0]+1)
        if add_end:
            endpoints_up.append(size-transitions_down_to_up[-1]-1)

    elif data[0] > split >= data[-1]:
        duration_up = transitions_up_to_down[1:] - transitions_down_to_up
        duration_dn = transitions_down_to_up - transitions_up_to_down[:-1]

        if add_start:
            endpoints_up.append(transitions_up_to_down[0]+1)
        if add_end:
            endpoints_dn.append(size-transitions_up_to_down[-1]-1)

    else:  # case: data[0] > split and data[-1] > split:
        duration_up = transitions_up_to_down[1:] - transitions_down_to_up[:-1]
        duration_dn = transitions_down_to_up - transitions_up_to_down

        if len(transitions_up_to_down) == 0:
            if add_start or add_end:
                endpoints_up.append(size)
        else:
            if add_start:
                endpoints_up.append(transitions_up_to_down[0]+1)
            if add_end:
                endpoints_up.append(size-transitions_down_to_up[-1]-1)

    duration_dn = np.concatenate((duration_dn, np.asarray(endpoints_dn, dtype=int)), dtype=int)
    duration_up = np.concatenate((duration_up, np.asarray(endpoints_up, dtype=int)), dtype=int)

    return duration_dn, duration_up


class FittingException(Exception):
    """ Fitting exception in RTS code """
    pass


def _plot_rts_histogram(data, num_bins, double_gaussian_fit, split, figure_title):
    _, bins, _ = plt.hist(data, bins=num_bins)
    bincentres = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 1)])

    get_left_mean_std_amplitude = operator.itemgetter(4, 2, 0)
    get_right_mean_std_amplitude = operator.itemgetter(5, 3, 1)
    left_gaussian = list(get_left_mean_std_amplitude(double_gaussian_fit))
    right_gaussian = list(get_right_mean_std_amplitude(double_gaussian_fit))

    plt.plot(bincentres, double_gaussian(bincentres, double_gaussian_fit), '-m', label='Fitted double gaussian')
    plt.plot(bincentres, gaussian(bincentres, *left_gaussian), 'g', label='Left Gaussian', alpha=.85, linewidth=.75)
    plt.plot(bincentres, gaussian(bincentres, *right_gaussian), 'r', label='Right Gaussian', alpha=.85, linewidth=.75)
    plt.plot(split, double_gaussian(split, double_gaussian_fit), 'ro', markersize=8, label='split: %.3f' % split)
    plt.xlabel('Measured value (a.u.)')
    plt.ylabel('Data points per bin')
    plt.legend()
    plt.title(figure_title)


def two_level_threshold(data: np.ndarray, number_of_bins: int = 40) -> dict:
    """ Determine threshold for separation of two-level signal

    Typical examples of such a signal are an RTS signal or Elzerman readout.

    Args:
        data: Two dimensional array with single traces
        number_of_bins: Number of bins to use for calculation of double histogram

    Returns:
        Dictionary with results. The key readout_threshold contains the calculated threshold
    """
    counts, bins = np.histogram(data, bins=number_of_bins)
    bin_centres = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 1)])
    _, result_dict = fit_double_gaussian(bin_centres, counts)
    result_dict = refit_double_gaussian(result_dict, bin_centres, counts)

    result = {'signal_threshold': result_dict['split'], 'double_gaussian_fit': result_dict,
              'separation': result_dict['separation'],
              'histogram': {'counts': counts, 'bins': bins, 'bin_centres': bin_centres}}
    return result


def plot_two_level_threshold(results: dict, fig: int = 100, plot_initial_estimate: bool = False):
    separation = results['separation']
    threshold = results['signal_threshold']

    ax = get_axis(fig)
    bin_centres = results['histogram']['bin_centres']
    counts = results['histogram']['counts']
    ax.bar(bin_centres, counts, width=bin_centres[1] - bin_centres[0], label='histogram')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Signal [a.u.]')
    plot_vertical_line(threshold, label='threshold')
    plot_double_gaussian_fit(results['double_gaussian_fit'], bin_centres)

    ax.set_title(f'Two-level signal: separation {separation:.3f}, threshold {threshold:.3g}')

    if plot_initial_estimate:
        xdata = np.linspace(bin_centres[0], bin_centres[-1], 300)
        initial_estimate = results['double_gaussian_fit']['parameters initial guess']
        left0 = initial_estimate[::2][::-1]
        right0 = initial_estimate[1::2][::-1]
        ax.plot(xdata, gaussian(xdata, *left0), ':g', label='initial estimate left')
        ax.plot(xdata, gaussian(xdata, *right0), ':r', label='initial estimate right')


def _create_integer_histogram(durations):
    """ Calculate number of bins, bin edges and histogram for durations

    This method works if the data is sampled at integer durations.
    """
    numbins = int(np.sqrt(len(durations)))

    if numbins == 0:
        raise Exception('cannot create histogram with zero bins')

    bin_size = int(np.ceil((durations.max() - (durations.min() - .5)) / numbins))
    # choose bins carefully, since our data is sampled only at discrete times
    bins = np.arange(durations.min() - .5, durations.max() + bin_size, bin_size)
    counts, bin_edges = np.histogram(durations, bins=bins)
    return counts, bin_edges, bin_size


def tunnelrates_RTS(data: Union[np.ndarray, qcodes_loop.data.data_set.DataSet], samplerate: Optional[float] = None,
                    min_sep: float = 2.0, max_sep: float = 7.0, min_duration: int = 5,
                    num_bins: Optional[int] = None, fig: Optional[int] = None, ppt=None,
                    verbose: int = 0,
                    offset_parameter: Optional[float] = None) -> Tuple[Optional[float], Optional[float], dict]:
    """
    This function takes an RTS dataset, fits a double gaussian, finds the split between the two levels,
    determines the durations in these two levels, fits a decaying exponential on two arrays of durations,
    which gives the tunneling frequency for both the levels. If the number of datapoints is too low to get enough
    points per bin for the exponential fit (for either the up or the down level), this analysis step is passed over.
    tunnelrate_dn and tunnelrate_up are returned as None, but similar information can be substracted from
    parameters['down_segments'] and parameters['up_segments'].

    Args:
        data: qcodes DataSet (or 1d data array) with the RTS data
        samplerate: sampling rate of the acquisition device, optional if given in the metadata
                                   of the measured data
        min_sep: if the separation found for the fit of the double gaussian is less then this value, the
                         fit probably failed and a FittingException is raised
        max_sep: if the separation found for the fit of the double gaussian is more then this value, the
                         fit probably failed and a FittingException is raised
        min_duration: minimal number of datapoints a duration should last to be taking into account for the
                            analysis
        num_bins: number of bins for the histogram of signal values. If None, then determine based
                            on the size of the data
        fig: shows figures and sends them to the ppt when is not None
        ppt: determines if the figures are send to a powerpoint presentation
        verbose: prints info to the console when > 0
        offset_parameter: Offset parameter for fitting of exponential decay

    Returns:
        tunnelrate_dn: tunneling rate of the down level to the up level (kHz) or None in case of
                                       not enough datapoints
        tunnelrate_up: tunneling rate of the up level to the down level (kHz) or None in case of
                                       not enough datapoints
        parameters: dictionary with relevent (fit) parameters. this includes:
                tunnelrate_down (float): tunnel rate in Hz
                tunnelrate_up (float): tunnel rate up in Hz

    """
    if isinstance(data, qcodes_loop.data.data_set.DataSet):
        if samplerate is None:
            samplerate = data.metadata.get('samplerate', None)

        data = np.array(data.default_parameter_array())

    if samplerate is None:
        raise ValueError('samplerate should be set to the data samplerate in Hz')

    # plotting a 2d histogram of the RTS
    if fig is not None:
        max_num_bins_time_domain = 1200
        max_num_bins_signal_domain = 800
        xdata = np.arange(len(data)) / (samplerate / 1_000)
        ny = min(int(np.sqrt(len(data))/2), max_num_bins_signal_domain)
        nx = min(int(np.sqrt(len(xdata))/2), max_num_bins_time_domain)
        Z, xedges, yedges = np.histogram2d(xdata, data, bins=[nx, ny])
        title = '2d histogram RTS'
        Fig = plt.figure(fig)
        plt.clf()
        plt.pcolormesh(xedges, yedges, Z.T)
        cb = plt.colorbar()
        cb.set_label('Data points per bin')
        plt.xlabel('Time (ms)')
        plt.ylabel('Signal sensing dot (a.u.)')
        plt.title(title)
        if ppt:
            addPPTslide(title=title, fig=Fig)

    # binning the data and determining the bincentres
    if num_bins is None:
        max_num_bins_fitting = 1200
        num_bins = min(int(np.sqrt(len(data))), max_num_bins_fitting)

    fit_results = two_level_threshold(data, number_of_bins=num_bins)
    separation = fit_results['separation']
    split = fit_results['signal_threshold']
    double_gaussian_fit_parameters = fit_results['double_gaussian_fit']['parameters']

    if verbose:
        print('Fit parameters double gaussian:\n mean down: %.3f counts' %
              double_gaussian_fit_parameters[4] + ', mean up: %.3f counts' % double_gaussian_fit_parameters[
                  5] + ', std down: %.3f counts' % double_gaussian_fit_parameters[2] + ', std up:%.3f counts' %
              double_gaussian_fit_parameters[3])
        print('Separation between peaks gaussians: %.3f std' % separation)
        print('Split between two levels: %.3f' % split)

    # plotting the data in a histogram, the fitted two gaussian model and the split
    if fig:
        figure_title = 'Histogram of two level signal' + f'\nseparation: {separation:.1f} [std]'
        Fig = plt.figure(fig + 1)
        plt.clf()
        _plot_rts_histogram(data, num_bins, double_gaussian_fit_parameters, split, figure_title)

        if ppt:
            addPPTslide(title=title, fig=Fig, notes='Fit parameters double gaussian:\n mean down: %.3f counts' %
                                                    double_gaussian_fit_parameters[4] + ', mean up:%.3f counts' %
                                                    double_gaussian_fit_parameters[5] + ', std down: %.3f counts' %
                                                    double_gaussian_fit_parameters[2] + ', std up:%.3f counts' %
                                                    double_gaussian_fit_parameters[
                                                        3] + '.Separation between peaks gaussians: %.3f std' % separation + '. Split between two levels: %.3f' % split)

    if separation < min_sep:
        raise FittingException(
            'Separation between the peaks of the gaussian %.1f is less then %.1f std, indicating that the fit was not succesfull.' % (
                separation, min_sep))

    if separation > max_sep:
        raise FittingException(
            'Separation between the peaks of the gaussian %.1f is more then %.1f std, indicating that the fit was not succesfull.' % (
                separation, max_sep))

    thresholded_data = data > split
    fraction_up = np.sum(thresholded_data) / data.size
    fraction_down = 1 - fraction_up

    # count the number of transitions and their duration
    durations_dn_idx, durations_up_idx = transitions_durations(data, split)

    # throwing away the durations with less data points then min_duration
    durations_up_min_duration = durations_up_idx >= min_duration
    durations_up = durations_up_idx[durations_up_min_duration]
    durations_dn_min_duration = durations_dn_idx >= min_duration
    durations_dn = durations_dn_idx[durations_dn_min_duration]

    if len(durations_up) < 1:
        raise FittingException('All durations_up are shorter than the minimal duration.')

    if len(durations_dn) < 1:
        raise FittingException('All durations_dn are shorter than the minimal duration.')

    # calculating the number of bins and counts for down level
    counts_dn, bins_dn, _ = _create_integer_histogram(durations_dn)

    # calculating the number of bins and counts for up level
    counts_up, bins_up, _ = _create_integer_histogram(durations_up)

    if verbose >= 2:
        print(f' _create_integer_histogram: up/down: number of bins {len(bins_up)}/{len(bins_dn)}')

    bins_dn = bins_dn / samplerate
    bins_up = bins_up / samplerate

    if verbose >= 2:
        print('counts_dn %d, counts_up %d' % (counts_dn[0], counts_up[0]))

    tunnelrate_dn = None
    tunnelrate_up = None

    minimal_count_number = 50

    if counts_dn[0] < minimal_count_number:
        warnings.warn(
            f'Number of down datapoints {counts_dn[0]} is not enough (minimal_count_number {minimal_count_number})'
            f' to make an accurate fit of the exponential decay for level down. ' +
            'Look therefore at the mean value of the measurement segments')

    if counts_up[0] < minimal_count_number:
        warnings.warn(
            f'Number of up datapoints {counts_up[0]} is not enough (minimal_count_number {minimal_count_number}) to make an acurate fit of the exponential decay for level up. '
            + 'Look therefore at the mean value of the measurement segments')

    parameters = {'sampling rate': samplerate,
                  'fit parameters double gaussian': double_gaussian_fit_parameters,
                  'separations between peaks gaussians': separation,
                  'split between the two levels': split}

    parameters['down_segments'] = {'number': len(durations_dn_idx), 'mean': np.mean(durations_dn_idx) / samplerate, 'p50': np.percentile(
        durations_dn_idx, 50) / samplerate, 'number_filtered': len(durations_dn), 'mean_filtered': np.mean(durations_dn)}
    parameters['up_segments'] = {'number': len(durations_up_idx), 'mean': np.mean(durations_up_idx) / samplerate, 'p50': np.percentile(
        durations_up_idx, 50) / samplerate, 'number_filtered': len(durations_up), 'mean_filtered': np.mean(durations_up)}
    parameters['tunnelrate_down_to_up'] = 1. / parameters['down_segments']['mean']
    parameters['tunnelrate_up_to_down'] = 1. / parameters['up_segments']['mean']

    parameters['fraction_down'] = fraction_down
    parameters['fraction_up'] = fraction_up

    parameters['bins_dn'] = {'number': len(bins_dn), 'size': np.diff(bins_dn).mean(), 'start': bins_dn[0]}
    parameters['bins_up'] = {'number': len(bins_up), 'size': np.diff(bins_up).mean(), 'start': bins_up[0]}
    parameters['tunnelrate_ratio'] = rts2tunnel_ratio(thresholded_data)

    if (counts_dn[0] > minimal_count_number) and (counts_up[0] > minimal_count_number):

        def _fit_and_plot_decay(bincentres, counts, label, fig_label):
            """ Fitting and plotting of exponential decay for level """
            A_fit, B_fit, gamma_fit = fit_exp_decay(bincentres, counts, offset_parameter=offset_parameter)
            tunnelrate = gamma_fit / 1000

            other_label = 'up' if label == 'down' else 'down'

            if verbose:
                print(f'Tunnel rate {label} to {other_label}: %.1f kHz' % tunnelrate)

            time_scaling = 1e3

            if fig_label:
                title = f'Fitted exponential decay, level {label}'
                Fig = plt.figure(fig_label)
                plt.clf()
                plt.plot(time_scaling * bincentres, counts, 'o', label=f'Counts {label}')
                plt.plot(time_scaling * bincentres, exp_function(bincentres, A_fit, B_fit, gamma_fit),
                         'r', label=r'Fitted exponential decay $\Gamma_{\mathrm{%s\ to\ %s}}$: %.1f kHz' % (label, other_label, tunnelrate))
                plt.xlabel('Lifetime (ms)')
                plt.ylabel('Counts per bin')
                plt.legend()
                plt.title(title)
                if ppt:
                    addPPTslide(title=title, fig=Fig)

            fit_parameters = [A_fit, B_fit, gamma_fit]
            return tunnelrate, fit_parameters

        bincentres_dn = (bins_dn[:-1]+bins_dn[1:])/2
        fig_label = None if fig is None else fig + 2
        tunnelrate_dn, fit_parameters_down = _fit_and_plot_decay(
            bincentres_dn, counts_dn, label='down', fig_label=fig_label)

        bincentres_up = (bins_up[:-1]+bins_up[1:])/2
        fig_label = None if fig is None else fig + 3
        tunnelrate_up, fit_parameters_up = _fit_and_plot_decay(
            bincentres_up, counts_up, label='up', fig_label=fig_label)

        parameters['fit parameters exp. decay down'] = fit_parameters_down
        parameters['fit parameters exp. decay up'] = fit_parameters_up

        parameters['tunnelrate_down_exponential_fit'] = tunnelrate_dn
        parameters['tunnelrate_up_exponential_fit'] = tunnelrate_up
    else:
        parameters['tunnelrate_down_exponential_fit'] = None
        parameters['tunnelrate_up_exponential_fit'] = None

    return tunnelrate_dn, tunnelrate_up, parameters


def generate_RTS_signal(number_of_samples: int = 100000, std_gaussian_noise: float = 0.1,
                        uniform_noise: float = 0.05, rate_up: float = 10e3, rate_down: float = 15e3, samplerate: float = 1e6) -> np.ndarray:
    """ Generate a RTS signal

    Args:
        number_of_samples: Length the the trace to be generated
        std_normal_noise: std of Gaussian noise added to the signal
        uniform_noise: uniform noise in the range +- uniform_noise/2 is added to the signal
        rate_up: rate from down to up
        rate_down: rate from up to down
        samplerate: The samplerate of the signal to be generated
    Returns:
        Array with generated signal (0 is down, 1 is up)

    """

    rts_model = ContinuousTimeMarkovModel(['down', 'up'], [rate_up / samplerate,
                                                           rate_down / samplerate], np.array([[0., 1], [1, 0]]))

    data = rts_model.generate_sequence(number_of_samples, delta_time=1)

    if uniform_noise != 0:
        data = data + uniform_noise * (np.random.rand(data.size, ) - .5)
    if std_gaussian_noise != 0:
        data = data + np.random.normal(0, std_gaussian_noise, data.size)
    return data
