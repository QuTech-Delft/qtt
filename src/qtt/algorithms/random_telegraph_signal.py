""" Functionality to analyse random telegraph signals

Created on Wed Feb 28 10:20:46 2018

@author: riggelenfv
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import qcodes
from qtt.utilities.tools import addPPTslide
import warnings

from qtt.algorithms.functions import double_gaussian, fit_double_gaussian, exp_function, fit_exp_decay
from qtt.algorithms.markov_chain import ContinuousTimeMarkovModel
from qtt.utilities.visualization import plot_vertical_line, plot_double_gaussian_fit


# %% calculate durations of states


def transitions_durations(data, split):
    """ For data of a two level system (up and down), this funtion determines which datapoints belong to which
    level and finds the transitions, in order to determines
    how long the system stays in these levels.

    Args:
        data (numpy array): data from the two level system
        split (float): value that separates the up and down level

    Returns:
        duration_dn (numpy array): array of the durations (unit: data points) in the down level
        duration_up (numpy array): array of durations (unit: data points) in the up level
    """

    # split the data and find the index of the transitions, transitions from
    # up to down are marked with -1 and from down to up with 1
    b = data > split
    d = np.diff(b.astype(int))
    transitions_dn = (d == -1).nonzero()[0]
    transitions_up = (d == 1).nonzero()[0]

    # durations are calculated by taking the difference in data points between
    # the transitions, first and last duration are ignored
    if data[0] <= split and data[-1] <= split:
        duration_up = transitions_dn - transitions_up
        duration_dn = transitions_up[1:] - transitions_dn[:-1]

    elif data[0] < split and data[-1] > split:
        duration_up = transitions_dn - transitions_up[:-1]
        duration_dn = transitions_up[1:] - transitions_dn

    elif data[0] > split and data[-1] < split:
        duration_up = transitions_dn[1:] - transitions_up
        duration_dn = transitions_up - transitions_dn[:-1]

    elif data[0] >= split and data[-1] >= split:
        duration_up = transitions_dn[1:] - transitions_up[:-1]
        duration_dn = transitions_up - transitions_dn

    return duration_dn, duration_up


class FittingException(Exception):
    """ Fitting exception in RTS code """
    pass


def _plot_rts_histogram(data, num_bins, double_gaussian_fit, split, figure_title):
    _, bins, _ = plt.hist(data, bins=num_bins)
    bincentres = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 1)])

    plt.plot(bincentres, double_gaussian(bincentres, double_gaussian_fit), 'r', label='Fitted double gaussian')
    plt.plot(split, double_gaussian(split, double_gaussian_fit), 'ro', markersize=8, label='split: %.3f' % split)
    plt.xlabel('Measured value (a.u.)')
    plt.ylabel('Data points per bin')
    plt.legend()
    plt.title(figure_title)


def two_level_threshold(data, number_of_bins=40) -> dict:
    """ Determine threshold for separation of two-level signal

    Typical examples of such a signal are an RTS signal or Elzerman readout.

    Args:
        traces: Two dimensional array with single traces
        number_of_bins: Number of bins to use for calculation of double histogram

    Returns:
        Dictionary with results. The key readout_threshold contains the calculated threshold
    """
    counts, bins = np.histogram(data, bins=number_of_bins)
    bin_centres = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 1)])
    _, result_dict = fit_double_gaussian(bin_centres, counts)

    result = {'signal_threshold': result_dict['split'], 'double_gaussian_fit': result_dict,
              'separation': result_dict['separation'],
              'histogram': {'counts': counts, 'bins': bins, 'bin_centres': bin_centres}}
    return result


def plot_two_level_threshold(results, fig=100):
    plt.figure(fig)
    plt.clf()
    bin_centres = results['histogram']['bin_centres']
    counts = results['histogram']['counts']
    plt.bar(bin_centres, counts, width=bin_centres[1] - bin_centres[0], label='histogram')
    plt.ylabel('Counts')
    plt.xlabel('Signal [a.u.]')
    plot_vertical_line(results['signal_threshold'], label='threshold')
    plot_double_gaussian_fit(results['double_gaussian_fit'], bin_centres)
    plt.title('Result of two level threshold processing')


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


def tunnelrates_RTS(data, samplerate=None, min_sep=2.0, max_sep=7.0, min_duration=5,
                    num_bins=None, plungers=None, fig=None, ppt=None, verbose=0):
    """
    This function takes an RTS dataset, fits a double gaussian, finds the split between the two levels,
    determines the durations in these two levels, fits a decaying exponential on two arrays of durations,
    which gives the tunneling frequency for both the levels. If the number of datapoints is too low to get enough
    points per bin for the exponential fit (for either the up or the down level), this analysis step is passed over.
    tunnelrate_dn and tunnelrate_up are returned as None, but similar information can be substracted from
    parameters['down_segments'] and parameters['up_segments'].

    Args:
        data (array): qcodes DataSet (or 1d data array) with the RTS data
        samplerate (float): sampling rate of the acquisition device, optional if given in the metadata
                                   of the measured data
        min_sep (float): if the separation found for the fit of the double gaussian is less then this value, the
                         fit probably failed and a FittingException is raised
        max_sep (float): if the separation found for the fit of the double gaussian is more then this value, the
                         fit probably failed and a FittingException is raised
        min_duration (int): minimal number of datapoints a duration should last to be taking into account for the
                            analysis
        num_bins (int or None) : number of bins for the histogram of signal values. If None, then determine based
                            on the size of the data
        fig (None or int): shows figures and sends them to the ppt when is not None
        ppt (None or int): determines if the figures are send to a powerpoint presentation
        verbose (int): prints info to the console when > 0

    Returns:
        tunnelrate_dn (numpy.float64): tunneling rate of the down level to the up level (kHz) or None in case of
                                       not enough datapoints
        tunnelrate_up (numpy.float64): tunneling rate of the up level to the down level (kHz) or None in case of
                                       not enough datapoints
        parameters (dict): dictionary with relevent (fit) parameters. this includes:
                tunnelrate_down (float): tunnel rate in Hz
                tunnelrate_up (float): tunnel rate up in Hz

    """
    if plungers is not None:
        raise Exception('argument plungers is not used any more')

    if isinstance(data, qcodes.data.data_set.DataSet):
        if samplerate is None:
            samplerate = data.metadata.get('samplerate', None)

        data = np.array(data.default_parameter_array())

    if samplerate is None:
        raise ValueError('samplerate should be set to the data samplerate in Hz')

    # plotting a 2d histogram of the RTS
    if fig:
        xdata = np.array(range(0, len(data))) / samplerate * 1000
        Z, xedges, yedges = np.histogram2d(xdata, data, bins=[int(
            np.sqrt(len(xdata)) / 2), int(np.sqrt(len(data))/2) ])
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
        num_bins = int(np.sqrt(len(data)))

    fit_results = two_level_threshold(data, number_of_bins=num_bins)
    separation = fit_results['separation']
    split = fit_results['signal_threshold']
    double_gaussian_fit_parameters = fit_results['double_gaussian_fit']['parameters']

    if verbose:
        print('Fit parameters double gaussian:\n mean down: %.3f counts' %
              double_gaussian_fit_parameters[4] + ', mean up:%.3f counts' % double_gaussian_fit_parameters[
                  5] + ', std down: %.3f counts' % double_gaussian_fit_parameters[2] + ', std up:%.3f counts' %
              double_gaussian_fit_parameters[3])
        print('Separation between peaks gaussians: %.3f std' % separation)
        print('Split between two levels: %.3f' % split)

    # plotting the data in a histogram, the fitted two gaussian model and the split
    if fig:
        figure_title = 'Histogram of two levels RTS'
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

    fraction_down = np.sum(data < split) / data.size
    fraction_up = 1 - fraction_down

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

    parameters['down_segments'] = {'mean': np.mean(durations_dn_idx) / samplerate, 'p50': np.percentile(
        durations_dn_idx, 50) / samplerate, 'mean_filtered': np.mean(durations_dn_idx)}
    parameters['up_segments'] = {'mean': np.mean(durations_up_idx) / samplerate, 'p50': np.percentile(
        durations_up_idx, 50) / samplerate, 'mean_filtered': np.mean(durations_up_idx)}
    parameters['tunnelrate_down_to_up'] = 1. / parameters['down_segments']['mean']
    parameters['tunnelrate_up_to_down'] = 1. / parameters['up_segments']['mean']

    parameters['fraction_down'] = fraction_down
    parameters['fraction_up'] = fraction_up

    if (counts_dn[0] > minimal_count_number) and (counts_up[0] > minimal_count_number):

        def _fit_and_plot_decay(bincentres, counts, label, fig_label):
            """ Fitting and plotting of exponential decay for level """
            A_fit, B_fit, gamma_fit = fit_exp_decay(bincentres, counts)
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
                         'r', label=r'Fitted exponential decay $\Gamma_{\mathrm{%s\ to\ %s}}$: %.1f kHz' % (label, other_label, tunnelrate) )
                plt.xlabel('Lifetime (ms)')
                plt.ylabel('Counts per bin')
                plt.legend()
                plt.title(title)
                if ppt:
                    addPPTslide(title=title, fig=Fig)

            fit_parameters = [A_fit, B_fit, gamma_fit]
            return tunnelrate, fit_parameters

        bincentres_dn = np.array([(bins_dn[i] + bins_dn[i + 1]) / 2 for i in range(0, len(bins_dn) - 1)])
        fig_label = None if fig is None else fig + 2
        tunnelrate_dn, fit_parameters_down = _fit_and_plot_decay(bincentres_dn, counts_dn, label='down', fig_label=fig_label)

        bincentres_up = np.array([(bins_up[i] + bins_up[i + 1]) / 2 for i in range(0, len(bins_up) - 1)])
        fig_label = None if fig is None else fig + 3
        tunnelrate_up, fit_parameters_up = _fit_and_plot_decay(bincentres_up, counts_up, label='up', fig_label=fig_label)

        parameters['fit parameters exp. decay down'] = fit_parameters_down
        parameters['fit parameters exp. decay up'] = fit_parameters_up

        parameters['tunnelrate_down_exponential_fit'] = tunnelrate_dn
        parameters['tunnelrate_up_exponential_fit'] = tunnelrate_up
    else:
        parameters['tunnelrate_down_exponential_fit'] = None
        parameters['tunnelrate_up_exponential_fit'] = None

    return tunnelrate_dn, tunnelrate_up, parameters


def generate_RTS_signal(number_of_samples=100000, std_gaussian_noise=0.1,
                        uniform_noise=0.05, rate_up=10e3, rate_down=15e3, samplerate=1e6):
    """ Generate a RTS signal

    Args:
        number_of_samples (int): Length the the trace to be generated
        std_normal_noise (float): std of Gaussian noise added to the signal
        uniform_noise (float): uniform noise in the range +- uniform_noise/2 is added to the signal
        rate_up (float): rate from down to up
        rate_down (float): rate from up to down
        samplerate (float): The samplerate of the signal to be generated
    Returns:
        array: generated signal (0 is down, 1 is up)

    """

    rts_model = ContinuousTimeMarkovModel(['down', 'up'], [rate_up / samplerate,
                                                           rate_down / samplerate], np.array([[0., 1], [1, 0]]))

    data = rts_model.generate_sequence(number_of_samples, delta_time=1)

    if uniform_noise != 0:
        data = data + uniform_noise * (np.random.rand(data.size, ) - .5)
    if std_gaussian_noise != 0:
        data = data + np.random.normal(0, std_gaussian_noise, data.size)
    return data
