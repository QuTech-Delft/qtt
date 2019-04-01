""" Functionality to analyse random telegraph signals

Created on Wed Feb 28 10:20:46 2018

@author: riggelenfv
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import unittest
import qcodes
from qtt.utilities.tools import addPPTslide
import warnings

from qtt.algorithms.functions import double_gaussian, fit_double_gaussian, exp_function, fit_exp_decay
from qtt.algorithms.markov_chain import ContinuousTimeMarkovModel

# %% calculate durations of states


def transitions_durations(data, split):
    """ For data of a two level system (up and down), this funtion determines which datapoints belong to which level and finds the transitions, in order to determines
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


# %% function to analyse the RTS data
class FittingException(Exception):
    pass


def _plot_rts_histogram(data, num_bins, par_fit, split, figure_title):
    _, bins, _ = plt.hist(data, bins=num_bins)
    bincentres = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 1)])

    plt.plot(bincentres, double_gaussian(bincentres, par_fit), 'r', label='Fitted double gaussian')
    plt.plot(split, double_gaussian(split, par_fit), 'ro', markersize=8, label='split: %.3f' % split)
    plt.xlabel('Measured value (a.u.)')
    plt.ylabel('Data points per bin')
    plt.legend()
    plt.title(figure_title)


def tunnelrates_RTS(data, samplerate=None, min_sep=2.0, max_sep=7.0, min_duration=5,
                    num_bins=None, plungers=None, fig=None, ppt=None, verbose=0):
    """
    This function takes an RTS dataset, fits a double gaussian, finds the split between the two levels,
    determines the durations in these two levels, fits a decaying exponential on two arrays of durations,
    which gives the tunneling frequency for both the levels. If the number of datapoints is too low to get enough points per bin
    for the exponential fit (for either the up or the down level), this analysis step is passed over. tunnelrate_dn and tunnelrate_up
    are returned as None, but similar information can be substracted from parameters['down_segments'] and parameters['up_segments'].

    Args:
        data (array): qcodes DataSet (or 1d data array) with the RTS data
        plungers ([str, str]): array of the two plungers used to perform the RTS measurement
        samplerate (int or float): sampling rate of the acquisition device, optional if given in the metadata
                of the measured data
        min_sep (float): if the separation found for the fit of the double gaussian is less then this value, the fit probably failed
            and a FittingException is raised
        max_sep (float): if the separation found for the fit of the double gaussian is more then this value, the fit probably failed
            and a FittingException is raised
        min_duration (int): minimal number of datapoints a duration should last to be taking into account for the analysis
        fig (None or int): shows figures and sends them to the ppt when is not None
        ppt (None or int): determines if the figures are send to a powerpoint presentation
        verbose (int): prints info to the console when > 0

    Returns:
        tunnelrate_dn (numpy.float64): tunneling rate of the down level to the up level (kHz) or None in case of not enough datapoints
        tunnelrate_up (numpy.float64): tunneling rate of the up level to the down level (kHz) or None in case of not enough datapoints
        parameters (dict): dictionary with relevent (fit) parameters. this includes:
                tunnelrate_down (float): tunnel rate in Hz
                tunnelrate_up (float): tunnel rate up in Hz

    """
    if plungers is None:
        plungers = []

    if isinstance(data, qcodes.data.data_set.DataSet):
        try:
            plungers = plungers
            metadata = data.metadata
            gates = metadata['allgatevalues']
            plungervalue = gates[plungers[0]]
        except BaseException:
            plungervalue = []
        if samplerate is None:
            metadata = data.metadata
            samplerate = metadata['samplerate']

        data = np.array(data.measured)
    else:
        plungervalue = []
        if samplerate is None:
            raise Exception('samplerate is None')

    # plotting a 2d histogram of the RTS
    if fig:
        xdata = np.array(range(0, len(data))) / samplerate * 1000
        Z, xedges, yedges = np.histogram2d(xdata, data, bins=[int(
            np.sqrt(len(xdata))) / 2, int(np.sqrt(len(data))) / 2])
        title = '2d histogram RTS'
        plt.figure(title)
        plt.clf()
        plt.pcolormesh(xedges, yedges, Z.T)
        cb = plt.colorbar()
        cb.set_label('Data points per bin')
        plt.xlabel('Time (ms)')
        plt.ylabel('Signal sensing dot (a.u.)')
        plt.title(title)
        if ppt:
            addPPTslide(title=title, fig=plt.figure(title))

    # binning the data and determining the bincentres
    if num_bins is None:
        num_bins = int(np.sqrt(len(data)))
    counts, bins = np.histogram(data, bins=num_bins)
    bincentres = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 1)])

    # fitting the double gaussian and finding the split between the up and the
    # down state, separation between the max of the two gaussians measured in
    # the sum of the std
    par_fit, result_dict = fit_double_gaussian(bincentres, counts)
    separation = result_dict['separation']
    split = result_dict['split']

    if verbose:
        print('Fit parameters double gaussian:\n mean down: %.3f counts' %
              par_fit[4] + ', mean up:%.3f counts' % par_fit[5] + ', std down: %.3f counts' % par_fit[2] + ', std up:%.3f counts' % par_fit[3])
        print('Separation between peaks gaussians: %.3f std' % separation)
        print('Split between two levels: %.3f' % split)

    # plotting the data in a histogram, the fitted two gaussian model and the split
    if fig:
        figure_title = 'Histogram of two levels RTS'
        plt.figure(figure_title)
        plt.clf()
        _plot_rts_histogram(data, num_bins, par_fit, split, figure_title)
        if ppt:
            addPPTslide(title=title, fig=plt.figure(title), notes='Fit parameters double gaussian:\n mean down: %.3f counts' %
                        par_fit[4] + ', mean up:%.3f counts' % par_fit[5] + ', std down: %.3f counts' % par_fit[2] + ', std up:%.3f counts' % par_fit[3] + '.Separation between peaks gaussians: %.3f std' % separation + '. Split between two levels: %.3f' % split)

    if separation < min_sep:
        raise FittingException(
            'Separation between the peaks of the gaussian %.1f is less then %.1f std, indicating that the fit was not succesfull.' % (separation, min_sep))

    if separation > max_sep:
        raise FittingException(
            'Separation between the peaks of the gaussian %.1f is more then %.1f std, indicating that the fit was not succesfull.' % (separation, max_sep))

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

    def _create_histogram(durations, level, verbose=verbose):
        """ Calculate number of bins, bin edges and histogram for durations

        This method works if the data is sampled at integer durations.
        """
        numbins = int(np.sqrt(len(durations)))
        bin_size = int(np.ceil((durations.max() - durations.min()) / numbins))
        # choose bins carefully, since our data is sampled only an discrete times
        bins = np.arange(durations.min() - .5, durations.max() + bin_size, bin_size)
        counts, bins = np.histogram(durations, bins=bins)
        if verbose:
            print(' _create_histogram level ' + level + ': nbins %d, %d' % (numbins, bin_size))
        return counts, bins

    # calculating the number of bins and counts for down level
    counts_dn, bins_dn = _create_histogram(durations_dn, level='down')

    # calculating the number of bins and counts for up level
    counts_up, bins_up = _create_histogram(durations_up, level='up')

    # calculating durations in seconds
    durations_dn = durations_dn / samplerate
    durations_up = durations_up / samplerate

    bins_dn = bins_dn / samplerate
    bins_up = bins_up / samplerate

    if verbose >= 2:
        print('counts_dn %d, counts_up %d' % (counts_dn[0], counts_up[0]))

    tunnelrate_dn = None
    tunnelrate_up = None

    if counts_dn[0] < 50:
        tunnelrate_dn = None
        tunnelrate_up = None
        warnings.warn(
            'Number of down datapoints %d is not be enough to make an acurate fit of the exponential decay for level down.' % counts_dn[0] + 'Look therefore at the mean value of the measurement segments')

    if counts_up[0] < 50:
        tunnelrate_dn = None
        tunnelrate_up = None
        warnings.warn(
            'Number of up datapoints %d is not be enough to make an acurate fit of the exponential decay for level up.' % counts_dn[0] + 'Look therefore at the mean value of the measurement segments')

    parameters = {'plunger value': plungervalue, 'sampling rate': samplerate, 'fit parameters double gaussian': par_fit,
                  'separations between peaks gaussians': separation,
                  'split between the two levels': split}

    parameters['down_segments'] = {'mean': np.mean(durations_dn_idx) / samplerate, 'p50': np.percentile(
        durations_dn_idx, 50) / samplerate, 'mean_filtered': np.mean(durations_dn_idx)}
    parameters['up_segments'] = {'mean': np.mean(durations_up_idx) / samplerate, 'p50': np.percentile(
        durations_up_idx, 50) / samplerate, 'mean_filtered': np.mean(durations_up_idx)}
    parameters['tunnelrate_down_to_up'] = 1. / parameters['down_segments']['mean']
    parameters['tunnelrate_up_to_down'] = 1. / parameters['up_segments']['mean']

    if (counts_dn[0] > 50) and (counts_up[0] > 50):

        bincentres_dn = np.array([(bins_dn[i] + bins_dn[i + 1]) / 2 for i in range(0, len(bins_dn) - 1)])

        # fitting exponential decay for down level
        A_dn_fit, B_dn_fit, gamma_dn_fit = fit_exp_decay(bincentres_dn, counts_dn)
        tunnelrate_dn = gamma_dn_fit / 1000

        if verbose:
            print('Tunnel rate down: %.1f kHz' % tunnelrate_dn)

        time_scaling = 1e3

        if fig:
            title = 'Fitted exponential decay, level down'
            plt.figure(title)
            plt.clf()
            plt.plot(time_scaling * bincentres_dn, counts_dn, 'o', label='Counts down')
            plt.plot(time_scaling * bincentres_dn, exp_function(bincentres_dn, A_dn_fit, B_dn_fit, gamma_dn_fit),
                     'r', label=r'Fitted exponential decay $\Gamma_{\mathrm{down\ to\ up}}$: %.1f kHz' % tunnelrate_dn)
            plt.xlabel('Lifetime (ms)')
            plt.ylabel('Counts per bin')
            plt.legend()
            plt.title(title)
            if ppt:
                addPPTslide(title=title, fig=plt.figure(title))

        bincentres_up = np.array([(bins_up[i] + bins_up[i + 1]) / 2 for i in range(0, len(bins_up) - 1)])

        # fitting exponential decay for up level
        A_up_fit, B_up_fit, gamma_up_fit = fit_exp_decay(bincentres_up, counts_up)
        tunnelrate_up = gamma_up_fit / 1000

        if verbose:
            print('Tunnel rate up: %.1f kHz' % tunnelrate_up)

        if fig:
            title = 'Fitted exponential decay, level up'
            plt.figure(title)
            plt.clf()
            plt.plot(time_scaling * bincentres_up, counts_up, 'o', label='Counts up')
            plt.plot(time_scaling * bincentres_up, exp_function(bincentres_up, A_up_fit, B_up_fit, gamma_up_fit),
                     'r', label=r'Fitted exponential decay $\Gamma_{\mathrm{up\ to\ down}}$: %.1f kHz' % tunnelrate_up)
            plt.xlabel('Lifetime (ms)')
            plt.ylabel('Data points per bin')
            plt.legend()
            plt.title(title)
            if ppt:
                addPPTslide(title=title, fig=plt.figure(title))

        parameters['fit parameters exp. decay down'] = [A_dn_fit, B_dn_fit, gamma_dn_fit]
        parameters['fit parameters exp. decay up'] = [A_up_fit, B_up_fit, gamma_up_fit]

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


class TestRandomTelegraphSignal(unittest.TestCase):

    def test_RTS(self, fig=None):
        data = np.random.rand(10000, )
        try:
            _ = tunnelrates_RTS(data)
            raise Exception('no samplerate available')
        except Exception as ex:
            # exception is good, since no samplerate was provided
            self.assertTrue('samplerate is None' in str(ex))
        try:
            _ = tunnelrates_RTS(data, samplerate=10e6)
            raise Exception('data should not fit to RTS')
        except FittingException as ex:
            # fitting exception is good, since data is random
            pass

        data = generate_RTS_signal(100, std_gaussian_noise=0, uniform_noise=.1)
        data = generate_RTS_signal(100, std_gaussian_noise=0.1, uniform_noise=.1)

        samplerate = 2e6
        data = generate_RTS_signal(100000, std_gaussian_noise=0.1, rate_up=10e3, rate_down=20e3, samplerate=samplerate)

        with warnings.catch_warnings():  # catch any warnings
            warnings.simplefilter("ignore")
            tunnelrate_dn, tunnelrate_up, parameters = tunnelrates_RTS(data, samplerate=samplerate, fig=fig)

            self.assertTrue(parameters['up_segments']['mean'] > 0)
            self.assertTrue(parameters['down_segments']['mean'] > 0)

        samplerate = 1e6
        rate_up = 200e3
        rate_down = 20e3
        data = generate_RTS_signal(100000, std_gaussian_noise=0.01, rate_up=rate_up,
                                   rate_down=rate_down, samplerate=samplerate)

        tunnelrate_dn, tunnelrate_up, _ = tunnelrates_RTS(data, samplerate=samplerate, min_sep=1.0, max_sep=2222,
                                                                min_duration=1, num_bins=40, fig=fig, verbose=2)

        self.assertTrue(np.abs(tunnelrate_dn - rate_up * 1e-3) < 100)
        self.assertTrue(np.abs(tunnelrate_up - rate_down * 1e-3) < 10)
