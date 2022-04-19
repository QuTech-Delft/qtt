""" Functions to fit and analyse Coulomb peaks """

import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.optimize as opt

import qtt.data
import qtt.measurements.scans
import qtt.pgeometry as pgeometry
from qtt.algorithms.functions import logistic
from qtt.algorithms.generic import issorted, nonmaxsuppts

# %% Functions related to detection of Coulumb peaks


def gauss(x, p):
    """ Gaussian function with parameters

    Args:
        x (array or float): input variable
        p (array): parameters [mean, std. dev., amplitude]
    Returns:
        array or float: calculated Gaussian
    """
    return p[2] * 1.0 / (p[1] * np.sqrt(2 * np.pi)) * np.exp(-(x - p[0]) ** 2 / (2 * p[1] ** 2))


def analyseCoulombPeaks(all_data, fig=None, verbose=1, parameters=None):
    """ Find Coulomb peaks in a 1D dataset

    Args:
        all_data (DataSet): The data to analyse.
        fig (int or None): Figure handle to the plot.
        parameters (dict): dictionary with parameters that is passed to subfunctions

    Returns:
        peaks (list): fitted peaks
    """
    x_data, y_data = qtt.data.dataset1Ddata(all_data)
    return analyseCoulombPeaksArray(x_data, y_data, fig=fig, verbose=verbose, parameters=parameters)


def analyseCoulombPeaksArray(x_data, y_data, fig=None, verbose=1, parameters=None):
    """ Find Coulomb peaks in arrays of data. This is very similar to analyseCoulombPeaks,
        but takes arrays of data as input. Hence the y_data can for example be either the
        I, Q or any combination of both obtained with RF reflectometry.

    Args:
        x_data (1D array): The data of varied parameter.
        y_data (1D array): The signal data.
        fig (None or int): figure handle
        parameters (dict): dictionary with parameters that is passed to subfunctions

    Returns:
        (list of dict): The detected peaks.
    """
    if parameters is None:
        parameters = {}
    sampling_rate = parameters.get('sampling_rate', (x_data[-1] - x_data[0]) / (x_data.size - 1))
    return coulombPeaks(x_data, y_data, verbose=verbose, fig=fig, plothalf=True, sampling_rate=sampling_rate,
                        parameters=parameters)


def fitCoulombPeaks(x_data, y_data, lowvalue=None, verbose=1, fig=None, sampling_rate=1):
    """ Fit Coulumb peaks in a measurement series.

    Args:
        x_data (1D array): The data of varied parameter.
        y_data (1D array): The signal data.
        sampling_rate (float): The sampling rate in mV/pixel.
        lowvalue (float or None): If None, select the 1st percentile of the dependent variable
        
    Returns:
        (list): A list with detected peaks.
    """
    p1, p5, p95, p99 = np.percentile(y_data, [1, 5, 95, 99])
    minval = p5 + 0.1 * (p95 - p5)
    local_maxima, _ = nonmaxsuppts(y_data, d=int(12 / sampling_rate), minval=minval)
    fit_data = fitPeaks(x_data, y_data, local_maxima, verbose=verbose >= 2, fig=fig)

    if lowvalue is None:
        lowvalue = p1
    highvalue = p99
    peaks = []
    for ii, f in enumerate(fit_data):
        p = local_maxima[ii]
        peak = dict({'p': p, 'x': x_data[p], 'y': y_data[p], 'gaussfit': f})
        peak['halfvaluelow'] = (y_data[p] - lowvalue) / 2 + lowvalue
        peak['height'] = (y_data[p] - lowvalue)
        if peak['height'] < .1 * (highvalue - lowvalue):
            peak['valid'] = 0
        else:
            peak['valid'] = 1
        peak['lowvalue'] = lowvalue
        peak['type'] = 'peak'
        peaks.append(peak)

        if verbose:
            print('fitCoulombPeaks: peak %d: position %.2f max %.2f valid %d' %
                  (ii, peak['x'], peak['y'], peak['valid']))
    return peaks


# %%


def plotPeaks(x, y, peaks, showPeaks=True, plotLabels=False, fig=10, plotScore=False, plotsmooth=True, plothalf=False,
              plotbottom=False, plotmarker='.-b'):
    """ Plot detected peaks

    Args:
        x (array): independent variable data
        y (array): dependent variable data
        peaks (list): list of peaks to plot
        showPeaks, plotLabels, plotScore, plothalf (bool): plotting options

    Returns:
        dictionary: graphics handles

    """
    kk = np.ones(3) / 3.
    ys = scipy.ndimage.correlate1d(y, kk, mode='nearest')
    stdY = np.std(y)
    pgeometry.cfigure(fig)
    plt.clf()
    h = plt.plot(x, y, plotmarker)
    if plotsmooth:
        plt.plot(x, ys, 'g')
    labelh = []
    first_label = True
    for jj, peak in enumerate(peaks):
        if peak['valid']:
            xpos = peak['x']
            ypos = peak['y']

            if showPeaks:
                label = 'peaks' if first_label else None
                first_label = False
                plt.plot(xpos, ypos, '.r', markersize=15, label=label)
            if plotLabels:
                tp = peak.get('type', 'peak')
                lbl = '%s %d' % (tp, jj)
                if plotScore:
                    lbl += ': score %.1f ' % peak['score']
                lh = plt.text(xpos, ypos + .05 * stdY, lbl)
                labelh += [lh]
    halfhandle = []
    if plothalf:
        first_label = True
        for peak in peaks:
            if 'xhalfl' in peak:
                label = 'peak at half height' if first_label else None
                first_label = False

                hh = plt.plot(peak['xhalfl'], peak['yhalfl'], '.m',
                              markersize=12, label=label)
                halfhandle += [hh[0]]
    if plotbottom:
        for peak in peaks:
            if peak['valid']:
                plt.plot(
                    peak['xbottoml'], peak['ybottoml'], '.', color=[.5, 1, .5], markersize=12, label='peak bottom left')

    th = plt.title('Fitted peaks')
    return dict({'linehandle': h, 'title': th, 'labelh': labelh, 'halfhandle': halfhandle})


# %%


def filterPeaks(x, y, peaks, verbose=1, minheight=None):
    """ Filter the detected peaks

    Args:
        x (array): independent variable data
        y (array): signal data
        peaks (list): list of peaks

    Returns:
        list : selected good peaks

    Filtering criteria are:

        - minimal height
        - overlapping of peaks

    """
    peaks = [peak for peak in peaks if peak['valid']]
    peaks.sort(key=lambda x: x['p'])

    ww = copy.copy([peak['p'] for peak in peaks])
    goodpeaks = copy.deepcopy(peaks)

    lowheights = [peak['y'] - peak['lowvalue'] for peak in goodpeaks]
    heights = [peak['y'] - peak['ybottoml'] for peak in goodpeaks]

    # minheight: None: automatic threshold
    if minheight is None:
        if len(heights) > 0:
            minheight = np.max(heights) * .1
        else:
            minheight = 0

    if verbose >= 2:
        print('filterPeaks: minheight %.2f' % minheight)

    # filter on peak height
    for ii, peak in enumerate(goodpeaks):
        if peak['y'] - peak['ybottoml'] < minheight:
            if peak['valid']:
                peak['valid'] = False
                peak['validreason'] = 'not high enough'
            # print('x')
        pass

    # filter on overlap
    for ii, peak in enumerate(goodpeaks):
        xx = (peak['pbottom'] > ww).sum()
        peak['score'] = peak['y'] - peak['vbottom']
        if verbose >= 3:
            print('filterPeaks: peak %d, xx %d' % (ii, xx))
        if xx < ii:
            if verbose >= 2:
                print('filterPeaks: peak %d: invalid peak' % ii)
            peak['valid'] = 0
    goodpeaks = [peak for peak in goodpeaks if peak['valid']]

    goodpeaks.sort(key=lambda x: -x['score'])

    if verbose:
        ngoodpeaks = len([p for p in goodpeaks if p['valid']])
        print('filterPeaks: %d -> %d good peaks' % (len(peaks), ngoodpeaks))
    return goodpeaks


# %%


def peakFindBottom(x, y, peaks, fig=None, verbose=1):
    """ Find the left bottom of a detected peak

    Args:
        x (array): independent variable data
        y (array): signal data
        peaks (list): list of detected peaks
        fig (None or int): if integer, then plot results
        verbose (int): verbosity level
    """
    kk = np.ones(3) / 3.
    ys = scipy.ndimage.correlate1d(y, kk, mode='nearest')
    peaks = copy.deepcopy(peaks)

    dy = np.diff(ys, n=1)
    dy = np.hstack((dy, [0]))
    kernel_size = [int(np.max([2, dy.size / 100])), ]
    dy = qtt.algorithms.generic.boxcar_filter(dy, kernel_size=kernel_size)
    for ii, peak in enumerate(peaks):
        if verbose:
            print('peakFindBottom: peak %d' % ii)

        if not peak['valid']:
            continue
        ind = range(peak['phalf0'])

        left_of_peak = 0 * y.copy()
        left_of_peak[ind] = 1
        r = range(y.size)
        left_of_peak_and_decreasing = left_of_peak * (dy < 0)  # set w to zero where the scan is increasing
        left_of_peak_and_decreasing[0] = 1  # make sure to stop at the left end of the scan...

        ww = left_of_peak_and_decreasing.nonzero()[0]
        if verbose >= 2:
            print('  peakFindBottom: size of decreasing area %d' % ww.size)

        if ww.size == 0:
            if peak['valid']:
                peak['valid'] = 0
                peak['validreason'] = 'peakFindBottom'
                if verbose >= 2:
                    print('peakFindBottom: invalid peak')
                    print(ind)
                    print(dy)
            continue
        bidx = ww[-1]
        peak['pbottomlow'] = bidx

        w = left_of_peak * (dy > 0)  # we need to be rising
        # we need to be above 10% of absolute low value
        w = w * (ys < ys[bidx] + .1 * (ys[peak['p']] - ys[bidx]))
        w = w * (r >= peak['pbottomlow'])
        ww = w.nonzero()[0]
        if ww.size == 0:
            if peak['valid']:
                peak['valid'] = 0
                peak['validreason'] = 'peakFindBottom'
                if verbose >= 2:
                    print('peakFindBottom: invalid peak (%s)' % ('rising part ww.size == 0',))
                    print(w)
                    print(ys)
            continue
        bidx = ww[-1]

        peak['pbottom'] = bidx
        peak['pbottoml'] = bidx
        peak['xbottom'] = x[bidx]
        peak['xbottoml'] = x[bidx]
        peak['vbottom'] = y[bidx]  # legacy
        peak['ybottoml'] = y[bidx]

        if verbose >= 3:
            plt.figure(53)
            plt.clf()
            plt.plot(x[ind], 0 * np.array(ind) + 1, '.b', label='ind')
            plt.plot(x[range(y.size)], w, 'or', label='w')
            plt.plot(x[range(y.size)], dy < 0, 'dg',
                     markersize=12, label='dy<0')
            pgeometry.enlargelims()
            pgeometry.plot2Dline([-1, 0, peak['x']], '--c', label='x')
            pgeometry.plot2Dline([-1, 0, x[peak['phalf0']]],
                                 '--y', label='phalf0')

            pgeometry.plot2Dline([-1, 0, x[peak['pbottomlow']]],
                                 ':k', label='pbottomlow')

            pgeometry.plot2Dline([-1, 0, peak['xbottoml']],
                                 '--y', label='xbottoml')

            plt.legend(loc=0)

    return peaks


# %%


def fitPeaks(XX, YY, points, fig=None, verbose=0):
    """ Fit Gaussian model on local maxima

    Args:
        XX (array): independent variable data
        YY (array): dependent variable data
        points (list): indices of points to fit
        fig (None or int): if int, plot results
        verbose (int): verbosity level
     Returns:
         fit_data (array): for each point the fitted Gaussian
     """
    points = np.array(points)
    fit_data = np.zeros((points.size, 3))
    for ii, point_index in enumerate(points):
        if verbose:
            print('fitPeaks: peak at %.1f %.1f' % (XX[point_index], YY[point_index]))
        # fixme: boundaries based on mV
        fit_range = [point_index - 30, point_index + 30]
        fit_range[1] = np.minimum(fit_range[1], XX.size - 1)
        fit_range[0] = np.maximum(fit_range[0], 0)
        X = XX[fit_range[0]:(fit_range[1] + 1)]
        Y = YY[fit_range[0]:(fit_range[1] + 1)]
        sel = range(fit_range[0], fit_range[1] + 1)
        point_sub_index = sel.index(point_index)

        # Fit a Gaussian
        p0 = [X[point_sub_index], 1, 2 * Y[point_sub_index]]  # Initial guess is a normal distribution

        # Distance to the target function

        def errfunc(p, x, y):
            return gauss(x, p) - y

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gaussian_parameters, _ = opt.leastsq(errfunc, p0[:], args=(X, Y), maxfev=1800)

        _, fit_stdev, _ = gaussian_parameters

        FWHM = 2 * np.sqrt(2 * np.log(2)) * fit_stdev
        if verbose:
            print("fitPeaks: peak %d: FWHM %.3f" % (ii, FWHM))
        fit_data[ii, :] = gaussian_parameters

    if fig:
        plt.figure(fig)
        plt.clf()
        plt.plot(XX, YY, '-b', label='data')
        for ii, point_index in enumerate(points):
            gaussian_parameters = fit_data[ii, :]
            fit_mu, fit_stdev, _ = gaussian_parameters

            FWHM = 2 * np.sqrt(2 * np.log(2)) * fit_stdev
            plt.plot(XX, gauss(XX, gaussian_parameters), lw=3, alpha=.5, color='r', label='point %d' % ii)
            plt.axvspan(
                fit_mu - FWHM / 2, fit_mu + FWHM / 2, facecolor='g', alpha=0.25)
        plt.show()
    return fit_data


# %%


def peakScores(peaksx, x, y, hwtypical=10, verbose=1, fig=None):
    """ Calculate scores for list of peaks

    Args:
        x (array): independent variable data
        y (array): dependent variable data
        peaksx (list): list with detected peaks

    """
    lowvalue, highvalue = np.percentile(y, [5, 99])

    kk = np.ones(3) / 3.
    ys = y
    for ki in range(8):
        ys = scipy.ndimage.correlate1d(ys, kk, mode='nearest')
    noise = np.std(ys - y)
    stn2 = np.log2((highvalue - lowvalue) / noise)

    if fig is not None:
        plotPeaks(x, y, peaksx, plotLabels=True, fig=fig)
        plt.plot(x, ys, '-g')

    noisefac = logistic(stn2, x0=3.5, alpha=1)
    if verbose:
        print('peakScores: noise factor %.2f' % noisefac)

    for ii, peak in enumerate(peaksx):
        if not peak['valid']:
            peak['score'] = 0
            continue
        h = peak['height']  # original height
        h = peak['y'] - peak['ybottoml']  # diff between top and bottom left

        h2 = 2 * (peak['y'] - peak['yhalfl'])
        if (h2 / h) < .3:
            # special case
            h = (h2 + h) / 2

        hw = peak['p'] - peak['pbottom']
        hw = np.abs(x[peak['p']] - peak['xhalfl'])
        vv = peak['phalf0']
        slope1 = (y[vv + 1] - y[vv]) / (x[vv + 1] - x[vv])
        slope2 = (y[vv] - y[vv - 1]) / (x[vv] - x[vv - 1])
        slope1 = (ys[vv + 1] - ys[vv]) / (x[vv + 1] - x[vv])
        slope2 = (ys[vv] - ys[vv - 1]) / (x[vv] - x[vv - 1])
        peak['slope'] = (slope1 + slope2) / 2

        peak['heightscore'] = h / (highvalue - lowvalue)
        peak['score'] = h * (2 / (1 + hw / hwtypical))
        peak['scorerelative'] = (
            h / (highvalue - lowvalue)) * (2 / (1 + hw / hwtypical))
        peak['noisefactor'] = noisefac
        if verbose:
            print('peakScores: %d: height %.1f halfwidth %.1f, score %.2f' %
                  (ii, h, hw, peak['score']))
            if verbose >= 2:
                print('   slope: %.1f, heightscore %.2f, score %.2f' %
                      (peak['slope'], peak['heightscore'], peak['score']))


# %%


def analysePeaks(x, y, peaks, verbose=1, doplot=0, typicalhalfwidth=None, parameters=None, istep=None):
    """ Analyse Coulomb peaks

    Args:
        x,y: arrays with data
            data in mV
        peaks: list of detected peaks to be analysed
        typicalhalfwidth : float
            typical width of peak (half side) in mV (mV ??)
    """
    if parameters is None:
        parameters = {}
    if istep is not None:
        warnings.warn('ignoring legacy argument istep')

    if typicalhalfwidth is not None:
        raise Exception('please set typicalhalfwidth in the parameters argument')

    typicalhalfwidth = parameters.get('typicalhalfwidth', 13)

    if not issorted(x):
        pass
    if x[0] > x[-1]:
        print('analysePeaks: warning: x values are not sorted!!!!')

    leftp = max(3, x.size / 200)  # ignore all data to the left of this point

    for ii, peak in enumerate(peaks):
        p = peak['p']
        if verbose:
            print('analysePeaks: peak %d: max %.1f' % (ii, peak['y']))

        if p < leftp:
            # discard all measurements to the left of the scan
            peak['valid'] = 0
            peak['xhalf'] = np.NaN
            peak['xhalfl'] = np.NaN
            peak['phalf'] = np.NaN
            continue

        # determine starting points for search of peak
        zi = np.interp(
            [x[p] - 3. * typicalhalfwidth, x[p], x[p] + 3. * typicalhalfwidth], x, range(x.size))
        zi = np.round(zi).astype(int)

        zi[0] = max(zi[0], leftp)  # discard points on the left of scan
        zi[1] = max(zi[1], leftp)

        if doplot >= 2:
            plt.plot(x[zi[0]], y[zi[0]], '.g', markersize=11, label='mu-3*thw')
            plt.plot(
                x[zi[-1]], y[zi[-1]], '.', color=[0, .27, 0], markersize=11, label='mu+3*thw')

        ind = range(zi[0], zi[1])
        if len(ind) == 0:
            if verbose >= 2:
                print('analysePeaks: error? x[p] %f' % x[p])
            peak['valid'] = 0
            peak['xhalf'] = np.NaN
            peak['phalf'] = np.NaN
            continue
        if verbose >= 2:
            print('  peak %d: range to search for half width %d to %d' % (ii, zi[0], zi[1]))
        xl, yl = x[ind], y[ind]

        hminval0 = .5 * (peak['y'] + np.min(y[ind]))
        fv = peak['y'] - 1.8 * (peak['y'] - hminval0)

        xh = np.interp(hminval0, yl, xl)
        ph = np.interp(hminval0, yl, range(xl.size))
        ph0 = ind[int(ph)]
        xf = np.interp(fv, yl, xl)
        peak['phalf0'] = ph0
        peak['phalfl'] = None

        phalfvalue = np.interp(ph, range(xl.size), yl)
        yhalfl = np.interp(ph, range(xl.size), yl)
        peak['xhalfl'] = xh
        peak['xfoot'] = xf
        peak['yhalfl'] = yhalfl

        if doplot >= 2:
            plt.plot(
                peak['xhalfl'], yhalfl, '.', color=[1, 1, 0], markersize=11)
            pgeometry.plot2Dline(
                [-1, 0, x[p] - 3 * typicalhalfwidth], ':c', label='3*thw')
            pgeometry.plot2Dline([-1, 0, peak['xfoot']], ':y', label='xfoot')

        pratio = np.abs(
            phalfvalue - peak['y']) / (-peak['halfvaluelow'] + peak['y'])
        if verbose >= 2:
            print('  paratio %.2f' % pratio)

        if verbose >= 2:
            print(
                np.abs(phalfvalue - peak['y']) / (-peak['halfvaluelow'] + peak['y']))
        if pratio > .1:
            peak['valid'] = peak['valid']
        else:
            peak['valid'] = 0

        if verbose:
            print('   peak %d: valid %d' % (ii, peak['valid']))
    return peaks


def peakdataOrientation(x, y):
    """ For measured 1D scans order the data such that the independent variable is ordered

    Args:
        x (array): independent variable data
        y (array): dependent variable data
    Returns:
        x,y (array): reordered data
    """
    i = np.argsort(x)
    x = x[i]
    y = y[i]
    return x, y


def sort_peaks_inplace(peaks):
    """ Sort a list of peaks according to the score field """
    peaks.sort(key=lambda x: -x['score'])


def sort_peaks(peaks):
    """ Sort a list of peaks according to the score field """
    peaks_sorted = sorted(peaks, key=lambda p: -p['score'])
    return peaks_sorted


def coulombPeaks(x_data, y_data, verbose=1, fig=None, plothalf=False, sampling_rate=None, parameters=None):
    """ Detects the Coulumb peaks in a 1D scan.

    Args:
        x_data, y_data (arrays): indep
        verbose (int)
        fig (int or None)
        sampling_rate (float or None): The sampling rate in in mV/pixel.
        parameters (dict): parameters passed to subfunctions
    """
    if parameters is None:
        parameters = {}

    if sampling_rate is None:
        warnings.warn('sampling_rate is None, please add sampling_rate as a parameter')

    x, y = peakdataOrientation(x_data, y_data)
    peaks = fitCoulombPeaks(x, y, verbose=verbose, fig=None, sampling_rate=sampling_rate)
    peaks = analysePeaks(x, y, peaks, verbose=verbose >= 2, doplot=0, parameters=parameters)
    peaks = peakFindBottom(x, y, peaks, verbose=verbose >= 2)
    goodpeaks = filterPeaks(x, y, peaks, verbose=verbose)

    peakScores(goodpeaks, x, y, verbose=verbose)
    sort_peaks_inplace(goodpeaks)

    if fig:
        plotPeaks(x, y, goodpeaks, fig=fig, plotLabels=True, plothalf=plothalf)
    return goodpeaks


# %% Find best slope


def _intervalOverlap(a, b):
    """ Return overlap between two intervals

    Args:
        a (list or tuple): first interval
        b (list or tuple): second interval
    Returns:
        float: overlap between the intervals

    Example:
    >>> _intervalOverlap( [0,2], [0,1])
    1
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def findBestSlope(x, y, minimal_derivative=None, fig=None, verbose=1):
    """ Find good slopes to use in sensing dot

    Args:
        x (array): independent variable data
        y (array): dependent variable data
        minimal_derivative (None or float): minimal derivative

    Returns:
        slopes (...)
        results (object): additional data
    """
    lowvalue, highvalue = np.percentile(y, [1, 99])

    H = highvalue - lowvalue

    if minimal_derivative is None:
        minimal_derivative = (highvalue - lowvalue) / 100

    k = np.array([1, 0, -1])
    dy = scipy.ndimage.convolve(y, k, mode='nearest')

    labeled_array, num_features = scipy.ndimage.label(
        dy >= minimal_derivative)

    slopes = []
    for ii in range(num_features):
        # calculate score for region
        rr = labeled_array == (ii + 1)
        vv = rr.nonzero()[0]
        p = vv.max()
        if p < x.size - 1:
            if y[p + 1] > y[p]:
                p = p + 1
        pbottom = vv.min()

        if p - pbottom <= 1:
            continue
        slope = dict({'pbottom': pbottom, 'xbottom': x[pbottom], 'xbottoml': x[pbottom], 'ybottoml': y[
            pbottom], 'x': x[p], 'y': y[p], 'p': p, 'lowvalue': lowvalue})

        slope['halfvalue'] = (slope['y'] + slope['ybottoml']) / 2
        halfvalue = slope['halfvalue']
        phalfl = int(np.round((p + pbottom) / 2))
        slope['phalfl'] = phalfl
        slope['phalf0'] = phalfl
        slope['yhalfl'] = y[slope['phalfl']]
        slope['xhalfl'] = x[slope['phalfl']]

        ind = np.arange(pbottom, p + 1)
        xl, yl = x[ind], y[ind]  # local coordinates

        xhalfl = np.interp(slope['halfvalue'], yl, xl)
        phlocal = np.interp(slope['halfvalue'], yl, range(xl.size))
        phl = np.interp(phlocal, range(ind.size), ind)

        slope['yhalfl'] = halfvalue
        slope['xhalfl'] = xhalfl
        slope['phalfl'] = phl
        slope['phalf0'] = int(phl)

        #slope['length'] = x - x[pbottom]
        slope['height'] = y[p] - y[pbottom]
        slope['valid'] = (slope['height'] / H) > .2
        slope['type'] = 'slope'

        if (p - pbottom <= 3) and ((slope['height'] / H) < .3):
            slope['valid'] = 0

        slopes.append(slope)

    if fig is not None:

        plt.figure(fig)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(x[:], y, '.-b')
        if len(slopes) > 0:
            slope = slopes[0]
            vv = np.arange(slope['pbottom'], slope['p'])
            plt.plot(x[vv], y[vv], '.r')
        plt.title('findBestSlopes: first slope')
        plt.subplot(2, 1, 2)
        plt.plot(x[:], dy, '.-b')
        plt.ylabel('Derivative')
        qtt.pgeometry.plot2Dline([0, -1, minimal_derivative], '--r', label='Minimum derivative')
        plt.title('findBestSlopes: derivative')

    peakScores(slopes, x, y, verbose=verbose)
    return slopes, (dy, minimal_derivative)


def findSensingDotPosition(x, y, verbose=1, fig=None, plotLabels=True, plotScore=True, plothalf=False, useslopes=True,
                           sampling_rate=None, parameters=None):
    """ Find best position for sensing dot

    Arguments:
        x,y (array): data
    verbose (int): output level

    Returns:
        goodpeaks (list): list of detected positions

    """
    goodpeaks = coulombPeaks(x, y, verbose=verbose, fig=None,
                             sampling_rate=sampling_rate, plothalf=False, parameters=parameters)

    if len(goodpeaks) == 0 or useslopes:
        slopes, (dy, minder) = findBestSlope(x, y, fig=None, verbose=verbose >= 2)
        goodpeaks = goodpeaks + slopes

    goodpeaks = filterOverlappingPeaks(goodpeaks, verbose=verbose >= 2)

    if fig:
        plotPeaks(x, y, goodpeaks, fig=fig, plotLabels=plotLabels,
                  plotScore=plotScore, plothalf=plothalf)
    return goodpeaks


def peakOverlap(p1, p2):
    """ Calculate overlap between two peaks or slopes

    Args:
        p1 (dict): Peak object
        p2 (dict): Peak object

    Returns:
        float: A number representing the amount of overlap. 0: no overlap, 1: complete overlap

    """
    a1 = p1['xbottoml'] - p1['x']
    a2 = p2['xbottoml'] - p2['x']
    o = _intervalOverlap([p1['xbottoml'], p1['x']], [p2['xbottoml'], p2['x']])
    s = (1 + o) / (1 + np.sqrt(a1 * a2))
    return s


def filterOverlappingPeaks(goodpeaks, threshold=.6, verbose=0):
    """ Filter peaks based on overlap """
    pp = sorted(goodpeaks, key=lambda p: p['x'])

    gidx = []
    rr = iter(range(0, len(pp)))
    for jj in rr:
        p1 = pp[jj]
        if jj == len(pp) - 1:
            gidx.append(jj)
            continue
        p2 = pp[jj + 1]
        s = peakOverlap(p1, p2)

        if s > threshold:
            if p1['score'] > p2['score']:
                gidx.append(jj)
                next(rr)  # rr.next()
            else:
                pass
        else:
            gidx.append(jj)

    if verbose:
        print('filterOverlappingPeaks: %d -> %d peaks' % (len(pp), len(gidx)))
    pp = [pp[jj] for jj in gidx]
    pp = sort_peaks(pp)
    return pp
