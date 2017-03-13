import numpy as np
import qcodes

from qtt.scans import scan1D
import qtt.data

import qtt.pgeometry as pmatlab
import matplotlib.pyplot as plt
from qtt.algorithms.generic import *
from qtt.algorithms.generic import issorted
from qtt.algorithms.functions import logistic

#%%

import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import skimage
import numpy as np

from qtt.algorithms.generic import localMaxima

#%%

#%% Functions related to detection of Coulumb peaks

import scipy.optimize as opt

# http://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak


def gauss(x, p):
    """ Gaussian function with parameters
        p[0]==mean, p[1]==stdev, p[2]=scale
    """
    return p[2] * 1.0 / (p[1] * np.sqrt(2 * np.pi)) * np.exp(-(x - p[0])**2 / (2 * p[1]**2))


def show1Dpeaks(alldata, fig=100, verbose=1):
    x = alldata['data_array'][:, 0]
    y = alldata['data_array'][:, 2]
    x, y = peakdataOrientation(x, y)

    istep = np.abs(alldata['sweepdata']['step'])
    goodpeaks = findSensingDotPosition(x, y, verbose=1, fig=fig, istep=istep, plotLabels=True, plotScore=True, plothalf=False, useslopes=True)

    return goodpeaks


def fitCoulombPeaks(x, y, lowvalue=None, verbose=1, fig=None, istep=1):
    """ Fit Coulumb peaks in a measurement series 

    Arguments
    ---------
        x, y - array
            data series
        istep - float
            sampling rate in [mV/pixel]


    """
    minval = np.percentile(
        y, 5) + .1 * (np.percentile(y, 95) - np.percentile(y, 5))
    pt, w = nonmaxsuppts(y, d=int(12 / istep), minval=minval)
    fp = fitPeaks(x, y, pt, verbose=verbose >= 2, fig=fig)

    if lowvalue == None:
        lowvalue = np.percentile(y, 1)
    highvalue = np.percentile(y, 99)
    peaks = []
    for ii, f in enumerate(fp):
        p = pt[ii]
        peak = dict(
            {'p': p, 'x': pt[ii], 'x': x[p], 'y': y[p], 'gaussfit': f})
        if verbose:
            print('fitCoulombPeaks: peak %d: position %.2f max %.1f' %
                  (ii, peak['x'], peak['y']))

        peak['halfvaluelow'] = (y[p] - lowvalue) / 2 + lowvalue
        peak['height'] = (y[p] - lowvalue)
        if peak['height'] < .1 * (highvalue - lowvalue):
            peak['valid'] = 0
        else:
            peak['valid'] = 1
        peak['lowvalue'] = lowvalue
        peak['type'] = 'peak'
        peaks.append(peak)
    return peaks

#%%


def plotPeaks(x, y, peaks, showPeaks=True, plotLabels=False, fig=10, plotScore=False, plotsmooth=True, plothalf=False, plotbottom=False, plotmarker='.-b'):
    """ Plot detected peaks


    Arguments
    ---------
        x,y : numpy arrays
            scandata
        peaks : list
            list of peaks to plot
        showPeaks, plotLabels, plotScore, plothalf : boolean
            plotting options
    Returns
    -------
        handles : dictionary
            graphics handles

    """
    kk = np.ones(3) / 3.
    ys = scipy.ndimage.filters.correlate1d(y, kk, mode='nearest')
    stdY = np.std(y)
    pmatlab.cfigure(fig)
    plt.clf()
    h = plt.plot(x, y, plotmarker)
    if plotsmooth:
        plt.plot(x, ys, 'g')
    # plt.plot(x,w)
    labelh = []
    for jj, peak in enumerate(peaks):
        if peak['valid']:
            p = peak['p']
            xpos = peak['x']
            ypos = peak['y']

            if showPeaks:
                plt.plot(xpos, ypos, '.r', markersize=15)
            if plotLabels:
                tp = peak.get('type', 'peak')
                lbl = '%s %d' % (tp, jj)
                if plotScore:
                    lbl += ': score %.1f ' % peak['score']
                # print('plot label!')
                lh = plt.text(xpos, ypos + .05 * stdY, lbl)
                labelh += [lh]
    halfhandle = []
    if plothalf:
        for p in peaks:
            hh = plt.plot(p['xhalfl'], p['yhalfl'], '.m', markersize=12)
            halfhandle += [hh[0]]
    if plotbottom:
        for p in peaks:
            if p['valid']:
                plt.plot(
                    p['xbottoml'], p['ybottoml'], '.', color=[.5, 1, .5], markersize=12)

    th = plt.title('Local maxima')
    return dict({'linehandle': h, 'title': th, 'labelh': labelh, 'halfhandle': halfhandle})

#%%


def filterPeaks(x, y, peaks, verbose=1, minheight=None):
    """ Filter the detected peaks

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
        print('filterPeaks: minheight %.2f' % (minheight))

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


def peakFindBottom(x, y, peaks, fig=None, verbose=1):
    """ Find the left bottom of a detected peak """
    kk = np.ones(3) / 3.
    ys = scipy.ndimage.filters.correlate1d(y, kk, mode='nearest')
    peaks = copy.deepcopy(peaks)

    dy = np.diff(ys, n=1)
    dy = np.hstack((dy, [0]))
    for ii, peak in enumerate(peaks):
        if verbose:
            print('peakFindBottom: peak %d' % ii)

        if not peak['valid']:
            continue
        ind = range(peak['phalf0'])

        w0 = 0 * y.copy()
        w0[ind] = 1
        r = range(y.size)
        w = w0 * (dy < 0)  # set w to zero where the scan is increasing
        w[0] = 1  # make sure to stop at the left end of the scan...

        ww = w.nonzero()[0]
        if verbose >= 2:
            print('  peakFindBottom: ww.size %d' % ww.size)

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

        w = w0 * (dy > 0)   # we need to be rising
        w = w * ((ys) < ys[bidx] + .1 * (ys[peak['p']] - ys[bidx]))  # we need to be above 10% of absolute low value
        w = w * (r >= peak['pbottomlow'])
        ww = w.nonzero()[0]
        if ww.size == 0:
            if peak['valid']:
                peak['valid'] = 0
                peak['validreason'] = 'peakFindBottom'
                if verbose >= 2:
                    print('peakFindBottom: invalid peak')
                    print(w)
                    print(ys)
 #                   print(w)
            continue
        bidx = ww[-1]

        peak['pbottom'] = bidx
        peak['pbottoml'] = bidx
        peak['xbottom'] = x[bidx]
        peak['xbottoml'] = x[bidx]
        peak['vbottom'] = y[bidx]  # legacy
        peak['ybottoml'] = y[bidx]

        if verbose >= 3:
            # for debugging
            plt.figure(53)
            plt.clf()
            plt.plot(x[ind], 0 * np.array(ind) + 1, '.b', label='ind')
            plt.plot(x[range(y.size)], w, 'or', label='w')
            plt.plot(x[range(y.size)], dy < 0, 'dg', markersize=12, label='dy<0')
            pmatlab.enlargelims()
            pmatlab.plot2Dline([-1, 0, peak['x']], '--c', label='x')
            pmatlab.plot2Dline([-1, 0, x[peak['phalf0']]], '--y', label='phalf0')

            pmatlab.plot2Dline([-1, 0, x[peak['pbottomlow']]], ':k', label='pbottomlow')

            pmatlab.plot2Dline([-1, 0, peak['xbottoml']], '--y', label='xbottoml')

            plt.legend(loc=0)

    return peaks

#%%


def fitPeaks(XX, YY, pt, fig=None, verbose=0):
    """ Fit Gaussian model on peak """
    pt = np.array(pt)
    fp = np.zeros((pt.size, 3))
    for ii, pg in enumerate(pt):
        if verbose:
            print('fitPeaks: peak at %.1f %.1f' % (XX[pg], YY[pg]))
        # fixme: boundaries based on mV
        r = [pg - 30, pg + 30]
        r[1] = np.minimum(r[1], XX.size - 1)
        r[0] = np.maximum(r[0], 0)
        X = XX[r[0]:(r[1] + 1)]
        Y = YY[r[0]:(r[1] + 1)]
        sel = range(r[0], r[1] + 1)
        p = sel.index(pg)
        # Renormalize to a proper PDF
        # Y /= ((xmax-xmin)/N)*Y.sum()

        # Fit a guassian
        p0 = [X[p], 1, 2 * Y[p]]  # Inital guess is a normal distribution
        # Distance to the target function
        errfunc = lambda p, x, y: gauss(x, p) - y
        p1, success = opt.leastsq(errfunc, p0[:], args=(X, Y))
        # errfunc = lambda p, x, y: -np.linalg.norm( gauss(x, p) - y ) # Distance to the target function
        # p1, success = opt.minimize(errfunc, p0[:], args=(X, Y),method='Powell')

        fit_mu, fit_stdev, fit_scale = p1

        FWHM = 2 * np.sqrt(2 * np.log(2)) * fit_stdev
        if verbose:
            print("fitPeaks: peak %d: FWHM %.3f" % (ii, FWHM))
        fp[ii, :] = p1

    if fig:
        plt.figure(fig)
        plt.clf()
        plt.plot(XX, YY, '-b')
        for ii, p in enumerate(pt):
            p1 = fp[ii, :]
            fit_mu, fit_stdev, fit_scale = p1

            FWHM = 2 * np.sqrt(2 * np.log(2)) * fit_stdev
            plt.plot(XX, gauss(XX, p1), lw=3, alpha=.5, color='r')
            plt.axvspan(
                fit_mu - FWHM / 2, fit_mu + FWHM / 2, facecolor='g', alpha=0.25)
        plt.show()
    return fp
#%%


def peakScores(peaksx, x, y, hwtypical=10, verbose=1, fig=None):
    """ Calculate score for peak 

    Arguments
    ---------

    x,y - arrays with 1D scan data
    peaks -  list with detected peaks

    """
    lowvalue = np.percentile(y, 5)
    highvalue = np.percentile(y, 99)

    kk = np.ones(3) / 3.
    ys = y
    for ki in range(8):
        ys = scipy.ndimage.filters.correlate1d(ys, kk, mode='nearest')
    noise = np.std(ys - y)
    stn2 = np.log2((highvalue - lowvalue) / noise)

    if not fig is None:
        plotPeaks(x, y, peaksx, plotLabels=True, fig=fig)
        plt.plot(x, ys, '-g')

    noisefac = logistic(stn2, x0=3.5, alpha=1)
    if verbose:
        print('peakScores: noise factor %.2f' % noisefac)

    for ii, peak in enumerate(peaksx):
        if not peak['valid']:
            peak['score'] = 0
            continue
        h = peak['height']
        h2 = 2 * (peak['y'] - peak['yhalfl'])
        if (h2 / h) < .3:
            # special case
            h = (h2 + h) / 2

        # hw=peak['halfwidth']
        hw = peak['p'] - peak['pbottom']
        hw = np.abs(x[peak['p']] - peak['xhalfl'])
        # peak['halfwidth']=hw
        # slopeX=h/(hw+.5)
        vv = peak['phalf0']
        slope1 = (y[vv + 1] - y[vv]) / (x[vv + 1] - x[vv])
        slope2 = (y[vv] - y[vv - 1]) / (x[vv] - x[vv - 1])
        slope1 = (ys[vv + 1] - ys[vv]) / (x[vv + 1] - x[vv])
        slope2 = (ys[vv] - ys[vv - 1]) / (x[vv] - x[vv - 1])
        peak['slope'] = (slope1 + slope2) / 2

        peak['heightscore'] = h / (highvalue - lowvalue)
        peak['score'] = (h) * (2 / (1 + hw / hwtypical))
        peak['scorerelative'] = (
            h / (highvalue - lowvalue)) * (2 / (1 + hw / hwtypical))
        # peak['score']= peak['score']*noisefac
        peak['noisefactor'] = noisefac
        if verbose:
            print('peakScores: %d: height %.1f halfwidth %.1f, score %.2f' %
                  (ii, h, hw, peak['score']))
            if verbose >= 2:
                print('   slope: %.1f, heightscore %.2f, score %.2f' %
                      (peak['slope'], peak['heightscore'], peak['score']))

#%%


def analysePeaks(x, y, peaks, verbose=1, doplot=0, typicalhalfwidth=13, istep=1):
    """ Analyse Coulomb peaks

    Arguments
    ---------

    x,y: arrays with data
        data in mV
    peaks: list of detected peaks to be analysed
    typicalhalfwidth : float
        typical width of peak (half side)
    """

    if not issorted(x):
        pass
    if x[0] > x[-1]:
        print('analysePeaks: warning: x values are not sorted!!!!')

    for ii, peak in enumerate(peaks):
        p = peak['p']
        # peak['valid']=1 # peak is valid by default
        if verbose:
            print('analysePeaks: peak %d: max %.1f' % (ii, peak['y']))

        if p < 3:
            # discard all measurements to the left of the scan
            peak['valid'] = 0
            peak['xhalf'] = np.NaN
            peak['xhalfl'] = np.NaN
            peak['phalf'] = np.NaN
            continue

        # ind=np.argsort(x)
        # y=np.arange(x.size)
        # ff=scipy.interpolate.interp1d(x[ind], y[ind]  )
        # zi=ff([ x[p]-3*typicalhalfwidth, x[p]] )

        # determine starting points for search of peak
        zi = np.interp(
            [x[p] - 3. * typicalhalfwidth / 1, x[p], x[p] + 3. * typicalhalfwidth / 1], x, range(x.size))
        zi = np.round(zi).astype(int)
        # print(zi)
        # print(x[zi])
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
        xl, yl = x[ind], y[ind]

        if 0:
            hminval0 = peak['halfvaluelow']
            fv = peak['y'] - 1.8 * (peak['y'] - peak['halfvaluelow'])
        else:
            hminval0 = .5 * (peak['y'] + np.min(y[ind]))
            fv = peak['y'] - 1.8 * (peak['y'] - hminval0)

        xh = np.interp(hminval0, yl, xl)
        ph = np.interp(hminval0, yl, range(xl.size))
        ph0 = ind[int(ph)]
        xf = np.interp(fv, yl, xl)
        peak['phalf0'] = ph0
        peak['phalfl'] = None

        peak['indlocal'] = ind

        phalfvalue = np.interp(ph, range(xl.size), yl)
        yhalfl = np.interp(ph, range(xl.size), yl)
        # peak['xhalf'] = xh  # legacy
        peak['xhalfl'] = xh
        peak['xfoot'] = xf
        peak['yhalfl'] = yhalfl

        if doplot >= 2:
            plt.plot(
                peak['xhalfl'], yhalfl, '.', color=[1, 1, 0], markersize=11)
            pmatlab.plot2Dline([-1, 0, x[p] - 3 * typicalhalfwidth / istep], ':c', label='3*thw')
            pmatlab.plot2Dline([-1, 0, peak['xfoot']], ':y', label='xfoot')

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
    ''' Make sure x and y data are ordered '''
    i = np.argsort(x)
    x = x[i]
    y = y[i]
    return x, y


def coulombPeaks(x, y, verbose=1, fig=None, plothalf=False, istep=None):
    """ Detect Coulumb peaks in a 1D scan """

    if istep is None:
        warnings.warn('istep is None, please add istep as a parameter')

    x, y = peakdataOrientation(x, y)  # i=np.argsort(x); x=x[i]; y=y[i]

    peaks = fitCoulombPeaks(x, y, verbose=verbose, fig=None, istep=istep)
    peaks = analysePeaks(x, y, peaks, verbose=0, doplot=0, istep=istep)
    peaks = peakFindBottom(x, y, peaks, verbose=0)
    goodpeaks = filterPeaks(x, y, peaks, verbose=verbose)

    peakScores(goodpeaks, x, y)
    goodpeaks.sort(key=lambda x: -x['score'])

    if fig:
        plotPeaks(x, y, goodpeaks, fig=fig, plotLabels=True, plothalf=plothalf)
    return goodpeaks

#%% Find best slope
import scipy.ndimage
import scipy.ndimage.measurements


def getOverlap(a, b):
    """ Return overlap between two intervals

    >>> getOverlap( [0,2], [0,1])    
    1
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def findBestSlope(x, y, minder=None, fig=None, verbose=1):
    """ Find good slopes to use in sensing dot """
    lowvalue = np.percentile(y, 1)
    highvalue = np.percentile(y, 99)
    H = highvalue - lowvalue

    if minder is None:
        minder = (highvalue - lowvalue) / 100

    k = np.array([1, 0, -1])
    dy = scipy.ndimage.filters.convolve(y, k, mode='nearest')

    labeled_array, num_features = scipy.ndimage.measurements.label(
        dy >= minder)

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

        if (p - pbottom <= 1):
            continue
        slope = dict({'pbottom': pbottom, 'xbottom': x[pbottom], 'xbottoml': x[pbottom], 'ybottoml': y[
                     pbottom], 'x': x[p], 'y': y[p], 'p': p, 'lowvalue': lowvalue})

        slope['halfvalue'] = (slope['y'] + slope['ybottoml']) / 2
        # slope['halfvalue']=(slope['y']-slope['ybottom'])/2
        halfvalue = slope['halfvalue']
        phalfl = np.round((p + pbottom) / 2)  # FIXME
        # print(phalfl)
        slope['phalfl'] = phalfl
        slope['phalf0'] = phalfl
        slope['yhalfl'] = y[slope['phalfl']]
        slope['xhalfl'] = x[slope['phalfl']]

        ind = np.arange(pbottom, p + 1)
        xl, yl = x[ind], y[ind]     # local coordinates

        xhalfl = np.interp(slope['halfvalue'], yl, xl)
        # xhalfl=np.interp(xhlocal, yl,xl)
        phlocal = np.interp(slope['halfvalue'], yl, range(xl.size))
        phl = np.interp(phlocal, range(ind.size), ind)

        slope['yhalfl'] = halfvalue
        slope['xhalfl'] = xhalfl
        slope['phalfl'] = phl
        slope['phalf0'] = int(phl)

        slope['length'] = x - x[pbottom]
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
        plot2Dline([0, -1, minder], '--r', label='Minimum derivative')
        plt.title('findBestSlopes: derivative')

    peakScores(slopes, x, y)
    return slopes, (dy, minder)


def findSensingDotPosition(x, y, verbose=1, fig=None, plotLabels=True, plotScore=True, plothalf=False, useslopes=True, istep=None):
    """ Find best position for sensing dot

    Arguments:
    x,y : array
        data
    verbose: integer
        output level
    Returns
    -------
    goodpeaks : list
        list of detected positions

    """
    goodpeaks = coulombPeaks(x, y, verbose=1, fig=None, istep=istep, plothalf=False)

    if len(goodpeaks) == 0 or useslopes:
        slopes, (dy, minder) = findBestSlope(x, y, fig=None)
        goodpeaks = goodpeaks + slopes

    goodpeaks = filterOverlappingPeaks(goodpeaks, verbose=verbose >= 2)

    if fig:
        plotPeaks(x, y, goodpeaks, fig=fig, plotLabels=plotLabels,
                  plotScore=plotScore, plothalf=plothalf)
    return goodpeaks


def peakOverlap(p1, p2):
    """ Calculate overlap between two peaks or sloped

    Arguments
    ---------
    p1 - Peak object
    p2 - Peak object

    Returns
    -------
    o - A number representing the amount of overlap. 0: no overlap, 1: complete overlap

    """
    a1 = p1['xbottoml'] - p1['x']
    a2 = p2['xbottoml'] - p2['x']
    o = getOverlap([p1['xbottoml'], p1['x']], [p2['xbottoml'], p2['x']])
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

        if verbose >= 2:
            print('%f %f %f -> %f ' % (a1, a2, o, s))

    if verbose:
        print('filterOverlappingPeaks: %d -> %d peaks' % (len(pp), len(gidx)))
    pp = [pp[jj] for jj in gidx]
    pp = sorted(pp, key=lambda p: p['score'])
    return pp
