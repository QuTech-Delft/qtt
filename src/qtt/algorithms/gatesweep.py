""" Functionality to analyse pinch-off scans """

# %% Load packages

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
from qcodes_loop.plots.qcmatplotlib import MatPlot

import qtt.data
import qtt.pgeometry
from qtt.pgeometry import plot2Dline

# %%


def analyseGateSweep(dd, fig=None, minthr=None, maxthr=None, verbose=1, drawsmoothed=False):
    """ Analyse sweep of a gate for pinch value, low value and high value

    Args:
        dd (1D qcodes DataSet): structure containing the scan data
        fig : TODO
        minthr (float) : TODO (default: None)
        maxthr (float) : TODO (default: None)
        verbose (int): Verbosity level
        drawsmoothed (bool): plot the smoothed data

    Returns:
        result (dict): dictionary with analysis results
    """

    goodgate = True

    data = dd

    # should be made generic
    setpoint_name = [x for x in list(data.arrays.keys()) if not x.endswith(
        'amplitude') and getattr(data, x).is_setpoint][0]
    value_parameter_name = data.default_parameter_name()  # e.g. 'amplitude'

    x = data.arrays[setpoint_name]
    value = np.array(data.arrays[value_parameter_name])

    # detect direction of scan
    scandirection = np.sign(x[-1] - x[0])
    if scandirection < 0:
        scandirection = 1
        x = x[::-1]
        value = value[::-1]

    # crude estimate of noise
    noise = np.nanpercentile(np.abs(np.diff(value)), 50)
    lowvalue = np.nanpercentile(value, 1)
    highvalue = np.nanpercentile(value, 90)
    # sometimes a channel is almost completely closed, then the percentile
    # approach does not function well
    ww = value[value >= (lowvalue + highvalue) / 2]
    highvalue = np.nanpercentile(ww, 90)

    if verbose >= 2:
        print('analyseGateSweep: lowvalue %.1f highvalue %.1f' %
              (lowvalue, highvalue))
    d = highvalue - lowvalue

    vv1 = value > (lowvalue + .2 * d)
    vv2 = value < (lowvalue + .8 * d)
    midpoint1 = vv1 * vv2
    ww = midpoint1.nonzero()[0]
    midpoint1 = np.mean(x[ww])

    # smooth signal
    kk = np.ones(3) / 3.
    smoothed_data = value
    for ii in range(4):
        smoothed_data = scipy.ndimage.correlate1d(smoothed_data, kk, mode='nearest')
    midvalue = .7 * lowvalue + .3 * highvalue
    if scandirection >= 0:
        mp = (smoothed_data >= (.7 * lowvalue + .3 * highvalue)).nonzero()[0][0]
    else:
        mp = (smoothed_data >= (.7 * lowvalue + .3 * highvalue)).nonzero()[0][-1]
    mp = max(mp, 2)  # fix for case with zero data signal
    midpoint2 = x[mp]

    if verbose >= 2:
        print('analyseGateSweep: midpoint2 %.1f midpoint1 %.1f' %
              (midpoint2, midpoint1))

    if minthr is not None and np.abs(lowvalue) > minthr:
        if verbose:
            print('analyseGateSweep: gate not good: gate is not closed')
            print(' minthr %s' % minthr)
        midpoint1 = np.percentile(x, .5)
        midpoint2 = np.percentile(x, .5)

        goodgate = False

    # check for gates that are fully open or closed
    if scandirection > 0:
        xleft = x[0:mp]
        leftval = smoothed_data[0:mp]
        rightval = smoothed_data[mp]
    else:
        xleft = x[mp:]
        leftval = smoothed_data[mp:]
        rightval = smoothed_data[0:mp]
    st = smoothed_data.std()
    if verbose >= 2:
        print('analyseGateSweep: leftval %.2f, rightval %.2f' %
              (leftval.mean(), rightval.mean()))
    if goodgate and (rightval.mean() - leftval.mean() < .3 * st):
        if verbose:
            print(
                'analyseGateSweep: gate not good: gate is not closed (or fully closed)')
        midpoint2 = midpoint1 = np.percentile(x, .5)
        goodgate = False

    # fit a polynomial to the left side
    if goodgate and leftval.size > 5:
        fit = np.polyfit(xleft, leftval, 1)
        # pp = np.polyval(fit, xleft)

        p0 = np.polyval(fit, xleft[-1])
        pmid = np.polyval(fit, xleft[0])
        if verbose >= 2:
            print('analyseGateSweep: p0 %.1f, pmid %.1f, leftval[0] %.1f' % (p0, pmid, leftval[0]))

        if pmid + (pmid - p0) * .25 > leftval[0]:
            midpoint2 = midpoint1 = np.percentile(x, .5)
            goodgate = False
            if verbose:
                print(
                    'analyseGateSweep: gate not good: gate is not closed (or fully closed) (line fit check)')

    # another check on closed gates
    if scandirection > 0:
        leftidx = range(0, mp)
        fitleft = np.polyfit(
            [x[leftidx[0]], x[leftidx[-1]]], [lowvalue, value[leftidx[-1]]], 1)
    else:
        leftidx = range(mp, value.shape[0])
        fitleft = np.polyfit(
            [x[leftidx[-1]], x[leftidx[0]]], [lowvalue, value[leftidx[0]]], 1)
    leftval = value[leftidx]

    leftpred = np.polyval(fitleft, x[leftidx])

    if np.abs(scandirection * (xleft[1] - xleft[0])) > 150 and xleft.size > 15:
        xleft0 = x[mp + 6:]
        leftval0 = smoothed_data[mp + 6:]
        fitL = np.polyfit(xleft0, leftval0, 1)
        nd = fitL[0] / (highvalue - lowvalue)
        if goodgate and (nd * 750 > 1):
            midpoint2 = midpoint1 = np.percentile(x, .5)
            goodgate = False
            if verbose:
                print('analyseGateSweep: gate not good: gate is not closed (or fully closed) (slope check)')
            pass

    if np.mean(leftval - leftpred) > noise:
        midpoint2 = midpoint1 = np.percentile(x, .5)
        goodgate = False
        if verbose:
            print(
                'analyseGateSweep: gate not good: gate is not closed (or fully closed) (left region check)')
        pass

    adata = dict({'description': 'pinchoff analysis', 'pinchvalue': 'use pinchoff_point instead',
                  '_pinchvalueX': midpoint1 - 50, 'goodgate': goodgate})
    adata['lowvalue'] = lowvalue
    adata['highvalue'] = highvalue
    adata['xlabel'] = 'Sweep %s [mV]' % setpoint_name
    adata['pinchoff_point'] = midpoint2 - 50
    pinchoff_index = np.interp(-70.5, x, np.arange(x.size))
    adata['pinchoff_value'] = value[int(pinchoff_index)]
    adata['midpoint'] = float(midpoint2)
    adata['_debug'] = {'midpoint1': midpoint1, 'midpoint2': midpoint2}
    adata['midvalue'] = midvalue
    adata['dataset'] = dd.location
    adata['type'] = 'gatesweep'

    if fig is not None:
        plot_pinchoff(adata, ds=dd, fig=fig)

        if drawsmoothed:
            plt.plot(x, smoothed_data, '-g', linewidth=1, label='smoothed data')

        if verbose >= 2:
            plt.plot(x[leftidx], leftpred, '--r', markersize=15, linewidth=1, label='leftpred')
            plt.plot(x[leftidx], leftval, '--m', markersize=15, linewidth=1, label='leftval')

    if verbose >= 1:
        print('analyseGateSweep: pinch-off point %.3f, value %.3f' % (adata['midpoint'], adata['midvalue']))

    if verbose >= 2:
        print('analyseGateSweep: gate status %d: pinchvalue %.1f' %
              (goodgate, adata['pinchoff_point']))
        adata['Xsmooth'] = smoothed_data
        adata['XX'] = None
        adata['X'] = value
        adata['x'] = x
        adata['smoothed_data'] = smoothed_data
        adata['_mp'] = mp

    return adata


# %% Testing

def plot_pinchoff(result, ds=None, fig=10, verbose=1):
    """ Plot result of a pinchoff scan """
    if ds is None:
        ds = qtt.data.get_dataset(result)

    if not result.get('type', 'none') in ['gatesweep', 'pinchoff']:
        raise Exception('calibration result of incorrect type')

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        MatPlot(ds.default_parameter_array(), num=fig)

        lowvalue = result['lowvalue']
        highvalue = result['highvalue']
        pinchoff_point = result['pinchoff_point']
        midpoint = result['midpoint']
        midvalue = result['midvalue']

        plot2Dline([0, -1, lowvalue], '--c', alpha=.5, label='low value')
        plot2Dline([0, -1, highvalue], '--c', alpha=.5, label='high value')

        plot2Dline([-1, 0, midpoint], ':m', linewidth=2, alpha=0.5, label='midpoint')
        if verbose >= 2:
            plt.plot(midpoint, midvalue, '.m', label='midpoint')
        plot2Dline([-1, 0, pinchoff_point], '--g', linewidth=1, alpha=0.5, label='pinchoff_point')
