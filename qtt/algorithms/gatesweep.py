""" Functionality to analyse pinch-off scans """
#%% Load packages

import scipy
import scipy.ndimage
from qtt.pgeometry import cfigure, plot2Dline
import qcodes
import numpy as np
import matplotlib.pyplot as plt

import qtt.data
from qtt.data import dataset2Dmetadata, image_transform, dataset2image, dataset2image2
import scipy
from qtt import pgeometry as pmatlab

#%%

# import qtt.scans # FIXME: circular


def analyseGateSweep(dd, fig=None, minthr=None, maxthr=None, verbose=1, drawsmoothed=True, drawmidpoints=True):
    """ Analyse sweep of a gate for pinch value, low value and high value

    Arguments
    ---------
        dd : DataSet
            structure containing the scan data
        minthr, maxthr : float
            parameters for the algorithm (default: None)

    """

    goodgate = True

    if isinstance(dd, qcodes.DataSet):
        data = dd
        XX = None

        # should be made generic
        g = [x for x in list(data.arrays.keys()) if not x.endswith('amplitude') and getattr(data, x).is_setpoint][0]
        value = data.default_parameter_name()  # e.g. 'amplitude'

        x = data.arrays[g]
        value = data.arrays[value]

        # detect direction of scan
        scandirection = np.sign(x[-1] - x[0])
        if scandirection < 0 and 1:
            pass
            scandirection = 1
            x = x[::-1]
            value = value[::-1]
    else:
        # legacy code
        XX = dd['data_array']
        datashape = XX.shape
        goodgate = True
        sweepdata = dd['sweepdata']
        g = sweepdata['gates'][0]

        sr = np.arange(sweepdata['start'], sweepdata['end'], sweepdata['step'])
        if datashape[0] == 2 * sr.size:
                    # double sweep, reduce
            if verbose:
                print('analyseGateSweep: scan with sweepback: reducing data')
            XX = XX[0:sr.size, :]

        x = XX[:, 0]
        value = XX[:, 2]
        XX = None

    lowvalue = np.percentile(value, 1)
    highvalue = np.percentile(value, 90)
    # sometimes a channel is almost completely closed, then the percentile
    # approach does not function well
    ww = value[value >= (lowvalue + highvalue) / 2]
    #[np.percentile(ww, 1), np.percentile(ww, 50), np.percentile(ww, 91) ]
    highvalue = np.percentile(ww, 90)

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
    ww = value
    for ii in range(4):
        ww = scipy.ndimage.filters.correlate1d(ww, kk, mode='nearest')
        # ww=scipy.signal.convolve(ww, kk, mode='same')
        # ww=scipy.signal.convolve2d(ww, kk, mode='same', boundary='symm')
    midvalue = .7 * lowvalue + .3 * highvalue
    if scandirection >= 0:
        mp = (ww >= (.7 * lowvalue + .3 * highvalue)).nonzero()[0][0]
    else:
        mp = (ww >= (.7 * lowvalue + .3 * highvalue)).nonzero()[0][-1]
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
        leftval = ww[0:mp]
        rightval = ww[mp]
    else:
        xleft = x[mp:]
        leftval = ww[mp:]
        rightval = ww[0:mp]
    st = ww.std()
    if verbose:
        print('analyseGateSweep: leftval %.1f, rightval %.1f' %
              (leftval.mean(), rightval.mean()))
    if goodgate and (rightval.mean() - leftval.mean() < .3 * st):
        if verbose:
            print(
                'analyseGateSweep: gate not good: gate is not closed (or fully closed)')
        midpoint1 = np.percentile(x, .5)
        midpoint2 = np.percentile(x, .5)
        goodgate = False

    # fit a polynomial to the left side
    if goodgate and leftval.size > 5:
        # TODO: make this a robust fit
        fit = np.polyfit(xleft, leftval, 1)
        pp = np.polyval(fit, xleft)

        # pmid = np.polyval(fit, midpoint2)
        p0 = np.polyval(fit, xleft[-1])
        pmid = np.polyval(fit, xleft[0])
        if verbose>=2:
            print('analyseGateSweep: p0 %.1f, pmid %.1f, leftval[0] %.1f' % (p0, pmid, leftval[0]))

        if pmid + (pmid - p0) * .25 > leftval[0]:
            midpoint1 = np.percentile(x, .5)
            midpoint2 = np.percentile(x, .5)
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

    # fitleft=np.polyfit([x[leftidx[-1]],x[leftidx[0]]], [lowvalue, midvalue], 1)

    leftpred = np.polyval(fitleft, x[leftidx])

    if np.abs(scandirection * (xleft[1] - xleft[0])) > 150 and xleft.size > 15:
        xleft0 = x[mp + 6:]
        leftval0 = ww[mp + 6:]
        fitL = np.polyfit(xleft0, leftval0, 1)
        pp = np.polyval(fitL, xleft0)
        nd = fitL[0] / (highvalue - lowvalue)
        if goodgate and (nd * 750 > 1):
            midpoint1 = np.percentile(x, .5)
            midpoint2 = np.percentile(x, .5)
            goodgate = False
            if verbose:
                print('analyseGateSweep: gate not good: gate is not closed (or fully closed) (slope check)')
            pass

    if np.sum(leftval - leftpred) > 0:
        midpoint1 = np.percentile(x, .5)
        midpoint2 = np.percentile(x, .5)
        goodgate = False
        if verbose:
            print(
                'analyseGateSweep: gate not good: gate is not closed (or fully closed) (left region check)')
        pass
    # gate not closed

    if fig is not None:
        cfigure(fig)
        plt.clf()
        plt.plot(x, value, '.-b', linewidth=2)
        plt.xlabel('Sweep %s [mV]' % g, fontsize=14)
        plt.ylabel('keithley [pA]', fontsize=14)

        if drawsmoothed:
            plt.plot(x, ww, '-g', linewidth=1)

        plot2Dline([0, -1, lowvalue], '--m', label='low value')
        plot2Dline([0, -1, highvalue], '--m', label='high value')

        if drawmidpoints:
            if verbose >= 2:
                plot2Dline([-1, 0, midpoint1], '--g', linewidth=1)
            plot2Dline([-1, 0, midpoint2], '--m', linewidth=2)

        if verbose >= 2:
            # plt.plot(x[leftidx], leftval, '.r', markersize=15)
            plt.plot(x[leftidx], leftpred, '--r', markersize=15, linewidth=1)

        if verbose >= 2:
            1
           # plt.plot(XX[ww,0].astype(int), XX[ww,2], '.g')

    adata = dict({'pinchvalue': midpoint2 - 50,
                  'pinchvalueX': midpoint1 - 50, 'goodgate': goodgate})
    adata['lowvalue'] = lowvalue
    adata['highvalue'] = highvalue
    adata['xlabel'] = 'Sweep %s [mV]' % g
    adata['mp'] = mp
    adata['midpoint'] = midpoint2
    adata['midvalue'] = midvalue

    if verbose >= 2:
        print('analyseGateSweep: gate status %d: pinchvalue %.1f' %
              (goodgate, adata['pinchvalue']))
        adata['Xsmooth'] = ww
        adata['XX'] = XX
        adata['X'] = value
        adata['x'] = x
        adata['ww'] = ww

    return adata


#%% Testing

if __name__ == '__main__':
    adata = analyseGateSweep(alldata, fig=10, minthr=None, maxthr=None)
