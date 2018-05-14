""" QuTech Algorithms

    TNO tools

    Pieter Eendebak <pieter.eendebak@tno.nl>
    2015
"""
#%% Import the modules used in this program:

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

raise Exception('no not use this module any more!')

import numpy as np
import math
import scipy.signal
import scipy.ndimage
import scipy.optimize
import scipy.constants
import scipy.optimize
import warnings
import copy
import qtt
import qtt.tools

try:
    import matplotlib.pyplot as plt
    import matplotlib
except:
    warnings.warn('could not import matplotlib, not all functionality available...')
    plt = None
    matplotlib = None
    pass
try:
    import cv2
except Exception as inst:
    # print(inst)
    warnings.warn('could not import OpenCv, not all functionality available...')
    pass
 
warnings.warn('do not import this module, it will be removed in the future', DeprecationWarning)

#%% Custom packages
from qtt import pgeometry as pmatlab


@qtt.tools.deprecated
def extent2fullextent(extent0, im):
    """ Convert extent to include half pixel border """
    nx = im.shape[1]
    ny = im.shape[0]
    dx = (extent0[1] - extent0[0]) / (nx - 1)
    dy = (extent0[3] - extent0[2]) / (ny - 1)
    extent = copy.copy(extent0)
    extent[0] = extent[0] - dx / 2
    extent[1] = extent[1] + dx / 2
    extent[2] = extent[2] - dy / 2
    extent[3] = extent[3] + dy / 2
    return extent


#%% Functions
ueV2Hz = scipy.constants.e / scipy.constants.h * 1e-6


@qtt.tools.rdeprecated('replaced by pat_functions.one_ele_pat_model')
def barrierModel(x, *p):
    r""" Model used for fitting tunnel barrier 

    This is:

    .. math::
        \phi=\sqrt{ { ( (x-x_0) leverarm) }^2 + 4 t^2 } \mathrm{ueV2Hz}

    Here :math:`x_0` is the offset in the scan direction. The leverarm is a free parameter 
    and t is the tunnel barrier.
    The parameter t is in micro eV.

    Arguments
    ---------
        x : array
            data points
        p : tuple
            xoffset, leverarm and t

    >>> barrierModel([1,2,3], 0, 40, 400)
    """
    if len(p) == 1:
        p = p[0]
    xoffset = p[0]
    leverarm = p[1]
    t = p[2]
    y = np.sqrt(np.power((x - xoffset) * leverarm, 2) + 4 * t**2) * ueV2Hz
    return y

#%%


@qtt.tools.rdeprecated('use pat_functions.plot_pat_fit')
def plotBarrierFit(imq, imextent, pp, fig=400, title='Fitted model'):
    """ Plot the fitted model of a V-shape """
    pmatlab.cfigure(fig)
    plt.clf()
    plt.imshow(imq, extent=extent2fullextent(imextent, imq), interpolation='nearest')
    plt.axis('tight')
    plt.title(title)
    plt.ylabel('Frequency')
    plt.ylabel('Frequency / [mV]')
    plt.axis('tight')

    x0 = np.linspace(imextent[0], imextent[1], 1000)
    yfit = barrierModel(x0, pp)
    plt.plot(x0, yfit, '-g', label='Model')

    ppx = copy.deepcopy(pp)
    ppx[2] = 0
    yfit = barrierModel(x0, ppx)
    plt.plot(x0, yfit, '--g')

import scipy.optimize

#%%

from qtt.pgeometry import robustCost


@qtt.tools.deprecated
def barrierScore(xd, yd, pp, weights=None, thr=3e9):
    """ Calculate score for barrier model """
    ppq = pp.copy()
    # ppq[0]+=3.05
    ydatax = barrierModel(xd, ppq)
    sc = np.abs(ydatax - yd)
    scalefac = thr
    #sc=robustCost(sc, thr)
    sc = np.sqrt(robustCost(sc / scalefac, thr / scalefac, 'BZ0')) * scalefac
    if weights is not None:
        sc = sc * weights
    sc = np.linalg.norm(sc, ord=4) / sc.size
    return sc


@qtt.tools.deprecated
def preprocessPAT(imextent, im0, im, fig=None):
    """ Preprocess a pair of calibration and PAT image """
    im0s = qtt.algorithms.generic.smoothImage(im0)
    ks = 15.
    kk = np.ones((1, ks)) / ks
    im0s = scipy.ndimage.filters.convolve(im0, kk, mode='nearest')

    imq = im - im0s

    imq = imq - np.mean(imq, axis=1).reshape((-1, 1))

    ks = 5.
    w = np.ones((1, ks)) / ks
    imx = scipy.ndimage.filters.convolve(imq, w, mode='nearest')

    qq = np.percentile(imx, [5, 50, 95])
    imx = imx - qq[1]
    qq = np.percentile(imx, [2, 50, 98])
    scale = np.mean([-qq[0], qq[2]])
    imx = imx / scale

    if fig is not None:
        plt.figure(fig)
        if 0:
            for x in im:
                plt.plot(x, '.c', label='data')
        plt.clf()
        plt.plot(im0.flatten(), '.b', label='calibration')
        plt.plot(im0s.flatten(), '.-r', label='calibration smoothed')
        plt.title('PAT calibration measurement')
        plt.legend(numpoints=1)

        plt.figure(fig + 1)
        plt.clf()
        plt.imshow(imq, interpolation='nearest')
        plt.axis('tight')
        plt.title('data minus calibration')
        plt.figure(fig + 2)
        plt.clf()
        plt.imshow(imx, extent=extent2fullextent(imextent, imx), interpolation='nearest')
        plt.axis('tight')
        plt.title('data minus calibration (scaled)')
        plt.colorbar()
    return imx, imq, im0s


@qtt.tools.rdeprecated('replaced by pat_functions.detect_peaks')
def detectVshape(imextent, xdata, ydata, imx, sigmamv=.25, fig=400, returndict=None):
    """ Helper function """
    scalefac = (imextent[1] - imextent[0]) / (imx.shape[1] - 1)  # mV/pixel

    # smooth input image
    kern = scipy.signal.gaussian(71, std=sigmamv / scalefac)
    kern = kern / kern.sum()
    imx2 = scipy.ndimage.filters.convolve(imx, kern.reshape((1, -1)), mode='nearest')

    mvthr = .2

    # get maximum value for each row
    mm1 = np.argmax(imx2, axis=1)
    val = imx2[np.arange(0, imx2.shape[0]), mm1]

    idx1 = np.where(np.abs(val) > .5)[0]    # only select indices above scaled threshold .5
    idx1 = idx1[xdata[mm1[idx1]] - xdata[0] > mvthr]  # discard points near border
    idx1 = idx1[xdata[mm1[idx1]] - xdata[-1] < -mvthr]  # discard points near border

    xx1 = np.vstack((xdata[mm1[idx1]], ydata[idx1]))  # position of selected points

    # get minimum value for each row
    mm2 = np.argmin(imx2, axis=1)
    val = imx2[np.arange(0, imx2.shape[0]), mm2]
    # remove points below threshold
    idx2 = np.where(np.abs(val) > .5)[0]

    idx2 = idx2[xdata[mm2[idx2]] - xdata[0] > mvthr]  # discard points near border
    idx2 = idx2[xdata[mm2[idx2]] - xdata[-1] < -mvthr]  # discard points near border
    xx2 = np.vstack((xdata[mm2[idx2]], ydata[idx2]))

    # join the two sets
    xx = np.hstack((xx1, xx2))
    vals = np.hstack((val[idx1], val[idx2]))

    # determine weights for the points
    qq = np.intersect1d(idx1, idx2)
    q1 = np.searchsorted(idx1, qq)
    q2 = np.searchsorted(idx2, qq)
    w1 = .5 * np.ones(len(idx1))
    w1[q1] = 1
    w2 = .5 * np.ones(len(idx2))
    w2[q2] = 1

    wfac = .1
    w1[np.abs(val[idx1]) < .6] = wfac
    w1[np.abs(val[idx1]) < .6] = wfac
    weight = np.hstack((w1, w2))

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        pmatlab.imshowz(imx2, extent=extent2fullextent(imextent, imx2), interpolation='nearest')
        plt.axis('tight')
        plt.title('response')
        plt.colorbar()
        plt.plot(xdata[mm1[idx1]], ydata[idx1], '.b', markersize=14)
        plt.plot(xdata[mm2[idx2]], ydata[idx2], '.r', markersize=14)

    if returndict is not None:
        returndict['xx1'] = xx1
        returndict['imx2'] = imx2
        returndict['vals'] = vals
    return xx, weight, (xx1, xx2, idx1, idx2, vals)

#%%

@qtt.tools.deprecated
def fitBarrierModel(pp0, xd, yd, weights=None, verbose=1, curvefit=False, dd=None):

    if curvefit:
        pp, pcov = scipy.optimize.curve_fit(barrierModel, xd.flatten(), yd.flatten(), pp0)
    else:
        pp = pp0.copy()

    ppx = pp.copy()

    if 1:
        ff = lambda x: barrierScore(xd, yd, [pp[0], pp[1], x], weights=weights)
        #r=scipy.optimize.minimize(ff, pp[2:], method='Nelder-Mead', options=dict({'disp': True}))
        r = scipy.optimize.brute(ff, ranges=[(0, 100)], Ns=20, disp=False)
        ppx[2] = r
        sc0 = barrierScore(xd, yd, pp, weights=weights)
        sc = barrierScore(xd, yd, ppx, weights=weights)
        if verbose >= 2:
            print('fitBarrierModel: %s: %.4f -> %.4f' % (['%.2f' % x for x in ppx], sc0 / 1e6, sc / 1e6))

    if 1:
        ff = lambda x: barrierScore(xd, yd, [x, pp[1], ppx[2]], weights=weights)
        #r=scipy.optimize.minimize(ff, pp[2:], method='Nelder-Mead', options=dict({'disp': True}))
        r = scipy.optimize.brute(ff, ranges=[(pp[0] - 2, pp[0] + 2)], Ns=20, disp=False)
        ppx[0] = r
        sc0 = barrierScore(xd, yd, pp, weights=weights)
        sc = barrierScore(xd, yd, ppx, weights=weights)
        if verbose >= 2:
            print('fitBarrierModel: %s: %.4f -> %.4f' % (['%.2f' % x for x in ppx], sc0 / 1e6, sc / 1e6))
    if 0:
        ff = lambda x: barrierScore(xd, yd, x, weights=weights)

        r = scipy.optimize.brute(ff, ranges=[(pp[0] - 2, pp[0] + 2)], Ns=20, disp=False)
        ppx[0] = r
        sc0 = barrierScore(xd, yd, pp, weights=weights)
        sc = barrierScore(xd, yd, ppx, weights=weights)
        if verbose >= 2:
            print('fitBarrierModel: %s: %.4f -> %.4f' % (['%.2f' % x for x in ppx], sc0 / 1e6, sc / 1e6))

    ff = lambda x: barrierScore(xd, yd, x, weights=weights)
    if 1:
        r = scipy.optimize.minimize(ff, ppx, method='Powell', options=dict({'disp': True}))
        ppx = r['x']

    if 1:
        ff = lambda x: barrierScore(xd, yd, x, weights=weights)
        r = scipy.optimize.minimize(ff, ppx, method='Powell', options=dict({'disp': True}))
        ppx = r['x']

    sc0 = barrierScore(xd, yd, pp, weights=weights)
    sc = barrierScore(xd, yd, ppx, weights=weights)
    if verbose:
        print('fitBarrierModel: %.4f -> %.4f' % (sc0 / 1e6, sc / 1e6))

    if dd is not None:
        dd['pp'] = pp
        dd['weights'] = weights
    return ppx

