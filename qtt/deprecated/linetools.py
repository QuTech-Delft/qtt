# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:40:53 2015

@author: eendebakpt
"""

# %% Load packages
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot
import scipy
import copy
import warnings
import skimage.filters

try:
    from skimage import morphology
except:
    pass

_linetoolswarn = False

try:
    import shapely
    import shapely.geometry
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    from shapely.geometry import LineString
except:
    if not _linetoolswarn:
        #warnings.warn('module shapely not found')
        _linetoolswarn = True
try:
    from descartes.patch import PolygonPatch
except:
    if not _linetoolswarn:
        #warnings.warn('module descartes not found')
        _linetoolswarn = True

import qtt
from qtt import pgeometry as pmatlab
from qtt.pgeometry import *
from qtt import pgeometry

from qtt.utilities.imagetools import createCross

import cv2

from qtt.algorithms.generic import scaleImage, smoothImage, localMaxima


warnings.warn('do not import this module, it will be removed in the future', DeprecationWarning)

# %% Functions


def showIm(ims, fig=1, title=''):
    """ Show image with nearest neighbor interpolation and axis scaling """
    matplotlib.pyplot.figure(fig)
    matplotlib.pyplot.clf()
    matplotlib.pyplot.imshow(ims, interpolation='nearest')
    matplotlib.pyplot.axis('image')

# %%


def dummy():
    print('plt: %s' % str(plt))
    print('matplotlib: %s' % str(matplotlib))

    plt.figure(10)
    return


@qtt.utilities.tools.deprecated
def getBlobPosition(ims, label_im, idx):
    """ Get starting position from blob """
    cms = scipy.ndimage.measurements.center_of_mass(
        ims, labels=label_im, index=idx)
    xstart0 = np.array(cms).reshape((2, 1))[[1, 0], :]
    ww = (label_im == idx).nonzero()
    ww = np.vstack((ww[1], ww[0])).T

    dd = ww - xstart0.T
    jj = np.argmin(np.linalg.norm(dd, axis=1))
    xstart = ww[jj, :].reshape((2, 1))

    return xstart


@qtt.utilities.tools.deprecated
def getpatch(ims, pp, samplesize, fig=None):
    """ Return image patch from parameters 
    """
    patch = sampleImage(ims, pp, samplesize=samplesize, fig=fig)
    return patch


def sampleImage(im, pp, samplesize, fig=None, clearfig=True, nrsub=1):
    """ Sample image patch

    The patch is sampled and displayed if fig is not None. The image patch is returned

    Arguments
    ---------
    im : numpy array
         The input image
    pp : list
        line parameters
    samplesize : int
        size of patch to sample
    fig :
    clearfig :
    nrsub :

    """

    H = createH(samplesize, pp)

    # H=pg_transl2H(1*c)*pg_rotation2H(rot2D(theta))*pg_transl2H(-cc) # image
    # to patch

    dsize = (samplesize[0], samplesize[1])

    #patch=cv2.warpPerspective(im.astype(float32), H, dsize)
    #patch=cv2.warpPerspective(im.astype(float32), H, dsize, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, -1)
    patch = cv2.warpPerspective(im.astype(
        np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)
    if not fig is None:
        cc = pp[0:2].reshape((2, 1))

        rr = np.array([[0., 0], dsize]).T
        rr = region2poly(rr)
        rrs = np.array([[dsize[0] * .2, 0], [dsize[0] * .8, dsize[1]]]).T
        rrs = region2poly(rrs)
        rrim = projectiveTransformation(H.I, rr)
        rrims = projectiveTransformation(H.I, rrs)
        ff = np.array([[dsize[0] / 2., 0]]).T
        ffim = projectiveTransformation(H.I, ff)
        plt.figure(fig)
        if clearfig:
            plt.clf()
        plt.subplot(nrsub, 2, 1)
        plt.imshow(im)
        plt.axis('image')
        plt.title('Input image')
        plotPoints(cc, '.r', markersize=14)
        plotPoints(rrim, 'b')
        plotPoints(ffim, '.b', markersize=14)
        plotPoints(rrims, '--b')
        ax = plt.subplot(nrsub, 2, 2)
        plt.imshow(patch, interpolation='nearest')
        plt.axis('off')
        # ax.invert_yaxis()
        plt.title('sampleImage')

    return patch


# %%
import math
from qtt.algorithms.misc import polyarea
from qtt.utilities.imagetools import semiLine, lineSegment


@pmatlab.static_var("HH", np.matrix(np.eye(3)))
def createH(samplesize, pp, scale=1):
    """ Create H matrix to transform image to patch coordinates """
    cx = (np.array(samplesize) / 2. - .5).reshape((2, 1))
    cc = pp[0:2].reshape((2, 1))
    theta = 0  # pp[2]
    # image to patch, written out
    H = createH.HH.copy()
    c = math.cos(theta)
    s = math.sin(theta)

    H.itemset(0, scale * c)
    H.itemset(1, scale * -s)
    H.itemset(2, scale * (-c * cc[0] + s * cc[1]) + cx[0])
    H.itemset(3, scale * s)
    H.itemset(4, scale * c)
    H.itemset(5, scale * (-s * cc[0] - c * cc[1]) + cx[1])
    return H

# %%


def findCrossTemplate(imx, ksize=31, fig=None, istep=2, verbose=1, widthmv=6, lenmv=20., sepmv=2.0, dy=5):
    """ Find crosses in image using template match
    Arguments
    ---------
        istep : float
            sampling rate in mV/pixel
        widthmv, lenmv, sepmv : float
            parameters of the cross model
    Returns
    -------
        ptsim : array
            fitted points
        rr : numpy array
            response of the filter
        results : dict
            more results

    """
    samplesize = np.array([ksize, ksize + dy])
    param = [None, None, sepmv / istep, 3 * np.pi
             / 8, -7 * np.pi / 8, 11 * np.pi / 8, np.pi / 8]
    modelpatch, cdata = createCross(param, samplesize, w=widthmv / istep,
                                    l=lenmv / istep, lsegment=lenmv / istep, H=100)

    imtmp = pmatlab.setregion(scaleImage(imx), scaleImage(modelpatch), [0, 0])

    #rr=cv2.matchTemplate(imx, modelpatch.astype(np.float32), method=cv2.TM_SQDIFF)
    rr = cv2.matchTemplate(scaleImage(imx), scaleImage(
        modelpatch.astype(imx.dtype)), method=cv2.TM_CCORR_NORMED)
    #rr=cv2.matchTemplate(scaleImage(imx), scaleImage(modelpatch.astype(np.float32)), method=cv2.TM_SQDIFF); rr=-rr
    rr = smoothImage(rr)

    thr = .65 * rr.max() + .35 * rr.mean()
    pts = localMaxima(rr, thr=thr, radius=10 / istep)
    pts = np.array(pts)
    pts = pts[[1, 0], :]

    ptsim = pts + ((samplesize - 1.) / 2).reshape((2, 1))

    if verbose:
        print('findCrossTemplate: threshold: %.1f, %d local maxima' % (thr, pts.shape[1]))

    if fig is not None:
        showIm(imtmp, fig=fig)
        plt.plot(ptsim[0], ptsim[1], '.m', markersize=22)
        showIm(rr, fig=fig + 1)
        plt.colorbar()
        plt.title('Template and image')
        plt.plot(pts[0], pts[1], '.m', markersize=22)
        plt.title('Template match')

        qtt.pgeometry.tilefigs([fig, fig + 1])

    return ptsim, rr, dict({'modelpatch': modelpatch})


from qtt.utilities.imagetools import evaluateCross


@qtt.utilities.tools.rdeprecated('use qtt.utilities.imagetools.fitModel instead', expire='1-6-2018')
def fitModel(param0, imx, verbose=1, cfig=None, ksizemv=41, istep=None,
             istepmodel=.5, cb=None, use_abs=False, w=2.5):
    """ Fit model of an anti-crossing 

    This is a wrapper around evaluateCross and the scipy optimization routines.

    Args:
        param0 (array): parameters for the anti-crossing model
        imx (array): input image


    """

    samplesize = [int(ksizemv / istepmodel), int(ksizemv / istepmodel)]

    def costfun(param0): return evaluateCross(param0, imx, fig=None,
                                              istepmodel=istepmodel, usemask=False, istep=istep, use_abs=use_abs)[0]

    vv = []

    def fmCallback(plocal, pglobal):
        """ Helper function to store intermediate results """
        vv.append((plocal, pglobal))
    if cfig is not None:
        def cb(x): return fmCallback(x, None)
        #cb= lambda param0: evaluateCross(param0, imx, ksize, fig=cfig)[0]
        #cb = lambda param0: print('fitModel: cost %.3f' % evaluateCross(param0, imx, ksize, fig=None)[0] )

    if 1:
        # simple brute force
        ranges = list([slice(x, x + .1, 1) for x in param0])
        for ii in range(2):
            ranges[ii] = slice(param0[ii] - 13, param0[ii] + 13, 1)
        ranges = tuple(ranges)
        res = scipy.optimize.brute(costfun, ranges)
        paramy = res
    else:
        paramy = param0
    res = scipy.optimize.minimize(costfun, paramy, method='nelder-mead',
                                  options={'maxiter': 1200, 'maxfev': 101400, 'xatol': 1e-8, 'disp': verbose >= 2}, callback=cb)
    #res = scipy.optimize.minimize(costfun, res.x, method='Powell',  options={'maxiter': 3000, 'maxfev': 101400, 'xtol': 1e-8, 'disp': verbose>=2}, callback=cb)

    if verbose:
        print('fitModel: score %.2f -> %.2f' % (costfun(param0), res.fun))
    return res


@qtt.utilities.tools.rdeprecated(expire='1-1-2018')
def calcSlope(pp):
    q = -np.diff(pp, axis=1)
    psi = math.atan2(q[1], q[0])
    slope = q[1] / q[0]

    return psi, slope


# %%

# %%
@pmatlab.static_var("scaling0", np.diag([1., 1, 1]))
def costFunctionLine(pp, imx, istep, maxshift=12, verbose=0, fig=None, maxangle=np.deg2rad(70), ksizemv=12, dthr=8, dwidth=3, alldata=None, px=None):
    """ Cost function for line fitting

        pp (list or array): line parameters
        imx (numpy array): image to fit to
        istep (float)
        px (array): translational offset to operate from

    """
    istepmodel = .5
    samplesize = [int(imx.shape[1] * istep / istepmodel), int(imx.shape[0] * istep / istepmodel)]

    LW = 2  # [mV]
    LL = 15  # [mV]

    H = costFunctionLine.scaling0.copy()
    H[0, 0] = istep / istepmodel
    H[1, 1] = istep / istepmodel

    #patch=linetools.sampleImage(im, pp, samplesize, fig=11, clearfig=True, nrsub=1)
    dsize = (samplesize[0], samplesize[1])
    patch = cv2.warpPerspective(imx.astype(np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)
    pm0 = np.array(pp[0:2]).reshape((1, 2)) / istepmodel  # [pixel]
    if px is None:
        pxpatch = [patch.shape[1] / 2, patch.shape[0] / 2]
    else:
        pxpatch = (float(istep) / istepmodel) * np.array(px)
    pm = pm0 + pxpatch
    #modelpatch, cdata=createCross(param, samplesize, centermodel=False, istep=istepmodel, verbose=0)

    lowv = np.percentile(imx, 1)
    highv = np.percentile(imx, 95)
    theta = pp[2]

    if verbose:
        print('costFunctionLine: sample line patch: lowv %.1f, highv %.1f' % (lowv, highv))
        # print(px)
    linepatch = lowv + np.zeros((samplesize[1], samplesize[0]))
    lineSegment(linepatch, pm, theta=pp[2], w=LW / istepmodel, l=LL / istepmodel, H=highv - lowv, ml=-6 / istepmodel)
    #plt.figure(99); plt.clf(); plt.imshow(lineseg, interpolation='nearest'); plt.colorbar()
    #plt.figure(99); plt.clf(); plt.imshow(linepatch-lineseg, interpolation='nearest'); plt.colorbar()
    #plt.figure(99); plt.clf(); plt.imshow(linepatch, interpolation='nearest'); plt.colorbar()
    dd = patch - (linepatch)
    cost = np.linalg.norm(dd)
    cost0 = cost

    if 1:
        ddx0 = np.linalg.norm(pm0)  # [pixel]
        ddx = np.linalg.norm(pm0)  # [pixel]
        if verbose:
            print('costFunctionLine: calculate additonal costs: dist %.1f [mV]' % (ddx * istepmodel))

        ddx = pmatlab.smoothstep(ddx, dthr / istepmodel, dwidth / istepmodel)
        if verbose >= 2:
            print('  ddx: %.3f, thr %.3f' % (ddx, dthr / istepmodel))
        cost += 100000 * ddx
    #cost = sLimits(cost, plocal, pm, maxshift, maxangle)

    if fig is not None:
        pmatlab.cfigure(fig)
        plt.clf()
        plt.imshow(patch, interpolation='nearest')
        plt.title('patch: cost %.2f, dist %.1f' % (cost, ddx0 * istep))
        plt.colorbar()
        pm = pm.flatten()
        #plt.plot(pm0.flatten()[0], pm0.flatten()[1], 'dk', markersize=12, label='initial starting point?')
        plt.plot(pm[0], pm[1], '.g', markersize=24, label='fitted point')
        plt.plot(pxpatch[0], pxpatch[1], '.m', markersize=18, label='offset for parameters')

        qq = np.array(pm.reshape(2, 1) + (LL / istepmodel) * pmatlab.rot2D(theta).dot(np.array([[1, -1], [0, 0]])))

        plt.plot(qq[0, :], qq[1, :], '--k', markersize=24, linewidth=2)

        # print(pm)
        plt.axis('image')
#       plt.colorbar()

        pmatlab.cfigure(fig + 1)
        plt.clf()
        plt.imshow(linepatch, interpolation='nearest')
        plt.title('line patch')
        plt.plot(px[0], px[1], '.m', markersize=24)
        plt.axis('image')
        plt.colorbar()
        pmatlab.tilefigs([fig, fig + 1])

        if verbose >= 2:
            pmatlab.cfigure(fig + 2)
            plt.clf()
            xx = np.arange(0, 20, .1)
            xxstep = istepmodel * pmatlab.smoothstep(xx / istepmodel, dthr / istepmodel, (1 / dwidth) / istepmodel)
            plt.plot(xx, xxstep, '.-b', label='distance step')
            plt.xlabel('Distance [mV]')
            plt.legend()

    if verbose:
        print('costFucntion: cost: base %.2f -> final %.2f' % (cost0, cost))
        if verbose >= 2:
            ww = np.abs(dd).mean(axis=0)

            print('costFunction: dd %s ' % ww)

    return cost


# %%

from scipy.optimize import minimize


def fitLine(alldata, param0=None, fig=None):
    """ Fit a line local to a model """
    if param0 is None:
        param0 = [0, 0, .5 * np.pi]  # x,y,theta,
    istep = .5
    verbose = 1
    cb = None
    imx = -np.array(alldata.diff_dir_xy)
    px = [imx.shape[1] / 2, imx.shape[0] / 2]

    def costfun(x): return costFunctionLine(x, imx, istep, verbose=0, px=px, dthr=7, dwidth=4)
    res = minimize(costfun, param0, method='powell', options={
                   'maxiter': 3000, 'maxfev': 101400, 'xtol': 1e-8, 'disp': verbose >= 2}, callback=cb)

    cgate = alldata.diff_dir_xy.set_arrays[1].name
    igate = alldata.diff_dir_xy.set_arrays[0].name
    c = costFunctionLine(res.x, imx, istep, verbose=1, fig=figl, px=px)
    plt.figure(figl)
    plt.xlabel(cgate)
    plt.ylabel(igate)

