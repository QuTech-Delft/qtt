""" Legacy functions (do not use) """
import copy

import numpy as np
import scipy
import matplotlib
import sys
import os
import logging
import cv2

import qcodes
# explicit import
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot

import qtt.data
from qtt.data import loadExperimentData
import qtt.algorithms.onedot 


#%%

from qtt.deprecated import linetools
# from qtt.legacy import scaleCmap, plotCircle
from qtt.data import dataset2Dmetadata, dataset2image

from qtt.algorithms.onedot import onedotGetBalanceFine
from qtt.measurements.scans import pinchoffFilename, fixReversal
from qtt.data import load_data, show2D
from qtt.tools import diffImage, diffImageSmooth


from qtt import pgeometry as pmatlab
from qtt.pgeometry import plotPoints, tilefigs
import matplotlib.pyplot as plt
import datetime

#%%

from qtt.measurements.scans import scan2D, scan1D
from qtt.tools import stripDataset


def positionScanjob(scanjob, pt):
    """ Helper function

    Changes an existing scanjob to scan at the centre of the specified point

    """
    scanjob = copy.deepcopy(scanjob)
    sh = float(pt[0] - (scanjob['sweepdata']['start'] + scanjob['sweepdata']['end']) / 2)
    scanjob['sweepdata']['start'] += sh
    scanjob['sweepdata']['end'] += sh

    sh = float(pt[1] - (scanjob['stepdata']['start'] + scanjob['stepdata']['end']) / 2)
    scanjob['stepdata']['start'] += sh
    scanjob['stepdata']['end'] += sh

    return scanjob

from qtt.measurements.scans import sample_data_t

def onedotScan(station, od, basevalues, outputdir, verbose=1, sample_data=sample_data_t(), scanrange=500, step=-8, full=1):
    """ Scan a one-dot

    Arguments
        station (qcodes station):
        od (dict)
        basevalues (list)
        outputdir (str)
        verbose (int): verbosity level

    """
    if verbose:
        print('onedotScan: one-dot: %s' % od['name'])
    gg = od['gates']
    keithleyidx = [od['instrument']]

    gates = station.gates
    gates.set(gg[1], float(basevalues[gg[1]] - 0))    # plunger

    pv1 = float(od['pinchvalues'][0]) 
    pv2 = float(od['pinchvalues'][2]) 
    
    r1=sample_data.gate_boundaries(gg[0])
    r2=sample_data.gate_boundaries(gg[2])
    
    stepstart = float(np.minimum(od['pinchvalues'][0] + scanrange, r1[1]))
    sweepstart = float(np.minimum(od['pinchvalues'][2] + scanrange, r2[1]))
    
    stepend = np.maximum(pv1-10, r1[0])
    sweepend = np.maximum(pv2-10, r2[0])
    
    stepdata = dict({'param': gg[0], 'start': stepstart, 'end': stepend, 'step': step})
    sweepdata = dict({'param': gg[2], 'start': sweepstart, 'end': sweepend, 'step': step})

    wait_time = qtt.scans.waitTime(gg[2], station=station)
    wait_time_base = qtt.scans.waitTime(gg[0], station=station)
    wait_time_sweep = np.minimum(wait_time / 6., .15)

    if full == 0:
        stepdata['step'] = -12
        sweepdata['step'] = -12
    if full == -1:
        stepdata['step'] = -42
        sweepdata['step'] = -42
        wait_time = 0

    scanjob = qtt.scans.scanjob_t({'stepdata': stepdata, 'sweepdata': sweepdata, 'minstrument': keithleyidx})
    scanjob['stepdata']['wait_time'] = wait_time_base + 3 * wait_time
    scanjob['sweepdata']['wait_time'] = wait_time_sweep
    alldata = qtt.scans.scan2D(station, scanjob )

    od, ptv, pt, ims, lv, wwarea = qtt.algorithms.onedot.onedotGetBalance(od, alldata, verbose=1, fig=None)

    alldata.metadata['od'] = od

    return alldata, od

import time


def onedotPlungerScan(station, od, verbose=1):
    """ Make a scan with the plunger of a one-dot """
    # do sweep with plunger
    gates = station.gates
    gg = od['gates']
    ptv = od['setpoint']

    pv = od['pinchvalues'][1]

    scanjob = dict({'minstrument': [od['instrument']]})
    scanjob['sweepdata'] = dict({'param': gg[1], 'start': 50, 'end': pv, 'step': -1})

    gates.set(gg[2], ptv[0, 0] + 20)    # left gate = step gate in 2D plot =  y axis
    gates.set(gg[0], ptv[1, 0] + 20)
    gates.set(gg[1], scanjob['sweepdata']['start'])

    wait_time = qtt.scans.waitTime(gg[1], station=station)
    scanjob['sweepdata']['wait_time']=wait_time / 4.
    time.sleep(wait_time)

    alldata = scan1D(station, scanjob=scanjob)
    alldata.metadata['od'] = od
    stripDataset(alldata)
    scandata = dict(dataset=alldata, od=od)
    return scandata

#%%

from qtt.measurements.scans import scanPinchValue


def onedotScanPinchValues(station, od, basevalues, outputdir, sample_data=None, cache=False, full=0, verbose=1):
    """ Scan the pinch-off values for the 3 main gates of a 1-dot """
    od['pinchvalue'] = np.zeros((3, 1))
    keithleyidx = [od['instrument']]

    for jj, g in enumerate(od['gates']):
        alldata = scanPinchValue(station, outputdir, gate=g, basevalues=basevalues, sample_data=sample_data, minstrument=keithleyidx, cache=cache, full=full)

        adata = alldata.metadata['adata']
        od['pinchvalue'][jj] = adata['pinchvalue']

    return od

#%%


def saveImage(resultsdir, name, fig=None, dpi=300, ext='png', tight=False):
    """ Save matplotlib figure to disk

    Arguments
    ---------
        name : str
            name of file to save
    Returns
    -------
        imfilerel, imfile : string
            filenames
    """
    imfile0 = '%s.%s' % (name, ext)
    imfile = os.path.join(resultsdir, 'pictures', imfile0)
    qtt.tools.mkdirc(os.path.join(resultsdir, 'pictures'))
    imfilerel = os.path.join('pictures', imfile0)

    if fig is not None:
        plt.figure(fig)
    if tight:
        plt.savefig(imfile, dpi=dpi, bbox_inches='tight', pad_inches=tight)
    else:
        plt.savefig(imfile, dpi=dpi)
    return imfilerel, imfile


def plotCircle(pt, radius=11.5, color='r', alpha=.5, linewidth=3, **kwargs):
    """ Plot a circle in a matplotlib figure

    Args:
        pt (array): center of circle
        radius (float): radius of circle
        color (str or list)
        alpha (float): transparency        
    """
    c2 = plt.Circle(pt, radius, color=color, fill=False, linewidth=3, alpha=alpha, **kwargs)
    plt.gca().add_artist(c2)
    return c2


def scaleCmap(imx, setclim=True, verbose=0):
    """ Scale colormap of sensing dot image """
    p99 = np.percentile(imx, 99.9)
    mval = p99

    # 0 <-> alpha
    # mval <->1

    w = np.array([0, 1])

    # cl=(1./mval)*(w)+.2)
    alpha = .23
    cl = (mval / (1 - alpha)) * (w - alpha)

    if verbose:
        print('scaleCmap to %.1f %.1f' % (cl[0], cl[1]))
    if setclim:
        plt.clim(cl)
    return cl


def writeBatchData(outputdir, tag, timestart, timecomplete):
    tt = datetime.datetime.now().strftime('%d%m%Y-%H%m%S')
    with open(os.path.join(outputdir, '%s-%s.txt' % (tag, tt)), 'wt') as fid:
        fid.write('Tag: %s\n' % tag)
        fid.write('Time start: %s\n' % timestart)
        fid.write('Time complete: %s\n' % timecomplete)
        fid.close()
        print('writeBatchData: %s' % fid.name)

#%%


def filterBG(imx, ksize, sigma=None):
    """ Filter away background using Gaussian filter """
    # imq = cv2.bilateralFilter(imx.astype(np.float32),9,75,75)
    # imq=cv2.medianBlur(imx.astype(np.uint8), 33)

    if ksize % 2 == 0:
        ksize = ksize + 1
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    # sigma=.8
    imq = imx.copy()
    imq = cv2.GaussianBlur(imq, (int(ksize), int(ksize)), sigma)
    imq = imx - imq
    return imq


def filterGabor(im, theta0=-np.pi / 8, istep=1, widthmv=2, lengthmv=10, gammax=1, cut=None, verbose=0, fig=None):
    """
    Filter image with Gabor

    step is in pixel/mV

    Input
    -----

    im : array
        input image
    theta0 : float
        angle of Gabor filter (in radians)

    """
    cwidth = 2. * widthmv * np.abs(istep)
    clength = .5 * lengthmv * np.abs(istep)

    # odd number, at least twice the length
    ksize = 2 * int(np.ceil(clength)) + 1

    if verbose:
        print('filterGabor: kernel size %d %d' % (ksize, ksize))
        print('filterGabor: width %.1f pixel (%.1f mV)' % (cwidth, widthmv))
        print('filterGabor: length %.1f pixel (%.1f mV)' % (clength, lengthmv))
    sigmax = cwidth / 2 * gammax
    sigmay = clength / 2

    gfilter = pmatlab.gaborFilter(ksize, sigma=sigmax, theta=theta0, Lambda=cwidth, psi=0, gamma=sigmax / sigmay, cut=cut)
    # gfilter=cv2.getGaborKernel( (ksize,ksize), sigma=sigmax, theta=theta0, lambd=cwidth, gamma=sigmax/sigmay, psi=0*np.pi/2)
    gfilter -= gfilter.sum() / gfilter.size
    imf = cv2.filter2D(im, -1, gfilter)

    if fig is not None:
        plt.figure(fig + 1)
        plt.clf()
        plt.imshow(r[0], interpolation='nearest')
        plt.colorbar()
        plt.clim([-1, 1])
    return imf, (gfilter, )


#%%

import math


def singleRegion(pt, imx, istep, fig=100, distmv=10, widthmv=70, phi=np.deg2rad(10)):
    """ Determine region where we have no electrons

    The output region is in pixel coordinates.

    Arguments
    ---------
        pt : array
            position of (0,0) point
        imx : array
            input image
        istep : float
            scale factor

    """
    pt0 = pt + np.array([-distmv, distmv]) / istep
    dd = widthmv / istep  # [mV]

    phi1 = np.deg2rad(np.pi + phi)
    phi2 = np.deg2rad(np.pi / 2 - phi)
    rr0 = np.zeros((4, 2))
    rr0[0] = pt0

    if 1:
        rr0[1] = pt0 + (np.array([-dd, 0]))
        rr0[3] = pt0 + (np.array([0, dd]))
        rr0[1][1] += (rr0[1][0] - rr0[0][0]) * math.sin(phi)
        rr0[3][0] += (rr0[3][1] - rr0[0][1]) * math.sin(phi)
    else:
        rr0[1] = pt0 + pmatlab.rot2D(phi1).dot(np.array([dd, 0]))
        rr0[3] = pt0 + pmatlab.rot2D(phi2).dot(np.array([dd, 0]))
    rr0[2] = np.array([rr0[1][0], rr0[3][1]])
    # make sure we are inside scan region. we add 1.0 to fix differentiaion issues at the border
    rr0[[1, 2], 0] = np.maximum(rr0[[1, 2], 0], 1.0)

    rr = np.vstack((rr0, rr0[0:1, :]))
    # print(rr0)

    if fig is not None:
        showIm(imx, fig=fig)
        plt.plot(pt[0], pt[1], '.m', markersize=20)

        plt.plot(rr[:, 0], rr[:, 1], '.-g', markersize=14)
    return rr0


from qtt.algorithms.generic import scaleImage


def singleElectronCheck(pt, imx, istep, fig=50, verbose=1):
    """ Check whether we are in the single-electron regime

    Arguments
    ---------
        pt : array
            zero-zero point
        imx : array
            2D scan image
        istep : float
            parameter

    Returns
    -------
        check : integer
            0: false, 1: undetermined, 2: good
    """
    rr0 = singleRegion(pt, imx, istep, fig=None)
    rr = np.vstack((rr0, rr0[0:1, :]))

    imtmp = imx.copy()
    pts = rr.reshape((-1, 1, 2)).astype(int)

    mask = 0 * imtmp.copy().astype(np.uint8)
    cv2.fillConvexPoly(mask, pts, color=[1])

    if 0:
        vvbg = fitBackground(imx, smooth=True, verbose=verbose, fig=None, order=int(3), removeoutliers=True)
    else:
        vvx = imx + filterBG(imx, ksize=math.ceil(45. / istep))
        vvbg = imx - filterBG(imx, ksize=math.ceil(45. / istep))
        # vv=vvbg
        if 0:
            showIm(imx, fig=123)
            showIm(imx - vvbg, fig=124, title='imx-vvbg')
            showIm(vvbg, fig=125, title='vvbg')
            showIm(imx - vvx, fig=126, title='imx-vvx')
        # imf=fourierHighPass(imx, nc=30)

    qq0 = cv2.meanStdDev(imx - vvbg)
    qq = cv2.meanStdDev(imx - vvbg, mask=mask)
    if verbose >= 2:
        print('qq0 %s' % (str(qq0)))
        print('qq %s' % (str(qq)))
        # print(qq)

    thr = 3. * qq[1]
    # thr=3.5*qq[1]
    imf = imx - vvbg
    # imf=smoothImage(imf)
    # imf=anisodiff(imf,niter=25,kappa=50)

    for ii in range(1):
        imf = smoothImage(imf)

    res = (np.abs(imf) > thr).astype(np.uint8)
    kernel = np.ones((3, 1)).astype(np.uint8)
    for ii in range(0):
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    if fig is not None:

        showIm(mask, fig=fig, title='mask')
        plt.plot(rr[:, 0], rr[:, 1], '.-g', markersize=14)
        if verbose >= 2:
            showIm(imf, fig=fig + 1)
            plt.colorbar()
            plt.title('filtered image')
            img = cv2.cvtColor(scaleImage(imx - vvbg), cv2.COLOR_GRAY2RGB)
            img = scaleImage(imx - vvbg)
            imq = np.hstack((img, scaleImage(res)))
            showIm(imq, fig=fig + 10)
            plt.title('filtered image + thresholded')

            plt.plot(rr[:, 0], rr[:, 1], '.-g', markersize=12)
            rr2 = rr + np.array([imx.shape[1], 0])
            plt.plot(rr2[:, 0], rr2[:, 1], '.-g', markersize=12)

            showIm(res, fig=fig + 11)
            # showIm(imf, fig=fig+11);
            plt.plot(rr[:, 0], rr[:, 1], '.-g', markersize=14)

    pixthr = (2.5 / istep)**2
    if (np.sum(res * mask) <= pixthr):
        if np.abs(pmatlab.polyarea(rr)) < ((40 / istep)**2):
            check = 1
        else:
            check = 2
    else:
        check = 0
    if verbose >= 2:
        print('singleElectronCheck: area of region: %.1f/%.1f [mv]^2' % (np.abs(pmatlab.polyarea(rr)), (40 / istep)**2))
    if verbose >= 2:
        print('singleElectronCheck: np.sum(res*mask) %d/%d ' % (np.sum(res * mask), pixthr))
    checks = dict({0: 'false', 1: 'undetermined', 2: 'good'})

    if verbose:
        print('singleElectronCheck: check %s: %s' % (check, checks[check]))
    return check, rr0, res, thr


#%%

#import matplotlib


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    old_LUT = np.array(map(reduced_cmap, step_list))
    new_LUT = np.array(map(function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(('red', 'green', 'blue')):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = sorted(map(lambda x: x + (x[1], ), this_cdict.items()))
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


def cmap_discretize(cmap, N, m=1024):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [
            (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % m, cdict, m)

#%%
from qtt.algorithms.images import straightenImage

#%%

import itertools


def polyfit2d(x, y, z, order=3):
    """ Fit a polynomial on 2D data """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m):
    """ Evaluate a 2D polynomial """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x)
    for a, (i, j) in zip(m, ij):
        z += a * x**i * y**j
    return z


from qtt.algorithms.generic import smoothImage


def fitBackground(im, smooth=True, fig=None, order=3, verbose=1, removeoutliers=False, returndict=None):
    """ Fit smooth background to 1D or 2D image """

    kk = len(im.shape)
    im = np.array(im)
    ny = im.shape[0]
    ydata0 = np.arange(0., ny)

    is1d = kk == 1
    if kk > 1:
        # 2d signal
        nx = im.shape[1]
        xdata0 = np.arange(0., nx)
    else:
        # 1D signal
        xdata0 = [0.]
    xx, yy = np.meshgrid(xdata0, ydata0)

    if smooth:
        ims = smoothImage(im)
    else:
        ims = im

    if 0:
        s2d = scipy.interpolate.RectBivariateSpline(ydata0, xdata0, im)
        vv = s2d.ev(xx, yy)

    if verbose:
        print('fitBackground: is1d %d, order %d' % (is1d, order))

    xxf = xx.flatten()
    yyf = yy.flatten()
    imsf = ims.flatten()
    s2d = polyfit2d(xxf, yyf, imsf, order=order)
    vv = polyval2d(xx, yy, s2d)

    if removeoutliers:
        ww = im - vv
        gidx = np.abs(ims.flatten() - vv.flatten()) < ww.std()
        # gidx=gidx.flatten()
        if verbose:
            print('fitBackGround: inliers %d/%d (std %.2f)' %
                  (gidx.sum(), gidx.size, ww.std()))
        s2d = polyfit2d(xxf[gidx], yyf[gidx], imsf[gidx], order=order)
        vv = polyval2d(xx, yy, s2d)

    if not fig is None:
        if kk == 1:
            plt.figure(fig)
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(im, '.b', label='signal')
            plt.plot(ims, 'og')
            # plt.plot(ims, '.c', label='signal');

            plt.plot(vv, '.-r', label='background')
            # plt.axis('image')
            plt.legend()
            plt.title('fitBackground: image')

            plt.subplot(2, 1, 2)
            plt.plot(ww, '.m')
            plt.title('Diff plot')
        else:
            plt.figure(fig)
            plt.clf()
            plt.subplot(3, 1, 1)
            plt.imshow(im, interpolation='nearest')
            plt.axis('image')
            plt.title('fitBackground: image')
            plt.subplot(3, 1, 2)
            plt.imshow(vv, interpolation='nearest')
            plt.axis('image')
            # plt.colorbar()
            plt.title('fitBackground: interpolation')
            plt.subplot(3, 1, 3)
            plt.imshow(im - vv, interpolation='nearest')
            plt.axis('image')
            plt.title('fitBackground: difference')

    if not returndict is None:
        # returndict['ww'] = ww
        returndict['xx'] = xx
        returndict['yy'] = yy
        returndict['ims'] = ims
    return vv


def cleanSensingImage(im, dy=0, sigma=None, order=3, fixreversal=True, removeoutliers=False, verbose=0):
    """ Clean up image from sensing dot
    
    Args:
        im (numpy array)
        dy (int or str): direction for differentiation
        order (int)
        fixreversal (bool)
        removeoutliers (bool)
    
    """
    verbose = int(verbose)
    removeoutliers = bool(removeoutliers)
    im = np.array(im)
    if sigma is None:
        imx = diffImage(im, dy=dy, size='same')
    else:
        imx = diffImageSmooth(im, dy=dy, sigma=sigma)
    if order >= 0:
        vv = fitBackground(imx, smooth=True, verbose=verbose, fig=None, order=int(order), removeoutliers=removeoutliers)
        ww = (imx - vv).copy()
    else:
        ww = imx.copy()
    if fixreversal:
        ww = fixReversal(ww, verbose=verbose)
    return ww



def showIm(ims, fig=1, title='', showz=False):
    """ Show image with nearest neighbor interpolation and axis scaling """
    plt.figure(fig)
    plt.clf()
    if showz:
        pmatlab.imshowz(ims, interpolation='nearest')
    else:
        plt.imshow(ims, interpolation='nearest')
    plt.axis('image')
    plt.title(title)


def analyse2dot(alldata, fig=300, istep=1, efig=None, verbose=1):
    """ Analyse a 2-dot scan

    Arguments
    ---------
        alldata : dict
            data of 2D scan
        fig : integer or None
            if not None plot results in figure
        istep: float
            conversion parameter
    Output
    ------
        pt : array
            point of zero-zero crossing
        results : dict
            results of the analysis

    """
    # imextent, xdata, ydata, im = get2Ddata(alldata, fastscan=None, verbose=0, fig=None, midx=2)
    extent, g0, g1, xdata, ydata, arrayname = dataset2Dmetadata(alldata)
    im, tr = dataset2image(alldata)
    imextent = tr.matplotlib_image_extent()

    im = fixReversal(im, verbose=0)
    imc = cleanSensingImage(im, sigma=.93, verbose=0)
    imtmp, (fw, fh, mvx, mvy, H) = straightenImage(imc, imextent, mvx=istep, verbose=verbose >= 2,)  # cv2.INTER_NEAREST
    imx = imtmp.astype(np.float64)

    ksize0 = int(math.ceil(31. / istep))
    ksize0 += (ksize0 - 1) % 2
    pts, rr, _ = linetools.findCrossTemplate(imx, ksize=ksize0, istep=istep, fig=efig, widthmv=6, sepmv=3.8)

    # Select best point
    bestidx = np.argsort(pts[0] - pts[1])[0]
    pt = pts[:, bestidx]

    ptq = pmatlab.projectiveTransformation((H.I), pt)
    ptmv = tr.pixel2scan(ptq)
    ims = imc
    # ims= scaleRatio(imc, imextent, verbose=1)

    check, rr0, res, thr = singleElectronCheck(pt, imx, istep, fig=None, verbose=1)
    se = dict({'rr0': rr0, 'res': res, 'check': check, 'thr': thr})

    # pmatlab.tilefigs([200,50, 60, 61])

    if fig is not None:
        ims, (fw, fh, mvx, mvy, _) = straightenImage(imc, imextent, mvx=1, verbose=0)
        show2D(alldata, ims, fig=fig, verbose=0)
        try:
            scaleCmap(ims)
        except:
            # ipython workaround
            pass    
        plt.axis('image')
        plt.title('zero-zero point (zoom)')

        # plotPoints(ptmv, '.m', markersize=22)
        c2 = plotCircle(ptmv, radius=11.5, color='r', linewidth=3, alpha=.5, label='fitted point')

        plt.axis('image')

        region = int(70 / mvx) * np.array([[-1, 1], [-1, 1]]) + ptmv.reshape(2, 1)
        region[0, 0] = max(region[0, 0], imextent[0])
        region[1, 0] = max(region[1, 0], imextent[2])
        region[0, 1] = min(region[0, 1], imextent[1])
        region[1, 1] = min(region[1, 1], imextent[3])
        plt.xlim(region[0])
        plt.ylim(region[1][::])
    results = dict({'ptmv': ptmv, 'pt': pt, 'imc': imc, 'imx': imx, 'singleelectron': se, 'istep': istep})
    return pt, results


def getTwoDotValues(td, ods, basevaluestd=dict({}), verbose=1):
    """ Return settings for a 2-dot, based on one-dot settings """
    # basevalues=dict()

    if verbose >= 2:
        print('getTwoDotValues: start: basevalues td: ')
        print(basevaluestd)

    bpleft = getODbalancepoint(ods[0])
    bpright = getODbalancepoint(ods[1])

    tddata = dict()

    if td['gates'][2] == td['gates'][3]:
        ggg = [None] * 3
        ggL = ods[0]['gates']
        ggR = ods[1]['gates']

        p1 = basevaluestd[ggL[1]]
        p2 = basevaluestd[ggR[1]]

        val = [bpleft[1, 0], p1, bpleft[0, 0]]
        leftval = val[0]
        ggg[0] = ggL[0]
        ggg[1] = ggL[2]
        for g, v in zip(ggL, val):
            basevaluestd[g] = v
        val = [bpright[1, 0], p2, bpright[0, 0]]
        rightval = val[2]
        for g, v in zip(ggR, val):
            basevaluestd[g] = v
        ggg[2] = ggR[2]

        g = ods[0]['gates'][2]
        v1 = bpleft[0, 0]
        v2 = bpright[1, 0]
        v = (v1 + v2) / 2
        if verbose:
            print(
                'getTwoDotValues: one-dots share a gate: %s: %.1f, %.1f [mV]' % (g, v1, v2))
        basevaluestd[g] = float(v)

        tddata['gates'] = [ggg[0], ggL[1], ggg[1], ggR[1], ggg[2]]
        tddata['gatevaluesleft'] = [bpleft[1, 0], basevaluestd[ggL[1]], bpleft[0, 0]]
        tddata['gatevaluesright'] = [bpright[1, 0], basevaluestd[ggR[1]], bpright[0, 0]]

        fac = .10
        fac = 0
        facplunger = .1

        cc = [-rightval * fac, -facplunger * rightval, -(leftval + rightval) * fac / 2, -facplunger * leftval, -leftval * fac]
        print('getTwoDotValues: one-dots share a gate: %s: compensate %s' %
              (str(tddata['gates']), str(cc)))
        for ii, g in enumerate(tddata['gates']):
            basevaluestd[g] += float(cc[ii])
            # basevalues[ggg[ii]]+=10

        tddata['v'] = [v1, v2, v]
        tddata['gatecorrection'] = cc
        tddata['gatevalues'] = [basevaluestd[gx] for gx in tddata['gates']]
        tddata['ods'] = ods
    else:
        gg = ods[0]['gates']
        val = [ods[0]['balancepoint'][1, 0], 0, ods[0]['balancepoint'][0, 0]]
        for g, v in zip(gg, val):
            basevaluestd[g] = v
        gg = ods[1]['gates']
        val = [ods[1]['balancepoint'][1, 0], 0, ods[0]['balancepoint'][0, 0]]
        for g, v in zip(gg, val):
            basevaluestd[g] = float(v)

    # make sure all values are nice floats (not scalar numpy arrays)
    for k in basevaluestd:
        basevaluestd[k] = float(basevaluestd[k])

    if verbose >= 2:
        print('getTwoDotValues: return basevalues: ')
        print(basevaluestd)

    return basevaluestd, tddata


#%%

def showODresults(od, dd2d, fig=200, imx=None, ww=None):
    ''' Show results of a 1-dot fit ? '''
    balancepoint = od['balancepoint']
    ptv0 = od['balancepoint0']
    if not fig:
        return

    tmp = show2D(dd2d, fig=fig, verbose=0)

    _ = show2D(dd2d, fig=fig + 1, verbose=0)
    plt.title('result')
    plt.axis('image')
    plotPoints(balancepoint, '.m', markersize=18)
    plotPoints(od['balancefit'], '--c')

    plotPoints(ptv0, 'or', markersize=10, mew=2.5, fillstyle='none')

    if not ww is None:
        plt.figure(fig + 2)
        plt.clf()
        plt.imshow(imx)
        plt.title('polygon')

        plt.figure(fig + 3)
        plt.clf()
        plt.imshow(imx == ww)
        plt.title('difference')
        plt.axis('image')
        # plotPoints(pt, '.m', markersize=18)

    tilefigs([fig, fig + 1, fig + 2], [3, 2])
#%%


def point_in_poly(x, y, poly):
    ''' Return true if a point is contained in a polygon '''
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def points_in_poly(points, poly_verts):
    ''' Determine whether points are contained in a polygon or not
    '''
    nn = points.shape[0]
    rr = np.zeros((nn,))
    for ii in range(nn):
        rr[ii] = point_in_poly(points[ii, 0], points[ii, 1], poly_verts)

    rr = rr.astype(np.bool)
    return rr


def fillPoly(im, poly_verts, color=None):
    """ Fill a polygon in an image with the specified color

    Replacement for OpenCV function cv2.fillConvexPoly

    Arugments:
        im (array)
        poly_verts (kx2 array): polygon vertices
        color (array or float)
    """
    ny, nx = im.shape[0], im.shape[1]

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((y, x)).T

    npts = int(poly_verts.size / 2)
    poly_verts = poly_verts.reshape((npts, 2))
    poly_verts = poly_verts[:, [1, 0]]

    try:
        from matplotlib.path import Path
        pp = Path(poly_verts)
        r = pp.contains_points(points)
    except:
        # slow version...
        r = points_in_poly(points, poly_verts)
        pass
    im.flatten()[r] = 1
    # grid = points_inside_poly(points, poly_verts)
    grid = r
    grid = grid.reshape((ny, nx))

    return grid


def getPinchvalues(od, xdir, verbose=1):
    """ Get pinch values from recorded data """
    gg = od['gates']
    od['pinchvalues'] = -800 * np.ones(3)
    for jj, g in enumerate(gg):
        # pp='%s-sweep-1d-%s.pickle' % (od['name'], g)
        pp = pinchoffFilename(g, od=None)
        pfile = os.path.join(xdir, pp)

        dd, metadata = qtt.data.loadDataset(pfile)

        adata = qtt.algorithms.gatesweep.analyseGateSweep(dd, fig=0, minthr=100, maxthr=800, verbose=0)
        if verbose:
            print('getPinchvalues: gate %s : %.2f' % (g, adata['pinchvalue']))
        od['pinchvalues'][jj] = adata['pinchvalue']
    return od


def createDoubleDotJobs(two_dots, one_dots, resultsdir, basevalues=dict(), sdinstruments=[], fig=None, verbose=1):
    """ Create settings for a double-dot from scans of the individual one-dots """
    # one_dots=get_one_dots(full=1)
    xdir = os.path.join(resultsdir, 'one_dot')

    jobs = []
    for jj, td in enumerate(two_dots):
        if verbose:
            print('\n#### analysing two-dot: %s' % str(td['gates']))

        try:
            od1 = 'dot-' + '-'.join(td['gates'][0:3])
            od1 = [x for x in one_dots if x['name'] == od1][0]
            od2 = 'dot-' + '-'.join(td['gates'][3:6])
            od2 = [x for x in one_dots if x['name'] == od2][0]
        except Exception as ex:
            print('createDoubleDotJobs: no one-dot data available for %s' %
                  td['name'])
            print(ex)
            continue
            pass

        if verbose >= 2:
            print('get balance point data')
        ods = []
        try:
            for ii, od in enumerate([od1, od2]):

                dstr = '%s-sweep-2d' % (od['name'])
                dd2d = loadExperimentData(resultsdir, tag='one_dot', dstr=dstr)

                od = getPinchvalues(od, xdir, verbose=verbose >= 2)

                if fig:
                    fign = 1000 + 100 * jj + 10 * ii
                    figm = fig + 10 * ii
                else:
                    fign = None
                    figm = None

                od, ptv, pt0, ims, lv, wwarea = qtt.algorithms.onedot.onedotGetBalance(od, dd2d, verbose=verbose >= 2, fig=fign)

                dstrhi = '%s-sweep-2d-hires' % (od['name'])
                tmphi = loadExperimentData(resultsdir, tag='one_dot', dstr=dstrhi)
                alldatahi = tmphi['dataset']
                if verbose >= 2:
                    print('  at onedotGetBalanceFine')
                if (alldatahi is not None) and True:
                    ptv, fimg, _ = onedotGetBalanceFine(dd=alldatahi, verbose=1, fig=None)
                    od['balancepointfine'] = ptv
                    od['setpoint'] = ptv + 10

                if verbose >= 2:
                    print('createDoubleDotJobs: at fillPoly')
                imx = 0 * wwarea.copy().astype(np.uint8)
                tmp = fillPoly(imx, od['balancefit'])
                # cv2.fillConvexPoly(imx, od['balancefit'],color=[1] )

                showODresults(od, dd2d, fig=figm, imx=imx, ww=wwarea)
                if 0:
                    plt.close(fig + 10 * ii + 0)
                    plt.close(fig + 10 * ii + 2)
                    plt.close(fig + 10 * ii + 3)
                ods.append(od)

            if fig:
                tilefigs([fig + 1, fig + 11], [2, 2])

            # Define base values

            tmp = copy.copy(basevalues)
            # print(tmp)
            # print('createDoubleDotJobs: call getTwoDotValues: ')
            basevaluesTD, tddata = getTwoDotValues(td, ods, basevaluestd=tmp, verbose=1)
            # print('### createDoubleDotJobs: debug here: ')
            td['basevalues'] = basevaluesTD
            td['tddata'] = tddata

            # Create scan job

            scanjob = qtt.scans.scanjob_t({'mode': '2d'})
            p1 = ods[0]['gates'][1]
            p2 = ods[1]['gates'][1]

            sweeprange = 240
            if p2 == 'P3':
                sweeprange = qtt.algorithms.generic.signedmin(sweeprange, 160)  # FIXME

            sweeprange = 240
            if p2 == 'P3':
                sweeprange = qtt.algorithms.generic.signedmin(sweeprange, 160)  # FIXME

            e1 = ods[0]['pinchvalues'][1]
            e2 = ods[1]['pinchvalues'][1]
            e1 = float(np.maximum(basevaluesTD[p1] - sweeprange / 2, e1))
            e2 = float(np.maximum(basevaluesTD[p2] - sweeprange / 2, e2))
            s1 = basevaluesTD[p1] + sweeprange / 2
            s2 = basevaluesTD[p2] + sweeprange / 2
            scanjob['stepdata'] = dict({'param': p1, 'start': s1, 'end': e1, 'step': -2})
            scanjob['sweepdata'] = dict({'param': p2, 'start': s2, 'end': e2, 'step': -4})

            scanjob['minstrument'] = sdinstruments
            scanjob['basename'] = 'doubledot-2d'
            scanjob['basevalues'] = basevaluesTD
            scanjob['td'] = td
            jobs.append(scanjob)

            print('createDoubleDotJobs: succesfully created job: %s' % str(basevaluesTD))
        except Exception as e:
            logging.exception("error with double-dot job!")
            print('createDoubleDotJobs: failed to create job file %s' % td['name'])
            continue

    return jobs


if __name__ == '__main__':
    jobs = createDoubleDotJobs(two_dots, one_dots, basevalues=basevalues0, resultsdir=outputdir, fig=None)


#%%

def stopbias(gates):
    """ Stop the bias currents in the sample """
    gates.set_bias_1(0)
    gates.set_bias_2(0)
    for ii in [3]:
        if hasattr(gates, 'set_bias_%d' % ii):
            gates.set('bias_%d' % ii, 0)


def stop_AWG(awg1):
    """ Stop the AWG """
    print('FIXME: add this function to the awg driver')
    if not awg1 is None:
        awg1.stop()
        awg1.set_ch1_status('off')
        awg1.set_ch2_status('off')
        awg1.set_ch3_status('off')
        awg1.set_ch4_status('off')
    print('stopped AWG...')


def printGateValues(gv, verbose=1):
    s = ', '.join(['%s: %.1f' % (x, gv[x]) for x in sorted(gv.keys())])
    return s


def getODbalancepoint(od):
    bp = od['balancepoint']
    if 'balancepointfine' in od:
        bp = od['balancepointfine']
    return bp

import pickle


def loadpickle(pkl_file):
    """ Load objects from file """
    try:
        output = open(pkl_file, 'rb')
        data2 = pickle.load(output)
        output.close()
    except:
        if sys.version_info.major >= 3:
            # if pickle file was saved in python2 we might fix issues with a different encoding
            output = open(pkl_file, 'rb')
            data2 = pickle.load(output, encoding='latin')
            # pickle.load(pkl_file, fix_imports=True, encoding="ASCII", errors="strict")
            output.close()
        else:
            data2 = None
    return data2


def load_qt(fname):
    """ Load qtlab style file """
    alldata = loadpickle(fname)
    if isinstance(alldata, tuple):
        alldata = alldata[0]
    return alldata
