#%%

import scipy
import scipy.ndimage
from qtt import cfigure, plot2Dline
import qcodes
import numpy as np
import matplotlib.pyplot as plt

import logging

from qtt.data import dataset2Dmetadata, image_transform, dataset2image, show2D
#from qtt.tools import *
import qtt.data
from qtt import pmatlab
import qtt.pgeometry as pgeometry
import qtt.scans  # import fixReversal, checkReversal

from qtt.algorithms.generic import getValuePixel
from qtt.algorithms.generic import detect_blobs_binary, weightedCentroid

import cv2

#%%


def onedotGetBlobs(fimg, fig=None):
    """ Extract blobs for a 2D scan of a one-dot """
    # thr=otsu(fimg)
    thr = np.median(fimg)
    x = np.percentile(fimg, 99.5)
    thr = thr + (x - thr) * .5
    bim = 30 * (fimg > thr).astype(np.uint8)
    # plt.clf(); plt.imshow(fimg, interpolation='nearest'); plt.colorbar();

    xx = detect_blobs_binary(bim)

    if int(cv2.__version__[0]) >= 3:
        # opencv 3
        ww, contours, tmp = cv2.findContours(
            bim.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, tmp = cv2.findContours(
            bim.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    qq = []
    for ii in range(len(contours)):
        qq += [weightedCentroid(fimg, contours, contourIdx=ii, fig=None)]
    xxw = np.array(qq)
    if fig is not None:
        plt.figure(fig)
        plt.clf()
        pmatlab.imshowz(fimg, interpolation='nearest')
        plt.axis('image')
        plt.colorbar()
        # ax = plt.gca()
        pmatlab.plotPoints(xx.T, '.g', markersize=16, label='blob centres')
        plt.title('Reponse image with detected blobs')

        plt.figure(fig + 1)
        plt.clf()
        pmatlab.imshowz(bim, interpolation='nearest')
        plt.axis('image')
        plt.colorbar()
        # ax = plt.gca()
        pmatlab.plotPoints(xxw.T, '.g', markersize=16, label='blob centres')
        pmatlab.plotPoints(xx.T, '.m', markersize=12, label='blob centres (alternative)')
        plt.title('Binary blobs')

        pgeometry.tilefigs([fig, fig + 1], [2, 2])

    return xxw, (xx, contours)


def onedotSelectBlob(im, xx, fimg=None, verbose=0):
    """ Select the best blob from a list of blob positions """
    ims = qtt.algorithms.generic.smoothImage(im)

    lowvalue = np.percentile(ims, 5)
    highvalue = np.percentile(ims, 95)
    thrvalue = lowvalue + (highvalue - lowvalue) * .1

    goodidx = np.ones(len(xx))
    for jj, p in enumerate(xx):
        v = getValuePixel(ims, p)
        if verbose:
            print('onedotSelectBlob %d: v %.2f/%.2f' % (jj, v, thrvalue))
        if v < thrvalue:
            goodidx[jj] = 0
    lowvalue = np.percentile(im, 5)
    highvalue = np.percentile(im, 95)

    if verbose:
        print('onedotSelectBlob: good %s' % goodidx)

    if xx.size == 0:
        print('FIXME: better return value... ')
        return np.array([1, 1])
    score = xx[:, 0] - xx[:, 1]
    score[goodidx == 0] += 10000
    idx = np.argmin(score)

    pt = xx[idx]
    return pt


def onedotGetBalanceFine(impixel=None, dd=None, verbose=1, fig=None, baseangle=-np.pi / 4, units=None):
    """ Determine central position of Coulomb peak in 2D scan

    The position is determined by scanning with Gabor filters and then performing blob detection

    image should be in pixel coordinates

    """
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(dd, arrayname=None)
    tr = qtt.data.image_transform(dd)
    if impixel is None:
        impixel, tr = dataset2image(dd, mode='pixel')
        im = np.array(impixel)
    else:

        im = np.array(impixel)

    theta0 = baseangle  # np.deg2rad(-45)
#    step = dd['sweepdata']['step']
    step = np.abs(np.nanmean(np.diff(vstep)))

    filters, angles, _ = qtt.algorithms.generic.makeCoulombFilter(theta0=theta0, step=step, fig=None)

    lowvalue = np.percentile(im, 5)
    highvalue = np.percentile(im, 95)

    # filters, angles = makeCoulombFilter(theta0=-np.pi/4, step=step, fig=fig)

    gfilter = filters[0]
    fimg = cv2.filter2D(im, -1, gfilter)

    bestvalue = highvalue * gfilter[gfilter > 0].sum() + lowvalue * gfilter[gfilter < 0].sum()

    xxw, _ = onedotGetBlobs(fimg, fig=None)
    vv = onedotSelectBlob(im, xxw, fimg=None)
    ptpixel = np.array(vv).reshape((1, 2))
    pt = tr.pixel2scan(ptpixel.T)
    ptvalue = fimg[int(ptpixel[0, 1]), int(ptpixel[0, 0])]

    if verbose:
        print('onedotGetBalanceFine: point/best filter value: %.2f/%.2f' % (ptvalue, bestvalue))

    if fig is not None:
        # od = dd.get('od', None) FIXME
        od = None
        xx = show2D(dd, impixel=im, fig=fig, verbose=1, title='input image for gabor', units=units)
        if od is not None:
            pt0 = od['balancepoint'].reshape((2, 1))
            pmatlab.plotPoints(pt0, '.m', markersize=12)
        plt.plot(pt[0], pt[1], '.', color=(0, .8, 0), markersize=16)
        plt.axis('image')

        xx = show2D(dd, impixel=fimg, fig=fig + 1, verbose=1, title='response image for gabor', units=units)
        if od is not None:
            pass
            # plotPoints(pt0, '.m', markersize=16)
        plt.plot(pt[0], pt[1], '.', color=(0, .8, 0), markersize=16, label='balance point fine')
        plt.axis('image')

    acc = 1

    if (np.abs(ptvalue) / bestvalue < 0.05):
        acc = 0
        logging.debug('accuracy: %d: %.2f' % (acc, (np.abs(ptvalue) / bestvalue)))
    return pt, fimg, dict({'step': step, 'ptv': pt, 'ptpixel': ptpixel, 'accuracy': acc, 'gfilter': gfilter})

# Testing
if __name__ == '__main__':
    fig = 100
    ptv, fimg, tmp = onedotGetBalanceFine(im, alldatahi, verbose=1, fig=fig)


#%%

def costscoreOD(a, b, pt, ww, verbose=0, output=False):
    """ Cost function for simple fit of one-dot open area

    Arguments:
        a,b (float): position along axis (a: x-axis)
        pt (numpy array): point in image

    """
    pts = np.array(
        [[a, 0], pt, [ww.shape[1] - 1, b], [ww.shape[1] - 1, 0], [a, 0]])
    pts = pts.reshape((5, 1, 2)).astype(int)
    imx = 0 * ww.copy().astype(np.uint8)
    cv2.fillConvexPoly(imx, pts, color=[1])
    # tmp=fillPoly(imx, pts)

    cost = -(imx == ww).sum()

    # add penalty for moving out of range
    cost += (.025 * ww.size) * np.maximum(b - ww.shape[0] - 1, 0) / ww.shape[0]
    cost += (.025 * ww.size) * np.maximum(-a, 0) / ww.shape[1]

    cost += (.025 * ww.size) * 2 * (pts[2, 0, 1] < 0)

    if verbose:
        print('costscore %.2f' % cost)
    if output:
        return cost, pts, imx
    else:
        return cost

#%%
#from qtt.legacy import fixReversal, checkReversal
import warnings


def onedotGetBalance(od, dd, verbose=1, fig=None, drawpoly=False, polylinewidth=2, linecolor='c'):
    """ Determine tuning point from a 2D scan of a 1-dot """
    # XX = dd['data_array']
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(dd, arrayname=None)

    im, impixel, tr = qtt.data.dataset2image2(dd)

    r = qtt.scans.checkReversal(im, verbose=verbose >= 2)
    if r == -1:
        warnings.warn('image sign is not correct!?')
        im = qtt.scans.fixReversal(im)
        impixel = r * impixel

    extentImage = [vsweep[0], vsweep[-1], vstep[-1], vstep[0]]  # matplotlib extent style

    ims = impixel.copy()
    kk = np.ones((3, 3)) / 9.
    for ii in range(2):
        ims = scipy.ndimage.convolve(ims, kk, mode='nearest', cval=0.0)

    r = np.percentile(ims, 99) - np.percentile(ims, 1)
    lv = np.percentile(ims, 2) + r / 100
    x = ims.flatten()
    lvstd = np.std(x[x < lv])
    lv = lv + lvstd / 2  # works for very smooth images

    lv = (.45 * pmatlab.otsu(ims) + .55 * lv)  # more robust
    if verbose >= 2:
        print('onedotGetBalance: threshold for low value %.1f' % lv)

    # balance point: method 1 (first point above threshold of 45 degree line)
    try:
        ww = np.nonzero(ims > lv)
        # ww[0]+ww[1]
        zz = -ww[0] + ww[1]
        idx = zz.argmin()
        pt = np.array([[ww[1][idx]], [ww[0][idx]]])
        ptv = tr.pixel2scan(pt)
        # print(pt); print(ptv)
        # print(vsweep)

    except:
        print('qutechtnotools: error in onedotGetBalance: please debug')
        idx = 0
        pt = np.array([[int(vstep.size / 2)], [int(vsweep.size / 2)]])
        ptv = np.array([[vstep[pt[0, 0]]], [vsweep[-pt[1, 0]]]])
        pass
    od['balancepoint0'] = ptv
    # od['balancepoint0'] = ptv

    # balance point: method 2 (fit quadrilateral)
    wwarea = ims > lv

    # x0=np.array( [pt[0],im.shape[0]+.1,pt[0], pt[1] ] )
    x0 = np.array([pt[0] - .1 * im.shape[1], pt[1] + .1 *
                   im.shape[0], pt[0], pt[1]]).reshape(4,)  # initial square
    ff = lambda x: costscoreOD(x[0], x[1], x[2:4], wwarea)
    # ff(x0)

    # scipy.optimize.show_options(method='Nelder-Mead')

    import distutils.version

    if distutils.version.LooseVersion(scipy.version.version) < '0.18':
        opts = dict({'disp': verbose >= 2, 'ftol': 1e-6, 'xtol': 1e-5})
    else:
        opts = dict({'disp': verbose >= 2, 'fatol': 1e-6, 'xatol': 1e-5})

    xx = scipy.optimize.minimize(ff, x0, method='Nelder-Mead', options=opts)
    # print('  optimize: %f->%f' % (ff(x0), ff(xx.x)) )
    opts['disp'] = verbose >= 2
    xx = scipy.optimize.minimize(ff, xx.x, method='Powell', options=opts)
    x = xx.x
    # print('  optimize: %f->%f' % (ff(x0), ff(xx.x)) )
    cost, pts, imx = costscoreOD(x0[0], x0[1], x0[2:4], wwarea, output=True)
    balancefitpixel0 = pts.reshape((-1, 2)).T.copy()
    cost, pts, imx = costscoreOD(x[0], x[1], x[2:4], wwarea, output=True)
    pt = pts[1, :, :].transpose()

    od['balancepointpixel'] = pt
    od['balancepointpolygon'] = tr.pixel2scan(pt)
    od['balancepoint'] = tr.pixel2scan(pt)
    od['balancefitpixel'] = pts.reshape((-1, 2)).T
    od['balancefit'] = tr.pixel2scan(od['balancefitpixel'])
    od['balancefit1'] = tr.pixel2scan(balancefitpixel0)
    od['setpoint'] = od['balancepoint'] + 8
    od['x0'] = x0
    # od['xx']=dict(xx)
    ptv = od['balancepoint']

    # print(balancefitpixel0)
    # print(od['balancefitpixel'])

    if verbose:
        print('onedotGetBalance %s: balance point 0 at: %.1f %.1f [mV]' % (od['name'], ptv[0, 0], ptv[1, 0]))
        print('onedotGetBalance: balance point at: %.1f %.1f [mV]' % (
            od['balancepoint'][0, 0], od['balancepoint'][1, 0]))

    if fig is not None:

        qtt.tools.showImage(im, extentImage, fig=fig)

        plt.axis('image')
        if verbose >= 2 or drawpoly:
            pmatlab.plotPoints(od['balancefit'], '--', color=linecolor, linewidth=polylinewidth, label='balancefit')
        if verbose >= 2:
            pmatlab.plotPoints(od['balancepoint0'], '.r', markersize=13, label='balancepoint0')
        pmatlab.plotPoints(od['balancepoint'], '.m', markersize=17, label='balancepoint')

        plt.title('image')
        plt.xlabel('%s (mV)' % g2)
        plt.ylabel('%s (mV)' % g0)

        qtt.tools.showImage(tr.itransform(ims), extentImage, fig=fig + 1)
        plt.axis('image')
        plt.title('Smoothed image')
        pmatlab.plotPoints(od['balancepoint'], '.m', markersize=16, label='balancepoint')
        plt.xlabel('%s (mV)' % g2)
        plt.ylabel('%s (mV)' % g0)

        qtt.tools.showImage(ims > lv, None, fig=fig + 2)
        # plt.imshow(ims > lv, extent=None, interpolation='nearest')
        pmatlab.plotPoints(balancefitpixel0, ':y', markersize=16, label='balancefit0')
        pmatlab.plotPoints(od['balancefitpixel'], '--c', markersize=16, label='balancefit')
        pmatlab.plotLabels(od['balancefitpixel'])
        plt.axis('image')
        plt.title('thresholded area')
        plt.xlabel('%s' % g2)
        plt.ylabel('%s' % g0)
        pmatlab.tilefigs([fig, fig + 1, fig + 2], [2, 2])

        if verbose >= 2:
            qq = ims.flatten()
            plt.figure(123)
            plt.clf()
            plt.hist(qq, 20)
            plot2Dline([-1, 0, np.percentile(ims, 1)], '--m')
            plot2Dline([-1, 0, np.percentile(ims, 2)], '--m')
            plot2Dline([-1, 0, np.percentile(ims, 99)], '--m')
            plot2Dline([-1, 0, lv], '--r', linewidth=2)
            plt.xlabel('Image (smoothed) values')

    return od, ptv, pt, ims, lv, wwarea

if __name__ == '__main__':
    from imp import reload
    reload(qtt.data)
    dd = alldata
    od, ptv, pt, ims, lv, wwarea = onedotGetBalance(od, alldata, verbose=1, fig=110)
