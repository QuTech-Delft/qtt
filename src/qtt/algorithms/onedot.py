""" Functionality for analysis of single quantum dots

For more details see https://arxiv.org/abs/1603.02274
"""
# %%

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
from qcodes_loop.plots.qcmatplotlib import MatPlot

import qtt.data
import qtt.pgeometry as pgeometry
from qtt.algorithms.generic import detect_blobs_binary, weightedCentroid
from qtt.data import dataset2Dmetadata, dataset2image, show2D
from qtt.pgeometry import plot2Dline

try:
    import cv2
except ImportError:
    import qtt.exceptions
    warnings.warn('could not find opencv, not all functionality available',
                  qtt.exceptions.MissingOptionalPackageWarning)


# %%

def _onedotGetBlobs(fimg, fig=None):
    """ Extract blobs for a 2D scan of a one-dot """
    # thr=otsu(fimg)
    thr = np.median(fimg)
    x = np.percentile(fimg, 99.5)
    thr = thr + (x - thr) * .5
    bim = 30 * (fimg > thr).astype(np.uint8)

    xx = detect_blobs_binary(bim)

    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(
            bim.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif int(cv2.__version__[0]) >= 3:
        _, contours, _ = cv2.findContours(
            bim.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(
            bim.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    qq = []
    for ii in range(len(contours)):
        qq += [weightedCentroid(fimg, contours, contourIdx=ii, fig=None)]
    xxw = np.array(qq)
    if fig is not None:
        plt.figure(fig)
        plt.clf()
        pgeometry.imshowz(fimg, interpolation='nearest')
        plt.axis('image')
        plt.colorbar()
        pgeometry.plotPoints(xx.T, '.g', markersize=16, label='blob centres')
        plt.title('Reponse image with detected blobs')

        plt.figure(fig + 1)
        plt.clf()
        pgeometry.imshowz(bim, interpolation='nearest')
        plt.axis('image')
        plt.colorbar()
        pgeometry.plotPoints(xxw.T, '.g', markersize=16, label='blob centres')
        pgeometry.plotPoints(xx.T, '.m', markersize=12, label='blob centres (alternative)')
        plt.title('Binary blobs')

        pgeometry.tilefigs([fig, fig + 1], [2, 2])

    return xxw, (xx, contours)


def _onedotSelectBlob(im, xx, fimg=None, verbose=0):
    """ Select the best blob from a list of blob positions """
    ims = qtt.algorithms.generic.smoothImage(im)

    lowvalue, highvalue = np.percentile(ims, [5, 95])
    thrvalue = lowvalue + (highvalue - lowvalue) * .1

    goodidx = np.ones(len(xx))
    for jj, p in enumerate(xx):
        v = qtt.algorithms.generic.getValuePixel(ims, p)
        if verbose:
            print('_onedotSelectBlob %d: v %.2f/%.2f' % (jj, v, thrvalue))
        if v < thrvalue:
            goodidx[jj] = 0

    if verbose:
        print('_onedotSelectBlob: good %s' % goodidx)

    if xx.size == 0:
        print('FIXME: better return value... ')
        return np.array([1, 1])
    score = xx[:, 0] - xx[:, 1]
    score[goodidx == 0] += 10000
    idx = np.argmin(score)

    pt = xx[idx]
    return pt


def onedotGetBalanceFine(impixel=None, dd=None, verbose=1, fig=None, baseangle=-np.pi / 4, units=None,
                         full_output=False):
    """ Determine central position of Coulomb peak in 2D scan

    The position is determined by scanning with Gabor filters and then performing blob detection

    The image should be in pixel coordinates


    Returns:
        pt (array): detected point
        results (dict): dictionary with all results
    """
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(dd, arrayname=None)
    tr = qtt.data.image_transform(dd)
    if impixel is None:
        impixel, tr = dataset2image(dd, mode='pixel')
        im = np.array(impixel)
    else:

        im = np.array(impixel)

    theta0 = baseangle
    step = np.abs(np.nanmean(np.diff(vstep)))

    filters, angles, _ = qtt.algorithms.generic.makeCoulombFilter(theta0=theta0, step=step, fig=None)

    lowvalue = np.percentile(im, 5)
    highvalue = np.percentile(im, 95)

    gfilter = filters[0]
    fimg = cv2.filter2D(im, -1, gfilter)

    bestvalue = highvalue * gfilter[gfilter > 0].sum() + lowvalue * gfilter[gfilter < 0].sum()

    xxw, _ = _onedotGetBlobs(fimg, fig=None)
    vv = _onedotSelectBlob(im, xxw, fimg=None)
    ptpixel = np.array(vv).reshape((1, 2))
    pt = tr.pixel2scan(ptpixel.T)
    ptvalue = fimg[int(ptpixel[0, 1]), int(ptpixel[0, 0])]

    if verbose:
        print('onedotGetBalanceFine: point/best filter value: %.2f/%.2f' % (ptvalue, bestvalue))

    if fig is not None:
        od = None
        xx = show2D(dd, impixel=im, fig=fig, verbose=1, title='input image for gabor', units=units)
        if od is not None:
            pt0 = od['balancepoint'].reshape((2, 1))
            pgeometry.plotPoints(pt0, '.m', markersize=12)
        plt.plot(pt[0], pt[1], '.', color=(0, .8, 0), markersize=16)
        plt.axis('image')

        xx = show2D(dd, impixel=fimg, fig=fig + 1, verbose=1, title='response image for gabor', units=units)
        plt.plot(pt[0], pt[1], '.', color=(0, .8, 0), markersize=16, label='balance point fine')
        plt.axis('image')

    acc = 1

    if np.abs(ptvalue) / bestvalue < 0.05:
        acc = 0
        logging.debug('accuracy: %d: %.2f' % (acc, (np.abs(ptvalue) / bestvalue)))

    results = dict({'step': step, 'ptv': pt, 'ptpixel': ptpixel, 'accuracy': acc, 'gfilter': gfilter})
    if full_output:
        results['fimg'] = fimg

    return pt, results


# %%

def costscoreOD(a, b, pt, ww, verbose=0, output=False):
    """ Cost function for simple fit of one-dot open area

    Args:
        a,b (float): position along axis (a is the x-axis)
        pt (numpy array): point in image
        ww (array)
        verbose (int)
        output (bool)

    Returns:
        cost (float)
    """
    pts = np.array(
        [[a, 0], pt, [ww.shape[1] - 1, b], [ww.shape[1] - 1, 0], [a, 0]])
    pts = pts.reshape((5, 1, 2)).astype(int)
    imx = 0 * ww.copy().astype(np.uint8)
    cv2.fillConvexPoly(imx, pts, color=[1])

    area = np.abs(pgeometry.polyarea(pts.reshape((-1, 2))))

    cost = -(imx == ww).sum()

    # add penalty for moving out of range
    cost += (.025 * ww.size) * np.maximum(b - ww.shape[0] - 1, 0) / ww.shape[0]
    cost += (.025 * ww.size) * np.maximum(-a, 0) / ww.shape[1]

    cost += (.025 * ww.size) * 2 * (pts[2, 0, 1] < 0)
    cost += (.025 * ww.size) * 2 * (pt[0] < 0)  # x too far left
    cost += (.025 * ww.size) * 2 * (pt[1] > ww.shape[0])  # y too far down
    cost += 1e-3 * area

    if verbose:
        print('costscore %.2f' % cost)
    if output:
        return cost, pts, imx
    else:
        return cost

# %%


def onedotGetBalance(dataset, verbose=1, fig=None, drawpoly=False, polylinewidth=2,
                     linecolor='c', full_output=False, od=None):
    """ Determine tuning point from a 2D scan of a 1-dot

    This function performs a simple fitting of the open (conducting region).

    Args:
        od (one-dot structure or None): data for one-dot
        dd (2D dataset): data containing charge stability diagram

    Returns:
        fitresults (dict): dictionary with fitting results
        od (obj): modified one-dot object

    """
    if od is not None:
        warnings.warn('od argument will be removed in the future', DeprecationWarning)

    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(dataset, arrayname=None)

    im, tr = qtt.data.dataset2image(dataset)

    extentImageMatlab = tr.matplotlib_image_extent()

    ims = im.copy()

    # simlpy smoothing of the image
    kk = np.ones((3, 3)) / 9.
    for _ in range(2):
        ims = scipy.ndimage.convolve(ims, kk, mode='nearest', cval=0.0)

    r = np.percentile(ims, 99) - np.percentile(ims, 1)
    lv = np.percentile(ims, 2) + r / 100
    x = ims.flatten()
    lvstd = np.std(x[x < lv])
    lv = lv + lvstd / 2  # works for very smooth images

    lv = (.45 * pgeometry.otsu(ims) + .55 * lv)  # more robust
    if verbose >= 2:
        print('onedotGetBalance: threshold for low value %.1f' % lv)

    # balance point: method 1 (first point above threshold of 45 degree line)
    try:
        ww = np.nonzero(ims > lv)
        zz = -ww[0] + ww[1]
        idx = zz.argmin()
        pt = np.array([[ww[1][idx]], [ww[0][idx]]])
        ptv = tr.pixel2scan(pt)
    except:
        print('qutechtnotools: error in onedotGetBalance: please debug')
        idx = 0
        pt = np.array([[int(vstep.size / 2)], [int(vsweep.size / 2)]])
        ptv = np.array([[vstep[pt[0, 0]]], [vsweep[-pt[1, 0]]]])
        pass

    # balance point: method 2 (fit quadrilateral)
    wwarea = ims > lv

    x0 = np.array([pt[0] - .1 * im.shape[1], pt[1] + .1 * im.shape[0], pt[0], pt[1]]).reshape(4,)  # initial square

    def ff(x): return costscoreOD(x[0], x[1], x[2:4], wwarea)

    # scipy.optimize.show_options(method='Nelder-Mead')

    opts = dict({'disp': verbose >= 2, 'fatol': 1e-6, 'xatol': 1e-5})
    powell_opts = dict({'disp': verbose >= 2, 'ftol': 1e-6, 'xtol': 1e-5})

    xx = scipy.optimize.minimize(ff, x0, method='Nelder-Mead', options=opts)
    # print('  optimize: %f->%f' % (ff(x0), ff(xx.x)) )
    opts['disp'] = verbose >= 2
    xx = scipy.optimize.minimize(ff, xx.x, method='Powell', options=powell_opts)
    x = xx.x
    cost, pts, imx = costscoreOD(x0[0], x0[1], x0[2:4], wwarea, output=True)
    balancefitpixel0 = pts.reshape((-1, 2)).T.copy()
    cost, pts, imx = costscoreOD(x[0], x[1], x[2:4], wwarea, output=True)
    pt = pts[1, :, :].transpose()

    fitresults = {}
    fitresults['balancepoint0'] = ptv
    fitresults['balancepointpixel'] = pt
    fitresults['balancepointpolygon'] = tr.pixel2scan(pt)
    fitresults['balancepoint'] = tr.pixel2scan(pt)
    fitresults['balancefitpixel'] = pts.reshape((-1, 2)).T.copy()
    fitresults['balancefit'] = tr.pixel2scan(fitresults['balancefitpixel'])
    fitresults['balancefit1'] = tr.pixel2scan(balancefitpixel0)
    fitresults['setpoint'] = fitresults['balancepoint'] + 8
    fitresults['x0'] = x0
    fitresults['gatevalues'] = dataset.metadata.get('allgatevalues', None)

    if od is not None:

        fitresults['gatevalues'][od['gates'][2]] = float(fitresults['balancepoint'][0])
        fitresults['gatevalues'][od['gates'][0]] = float(fitresults['balancepoint'][1])

    ptv = fitresults['balancepoint']

    if od is not None:
        # copy results into od structure
        for k in fitresults:
            od[k] = fitresults[k]
        od['onedotbalance'] = fitresults

        odname = od['name']
    else:
        odname = 'one-dot'

    if verbose:
        print('onedotGetBalance %s: balance point 0 at: %.1f %.1f [mV]' % (odname, ptv[0, 0], ptv[1, 0]))
        print('onedotGetBalance: balance point at: %.1f %.1f [mV]' % (
            fitresults['balancepoint'][0, 0], fitresults['balancepoint'][1, 0]))

    if verbose >= 3:
        # %
        plt.figure(9)
        plt.clf()
        plt.imshow(im, interpolation='nearest')
        pgeometry.plotPoints(balancefitpixel0, '.-r', label='balancefitpixel0')
        pgeometry.plotLabels(balancefitpixel0)
        pgeometry.plotPoints(fitresults['balancefitpixel'], '.-m')
        pgeometry.plotLabels(fitresults['balancefitpixel'])

        cost, pts, imx = costscoreOD(x[0], x[1], x[2:4], wwarea, output=True, verbose=1)

        # %
    if fig is not None:
        plot_onedot(fitresults, ds=dataset, verbose=2, fig=fig, linecolor='c',
                    ims=ims, extentImageMatlab=extentImageMatlab, lv=lv)

        qtt.utilities.tools.showImage(im, extentImageMatlab, fig=fig+1)

        if verbose >= 2 or drawpoly:
            pgeometry.plotPoints(fitresults['balancefit'], '--', color=linecolor,
                                 linewidth=polylinewidth, label='balancefit')
        if verbose >= 2:
            pgeometry.plotPoints(fitresults['balancepoint0'], '.r', markersize=13, label='balancepoint0')
        pgeometry.plotPoints(fitresults['balancepoint'], '.m', markersize=17, label='balancepoint')
        plt.axis('image')

    if full_output:
        fitresults['ims'] = ims
        fitresults['lv'] = lv
        fitresults['wwarea'] = wwarea

    return fitresults, ptv


def _plot_dataset(dataset, fig):
    plt.figure(fig)
    plt.clf()
    m = MatPlot(dataset.default_parameter_array(), num=fig)
    return m


def plot_onedot(results, ds=None, verbose=2, fig=100, linecolor='c', ims=None, extentImageMatlab=None, lv=None):
    """ Plot results of a barrier-barrier scan of a single dot

    Args:
        results (dict): results of the onedotGetBalance function
        ds (None or DataSet): dataset to use for plotting
        fig (int or None): figure window to plot to
    """

    if ds is None:
        ds = qtt.data.get_dataset(results)

    if fig is not None:
        _plot_dataset(ds, fig)

        if verbose >= 2:
            pgeometry.plotPoints(results['balancefit'], '--', color=linecolor, linewidth=2, label='balancefit')
        if verbose >= 2:
            pgeometry.plotPoints(results['balancepoint0'], '.r', markersize=13, label='balancepoint0')
        pgeometry.plotPoints(results['balancepoint'], '.m', markersize=17, label='balancepoint')

        if ims is not None:
            qtt.utilities.tools.showImage(ims, extentImageMatlab, fig=fig + 1)  # XX
            plt.axis('image')
            plt.title('Smoothed image')
            pgeometry.plotPoints(results['balancepoint'], '.m', markersize=16, label='balancepoint')

            qtt.utilities.tools.showImage(ims > lv, None, fig=fig + 2)
            pgeometry.plotPoints(results['balancefitpixel'], '--c', markersize=16, label='balancefit')
            pgeometry.plotLabels(results['balancefitpixel'])
            plt.axis('image')
            plt.title('thresholded area')

            if verbose >= 2:
                qq = ims.flatten()
                plt.figure(fig + 3)
                plt.clf()
                plt.hist(qq, 20)
                plot2Dline([-1, 0, np.percentile(ims, 1)], '--m', label='percentile 1')
                plot2Dline([-1, 0, np.percentile(ims, 2)], '--m', label='percentile 2')
                plot2Dline([-1, 0, np.percentile(ims, 99)], '--m', label='percentile 99')
                plot2Dline([-1, 0, lv], '--r', linewidth=2, label='lv')
                plt.legend(numpoints=1)
                plt.title('Histogram of image intensities')
                plt.xlabel('Image (smoothed) values')
