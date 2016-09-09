#%%

import scipy
import scipy.ndimage
from qtt import cfigure, plot2Dline
import qcodes
import numpy as np
import matplotlib.pyplot as plt

from qtt.data import dataset2Dmetadata, image_transform
from qtt.tools import *
import qtt.data
from qtt import pmatlab

from qtt.algorithms.generic import *
import cv2

#%%

def onedotGetBlobs(fimg, fig=None):
    """ Extract blobs for a 2D scan of a one-dot """
    # thr=otsu(fimg)
    thr = np.median(fimg)
    x = np.percentile(fimg, 99.5)
    thr = thr + (x - thr) * .5
    bim = 30 * (fimg > thr).astype(np.uint8)
    #plt.clf(); plt.imshow(fimg, interpolation='nearest'); plt.colorbar();

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
        #ax = plt.gca()
        pmatlab.plotPoints(xx.T, '.g', markersize=16, label='blob centres')
        plt.title('Reponse image with detected blobs')

        plt.figure(fig + 1)
        plt.clf()
        pmatlab.imshowz(bim, interpolation='nearest')
        plt.axis('image')
        plt.colorbar()
        #ax = plt.gca()
        pmatlab.plotPoints(xxw.T, '.g', markersize=16, label='blob centres')
        pmatlab.plotPoints(xx.T, '.m', markersize=12, label='blob centres (alternative)')
        plt.title('Binary blobs')

        tilefigs([fig, fig + 1], [2, 2])

    return xxw, (xx, contours)

def onedotSelectBlob(im, xx, fimg=None, verbose=0):
    """ Select the best blob from a list of blob positions """
    ims = smoothImage(im)

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

    if xx.size==0:
        print( 'FIXME: better return value... ')
        return np.array([1,1])
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
    extentscan, g0,g2,vstep, vsweep, arrayname=dataset2Dmetadata(dd, arrayname=None)
    tr = qtt.data.image_transform(dd)
    if impixel is None:
        impixel, tr=dataset2image(dd, mode='pixel')
        im=np.array(impixel)
    else:
    
        im=np.array(impixel)

    theta0 = baseangle  # np.deg2rad(-45)
#    step = dd['sweepdata']['step']
    step=np.abs(np.mean(np.diff(vstep)))

    filters, angles, _ = makeCoulombFilter(theta0=theta0, step=step, fig=None)

    lowvalue = np.percentile(im, 5)
    highvalue = np.percentile(im, 95)

    #filters, angles = makeCoulombFilter(theta0=-np.pi/4, step=step, fig=fig)

    gfilter = filters[0]
    fimg = cv2.filter2D(im, -1, gfilter)

    bestvalue = highvalue *  gfilter[gfilter > 0].sum() + lowvalue * gfilter[gfilter < 0].sum()

    xxw, _ = onedotGetBlobs(fimg, fig=None)
    vv = onedotSelectBlob(im, xxw, fimg=None)
    ptpixel = np.array(vv).reshape((1, 2))
    pt = tr.pixel2scan(ptpixel.T)
    ptvalue = fimg[int(ptpixel[0,1]), int(ptpixel[0,0]) ]
    
    if verbose:
        print('onedotGetBalanceFine: point/best filter value: %.2f/%.2f' % (ptvalue, bestvalue) )

    if fig is not None:
        #od = dd.get('od', None) FIXME
        od = None
        xx = show2D(dd, impixel=im, fig=fig, verbose=1, title='input image for gabor', units=units)
        if od is not None:
            pt0 = od['balancepoint'].reshape( (2,1))
            pmatlab.plotPoints(pt0, '.m', markersize=12)
        plt.plot(pt[0], pt[1], '.', color=(0, .8, 0), markersize=16)
        plt.axis('image')

        xx = show2D(dd, impixel=fimg, fig=fig + 1, verbose=1, title='response image for gabor', units=units)
        if od is not None:
            pass
            #plotPoints(pt0, '.m', markersize=16)
        plt.plot(pt[0], pt[1], '.', color=(0, .8, 0), markersize=16, label='balance point fine')
        plt.axis('image')

    acc=1
    
    if (np.abs(ptvalue)/bestvalue<0.05):
        acc=0
        logging.debug('accuracy: %d: %.2f' % (acc,  (np.abs(ptvalue)/bestvalue ) ) )
    return pt, fimg, dict({'step': step, 'ptv': pt, 'ptpixel': ptpixel, 'accuracy': acc, 'gfilter': gfilter})

# Testing
if __name__=='__main__':
    fig=100
    ptv, fimg, tmp= onedotGetBalanceFine(im, alldatahi, verbose=1, fig=fig)



#%%

