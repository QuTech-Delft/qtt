# -*- coding: utf-8 -*-
""" Functionality to analyse and pre-process images

@author: eendebakpt
"""

# %%
import qtt.pgeometry as pgeometry
import numpy as np
try:
    import cv2
    cv2_interpolation = cv2.INTER_AREA
except:
    cv2 = None
    cv2_interpolation = 3


def straightenImage(im, imextent, mvx=1, mvy=None, verbose=0, interpolation=cv2_interpolation):
    """ Scale image to make square pixels

    Arguments
    ---------
    im: array
        input image
    imextend: list of 4 floats
        coordinates of image region (x0, x1, y0, y1)
    mvx, mvy : float
        number of mV per pixel requested

    Returns
    -------
    ims: numpy array
         transformed image
    (fw, fh, mvx, mvy, H) : data
         H is the homogeneous transform from original to straightened image

    """
    if cv2 is None:
        raise Exception('opencv is not installed, method straightenImage is not available')

    dxmv = imextent[1] - imextent[0]
    dymv = imextent[3] - imextent[2]

    dx = im.shape[1]
    dy = im.shape[0]
    mvx0 = dxmv / float(dx - 1)     # mv/pixel
    mvy0 = dymv / float(dy - 1)

    if mvy is None:
        mvy = mvx

    fw = np.abs((float(mvx0) / mvx))
    fh = np.abs((float(mvy0) / mvy))

    if fw < .5:
        fwx = fw
        fac = 1
        ims = im
        while (fwx < .5):
            ims = cv2.resize(
                ims, None, fx=.5, fy=1, interpolation=cv2.INTER_LINEAR)
            fwx *= 2
            fac *= 2
        ims = cv2.resize(
            ims, None, fx=fac * fw, fy=fh, interpolation=interpolation)
    else:
        ims = cv2.resize(im, None, fx=fw, fy=fh, interpolation=interpolation)

    if verbose:
        print('straightenImage: size %s fx %.4f fy %.4f' % (im.shape, fw, fh))
        print('straightenImage: result size %s mvx %.4f mvy %.4f' % (ims.shape, mvx, mvy))

    H = pgeometry.pg_transl2H([-.5, -.5]) .dot(np.diag([fw, fh, 1]).dot(pgeometry.pg_transl2H([.5, .5])))

    return ims, (fw, fh, mvx, mvy, H)
