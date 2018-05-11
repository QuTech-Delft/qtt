""" Various functions """

# flake8: noqa (we don't need the "<...> imported but unused" error)

import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy

try:
    import cv2
except:
    import qtt.exceptions
    warnings.warn('could not find opencv, not all functionality available', qtt.exceptions.MissingOptionalPackageWarning)
    
from qtt import pgeometry
from qtt import pgeometry as pmatlab

from qtt.data import *

#%%

import warnings

try:
    import pylab
except:
    warnings.warn('could not load pylab')

#%%

import scipy.ndimage.filters as filters
import scipy.ndimage


def nonmaxsuppts(v, d, minval=None):
    """ Calculate maximum points in data """
    # input = np.sin(np.linspace(0, 4*np.pi, 20))
    # x = (input * 10).astype(np.int) # Makes it easier to read
    w = scipy.ndimage.maximum_filter1d(v, d, axis=0)
    pt = (w == v).nonzero()[0]
    if minval:
        pt = pt[v[pt] >= minval]
    return pt, w


def disk(radius):
    """ Create disk of specified radius """
    radius = int(radius)
    nn = 2 * radius + 1
    x, y = np.meshgrid(range(nn), range(nn))
    d = ((x - radius)**2 + (y - radius)**2 < 0.01 + radius**2).astype(int)
    return d


def localMaxima(arr, radius=1, thr=None):
    ''' Calculate local maxima in a 2D array '''
    strel = disk(radius)  # skimage.morphology.disk(radius)
    local_max = (filters.maximum_filter(arr, footprint=strel) == arr)

    if thr is not None:
        local_max[arr < thr] = 0
    return np.where(local_max)

import numpy as np
from skimage.feature import peak_local_max


def subpixelmax(A, mpos, verbose=0):
    """ Calculate maximum position with subpixel accuracy
    
    Args:
        A (1D array):
        mpos (array with integer indicess):
        verbose (int):
        
    Returns:
        subpos (array with subpixel positions):
        subval (array):
    """
    
    A=np.array(A)
    if np.array(mpos).size==0:
        # corner case
        import copy
        subpos = copy.copy(mpos)
        return subpos, []
    
    dsize=A.size
    val=A[mpos]
    
    mp = np.maximum(mpos-1,0);
    pp = np.minimum(mpos+1, dsize-1);
        
    valm=A[mp]; # value to the left
    valp=A[pp]; # value to the right
    
    cy = val;
    ay = (valm + valp)/2 - cy;
    by = ay + cy - valm
    shift = -by/(2*ay)  # Maxima of quadradic
    
    if verbose:
        print('subpixelmax: mp %d, pp %d\n', mp, pp);
        print('subpixelmax: ap %.3f, by %.3f , cy %.3f\n', ay, by, cy);
    
    shift[ay==0]=0;   # fix for flat areas
    subpos = mpos+shift

    subval= ay*shift*shift+by*shift+cy;

    if verbose:
        print('subpixelmax1d: shift %.3f\n', shift);
    
    return subpos, subval
    

def test_subpixel(fig=None):
    import qtt
    import matplotlib.pyplot as plt
    A = np.random.rand(40,)**2+1e1;
    A=qtt.algorithms.generic.smoothImage(A)

    mpos = peak_local_max(A, min_distance=3).flatten()
    subpos, subval=subpixelmax(A, mpos);

    if fig:    
        plt.figure(fig); plt.clf();
        plt.plot(np.arange(A.size), A, '.:r', label='data points');
        
        plt.plot(mpos, A[mpos], 'om', label='integer maxima');
        plt.plot(subpos, subval, '.g', markersize=15, label='subpixel maxima');
        plt.legend(numpoints=1 )  

#%%

def rescaleImage(im, imextent, mvx=None, mvy=None, verbose=0, interpolation=None, fig=None):
    """ Scale image to make pixels at specified resolution

    Args:
      im (array): input image
      imextend (list of 4 floats): coordinates of image region (x0, x1, y0, y1)
      mvx, mvy (float or None): number of units per pixel requested. If None then keep unchanged

    Returns:
       ims (array): transformed image
       H (array): transformation matrix from units to pixels. H is the homogeneous transform from original to scaled image
       mvx (float): internal data
       mvy (float): internal data
       fx (float):  internal data
       dy (float): internal data

    """
    if interpolation is None:
        interpolation = cv2.INTER_AREA
        
    dxmv = imextent[1] - imextent[0]
    dymv = imextent[3] - imextent[2]

    dx = im.shape[1]
    dy = im.shape[0]
    mvx0 = dxmv / float(dx - 1)     # current unit/pixel
    mvy0 = dymv / float(dy - 1)

    if mvy is None:
        mvy = mvy0
    if mvx is None:
        mvx = mvx0

    if im.dtype == np.int64 or im.dtype == np.int32:
        # opencv cannot handle int32 or int64 in resize
        im = im.astype(np.float32)
    # scale factors
    fw = np.abs((float(mvx0) / mvx))
    fh = np.abs((float(mvy0) / mvy))
    if verbose:
        print('rescaleImage: scale factorf x %.4f, factor y %.4f' % (fw, fh))
        print('rescaleImage: result unit/pixel x %.4f y %.4f' % (mvx, mvy))

    # scale in steps for the horizontal direction
    if fw < .5:
        fwx = fw
        fac = 1
        ims = im
        while (fwx < .5):
            ims = cv2.resize(
                ims, None, fx=.5, fy=1, interpolation=cv2.INTER_LINEAR)
            fwx *= 2
            fac *= 2
        # print('fw %f, fwx %f, fac %f' % (fw, fwx, fac))
        ims = cv2.resize(
            ims, None, fx=fac * fw, fy=fh, interpolation=interpolation)
    else:
        ims = cv2.resize(im, None, fx=fw, fy=fh, interpolation=interpolation)

    H = pgeometry.pg_transl2H(
        [-.5, -.5]) .dot(np.diag([fw, fh, 1]).dot(pgeometry.pg_transl2H([.5, .5])))

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(im, interpolation='nearest')
        plt.subplot(1, 2, 2)
        plt.imshow(ims, interpolation='nearest')
        plt.title('scaled')
    return ims, H, (mvx, mvy, fw, fh)


def test_rescale_image():
    im = np.random.rand(300, 600)
    _ = rescaleImage(im, [0, im.shape[1] - 1, 0,
                          im.shape[0] - 1], mvx=4, verbose=0, fig=None)


def scaleImage(image, display_min=None, display_max=None):
    """ Scale any image into uint8 range

        image (numpy array): input image
        display_min (float): value to map to min output range
        display_max (float): value to map to max output range
    """
    # Here I set copy=True in order to ensure the original image is not
    # modified. If you don't mind modifying the original image, you can
    # set copy=False or skip this step.
    image = np.array(image, copy=True)

    if display_min is None:
        display_min = np.percentile(image, .15)
    if display_max is None:
        display_max = np.percentile(image, 99.85)
        if display_max == display_min:
            display_max = np.max(image)
    image.clip(display_min, display_max, out=image)
    if image.dtype == np.uint8:
        image -= int(display_min)
        image = image.astype(np.float)
        image //= (display_max - display_min) / 255.
    else:
        image -= display_min
        image //= (display_max - display_min) / 255.
    image = image.astype(np.uint8)
    return image


def flowField(im, fig=None, blocksize=11, ksize=3, resizefactor=1, eigenvec=1):
    """ Calculate flowfield of an image

    Args:
        im (numpy array): input image
        fig (integer or None): number of visualization window
    Returns:
        flow (numpy array): flow
        ll (?): ??
    """
    im8 = scaleImage(im)
    im8 = cv2.resize(im8, None, fx=resizefactor, fy=resizefactor)
    h, w = im8.shape[:2]
    eigen = cv2.cornerEigenValsAndVecs(im8, blocksize, ksize=ksize)
    eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
    flow = eigen[:, :, eigenvec]
    ll = eigen[:, :, 0]

    if fig is not None:
        vis = im8.copy()
        vis[:] = (192 + np.uint32(vis)) / 2
        d = 12
        points = np.dstack(
            np.mgrid[int(d / 2):w:d, int(d / 2):h:d]).reshape(-1, 2)
        for x, y in points:
            vx, vy = np.int32(flow[y, x] * d)
            # vx,vy=int(ff*ll[y,x,0]*vx), int(ff*ll[y,x,0]*vy)
            try:
                linetype = cv2.LINE_AA
            except:
                linetype = 16  # older opencv

            cv2.line(vis, (x - vx, y - vy), (x + vx, y + vy),
                     (0, 0, 0), 1, linetype)
        cv2.imshow('input', im8)
        cv2.imshow('flow', vis)
    return flow, ll
#%%


def signedmin(val, w):
    """ Signed minimum value function """
    val = min(val, abs(w))
    val = max(val, -abs(w))
    return val


def signedabsX(val, w):
    """ Signed absolute value function """
    val = min(val, abs(w))
    val = max(val, -abs(w))
    return val


def issorted(l):
    """ Return True if the argument list is sorted """
    for i, el in enumerate(l[1:]):
        if el >= l[i - 1]:
            return False
    return True


def showFlowField(im, flow, ll=None, ff=None, d=12, fig=-1):
    """ Show flow field

    Args:
        im (numpy array): input image
        flow (numpy array): input image
        fig (integer or None): number of visualization window

    """
    im8 = scaleImage(im)
    h, w = im8.shape[:2]
    try:
        linetype = cv2.LINE_AA
    except:
        linetype = 16  # older opencv
    if ff is None:
        ff = (.01 * (h + w) / 2) / ll.max()
    if fig is not None:
        vis = im8.copy()
        vis[:] = (192 + np.uint32(vis)) / 2
        points = np.dstack(np.mgrid[d / 2:w:d, d / 2:h:d]).reshape(-1, 2)
        for x, y in points:
            x = int(x)
            y = int(y)
            vx, vy = np.int32(flow[y, x] * d)
            if ll is not None:
                vx, vy = int(ff * ll[y, x, 0] * vx), int(ff * ll[y, x, 0] * vy)
            cv2.line(
                vis, (x - vx, y - vy), (x + vx, y + vy), (0, 0, 0), 1, linetype)
        cv2.imshow('input', im8)
        cv2.imshow('flow', vis)
    return flow, ll

#%%


def showCoulombDirection(ptx, ww, im=None, dd=None, fig=100):
    """ Show direction of Coulomb peak in an image """
    if dd is None:
        pp = ptx
        ww = 10 * ww
        sigma = 5
    else:
        ptx = ptx.reshape((1, 2))
        ww = ww.reshape((1, 2))
        tr = qtt.data.image_transform(dd)

        pp = tr.pixel2scan(ptx.T).T

        pp2 = tr.pixel2scan((ptx + ww).T).T
        ww = (pp2 - pp).flatten()
        ww = 40 * ww / np.linalg.norm(ww)
        sigma = 5 * 3

    if fig is not None:
        if im is not None:
            plt.figure(fig)
            plt.clf()
            if dd is None:
                plt.imshow(im, interpolation='nearest')
            else:
                show2Dimage(im, dd, fig=fig)
        # plt.plot(pts[:,0], pts[:,1], '.m')
    if fig is not None:
        plt.figure(fig)
        hh = pylab.arrow(pp[0, 0], pp[0, 1], ww[0], ww[
                         1], linewidth=4, fc="k", ec="k", head_width=sigma / 2, head_length=sigma / 2)
        hh.set_alpha(0.8)
        # hh=pylab.arrow( ptx[0,0], ptx[0,1],-7,7, fc="g", ec="g", head_width=1.05, head_length=1 )
        # hh.set_alpha(0.4)

        plt.axis('image')


def findCoulombDirection(im, ptx, step, widthmv=8, fig=None, verbose=1):
    """ Find direction of Coulomb peak using second order derivative """
    cwidth = 2. * widthmv / np.abs(step)

    resizefactor = 2 * np.abs(step)  # resize to .5 mv/pixel
    flow, ll = flowField(im, fig=None, blocksize=int(
        1.5 * 2 * widthmv), resizefactor=resizefactor)

    ptxf = resizefactor * ptx  # [:,::-1]
    val = getValuePixel(flow, ptxf)
    l = getValuePixel(ll[:, :, 0], ptxf)

    if verbose:
        print('findCoulombDirection: initial: %s' % str(val))

    # improve estimate by taking a local average
    valr = pmatlab.rot2D(np.pi / 2) .dot(val.reshape((2, 1)))
    sidesteps = np.arange(-6, 6.01, 3).reshape((-1, 1)) * \
        np.matrix(valr.reshape((1, 2)))
    pts = ptx + .5 * np.array(sidesteps)
    ptsf = ptxf + resizefactor * sidesteps
    valx = np.array([getValuePixel(flow, p) for p in ptsf])

    a = pmatlab.directionMean(valx)
    val = np.array([np.sin(a), np.cos(a)]).flatten()
    val *= np.sign(val[1] - val[0])

    if verbose:
        print('findCoulombDirection: final: %s' % str(val))

    if fig is not None:
        showCoulombDirection(ptx, val, im=im, dd=None, fig=fig)
    return val


#%% Visualization


def extent2fullextent(extent0, im):
    """ Convert extent to include half pixel border """
    nx = im.shape[1]
    ny = im.shape[0]
    dx = (extent0[1] - extent0[0]) / (nx - 1)
    dy = (extent0[3] - extent0[2]) / (ny - 1)
    extent = [extent0[0] - dx / 2, extent0[1] + dx /
              2, extent0[2] - dy / 2, extent0[3] + dy / 2]
    return extent


def show2Dimage(im, dd, **kwargs):
    """ Show image in window

    Arguments
    ---------
    im : array
        image to show
    dd : dict with scan data
        data is used to scale the image to measurement resolution

    """
    _ = show2D(dd, im=im, **kwargs)
    return None

    try:
        extentImage, xdata, ydata, imdummy = get2Ddata(
            dd, fastscan=False, verbose=0, fig=None, midx=midx)
        mdata = dd
    except:
        extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
            dd, arrayname=None)
        extentImage = [vsweep[0], vsweep[-1], vstep[-1],
                       vstep[0]]  # matplotlib extent style
        mdata = dd.metadata

    pmatlab.cfigure(fig, facecolor=facecolor)
    plt.clf()
    if verbose >= 2:
        print('show2D: show image')
    imh = plt.imshow(im, extent=extent2fullextent(
        extentImage, im), interpolation='nearest')
    # imh=plt.imshow(im, extent=xx, interpolation='nearest')
    if units is not None:
        if 'stepdata' in mdata:
            plt.xlabel('%s (%s)' % (dd['sweepdata']['gates'][0], units))
            plt.ylabel('%s (%s)' % (dd['stepdata']['gates'][0], units))
    else:
        if 'stepdata' in mdata:
            plt.xlabel('%s' % dd['sweepdata']['gates'][0])
            plt.ylabel('%s' % dd['stepdata']['gates'][0])
    if not title is None:
        plt.title(title)
    # plt.axis('image')
    # ax=plt.gca()
    # ax.invert_yaxis()
    if colorbar:
        plt.colorbar()
    if verbose >= 2:
        print('show2D: at show')
    try:
        plt.show(block=False)
    except:
        # ipython backend does not know about block keyword...
        plt.show()
    return extentImage

if __name__ == '__main__':
    pass
    #show2Dimage(im, alldata)

#%%


def getValuePixel(imx, pt):
    """ Return interpolated pixel value in an image

    Args:
        im (numpy array): input image
        pt (numpy array): list of points
    
    Returns:
        vv (numpy array): interpolated value
    """
    ptf = np.array(pt).flatten()
    if len(imx.shape) == 2:
        vv = cv2.getRectSubPix(
            imx.astype(np.float32), (1, 1), (ptf[0], ptf[1]))
    else:
        imx = imx.astype(np.float32)
        vv = np.zeros((imx.shape[2]))
        for ii in range(imx.shape[2]):
            vv[ii] = cv2.getRectSubPix(imx[:, :, ii], (1, 1), (ptf[0], ptf[1]))
    return vv


def smoothImage(im, k=3):
    """ Super simple image smoothing

    Args:

    im : array
        input image
    k : int
        kernel size

    Returns:

    im : array
        smoothed image

    >>> ims = smoothImage(np.random.rand( 30,40) )
    """
    ndim = len(im.shape)
    k = np.ones((k,) * ndim) / k**ndim
    ims = scipy.ndimage.filters.convolve(im, k, mode='nearest')
    return ims


def detect_blobs_binary(bim):
    """ Simple blob detection in binary image

    Args:
        bim (numpy array): binary input image

    Output:
        xx (numpy array): detected blob centres

    Alternative implementation would be through findContours
    """
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = False
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 10.
    params.maxThreshold = 200.
    params.thresholdStep = 200.
    # params.minArea=1
    params.minDistBetweenBlobs = 1.
    params.minRepeatability = 1

    try:
        detector = cv2.SimpleBlobDetector_create(params)
    except:
        # old (OpenCV 2) style
        detector = cv2.SimpleBlobDetector(params)

    bim = 50 * (bim > 0).astype(np.uint8).copy()

    # Detect blobs.
    keypoints = detector.detect(bim)
    xx = np.array([p.pt for p in keypoints])
    # print(xx)

    return xx


def makeCoulombFilter(theta0=-np.pi / 4, step=1, ne=0, dphi=np.pi / 4, widthmv=10, lengthmv=50., verbose=0, fig=None):
    """ Create filters to detect Coulomb peaks """
    cwidth = 2. * widthmv / np.abs(step)
    clength = .5 * lengthmv / np.abs(step)

    # odd number, at least twice the length
    ksize = 2 * int(np.ceil(clength)) + 1

    filters = []
    angles = np.arange(-ne * dphi + theta0,
                       theta0 + ne * dphi + dphi / 2, dphi)
    for ii, theta in enumerate(angles):
        if verbose:
            print('ii %d: theta %.2f' % (ii, np.rad2deg(theta)))
        kk = cv2.getGaborKernel(
            (ksize, ksize), sigma=clength / 2, theta=theta, lambd=cwidth, gamma=1, psi=0 * np.pi / 2)
        # kk=gabor_kernel(.05,theta,5., sigma_x=5., sigma_y=5., offset=0*pi/2)
        kk = np.real(kk)
        filters.append(kk)
        if fig is not None:
            plt.figure(fig + ii)
            plt.clf()
            plt.imshow(kk, interpolation='nearest')
            plt.colorbar()
            plt.axis('image')
    return filters, angles, (cwidth, clength)


def weightedCentroid(im, contours, contourIdx, fig=None):
    """ Calculate weighted centroid from a contour

    The contours are in OpenCV format
    """
    mask = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    # cv2.FILLED = cv2.cv.CV_FILLED = -1
    cv2.drawContours(
        mask, contours, contourIdx=contourIdx, color=1, thickness=-1)

    yy, xx = np.meshgrid(range(im.shape[1]), range(im.shape[0]))
    xyw = np.array(
        [(im * mask * yy).sum(), (im * mask * xx).sum()]) / (mask * im).sum()
    if fig is not None:
        # pmatlab.imshowz(mask, interpolation='nearest')
        yx = np.array([(mask * xx).sum(), (mask * yy).sum()]) / mask.sum()
        plt.figure(11)
        plt.clf()
        plt.imshow(im, interpolation='nearest')
        plt.plot(yx[1], yx[0], '.m', markersize=12)
        plt.plot(xyw[0], xyw[1], '.g', markersize=17)
    return xyw
