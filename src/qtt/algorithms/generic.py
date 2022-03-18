""" Various functions """

import copy
import warnings
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy

try:
    import cv2
except:
    import qtt.exceptions

    warnings.warn('could not find opencv, not all functionality available',
                  qtt.exceptions.MissingOptionalPackageWarning)

import warnings

import qtt.utilities.tools
from qtt import pgeometry
from qtt.pgeometry import scaleImage

# %%


try:
    import pylab
except:
    warnings.warn('could not load pylab')

# %%

import scipy.ndimage as filters


def nonmaxsuppts(v, d, minval=None):
    """ Calculate maximum points in data """
    # input = np.sin(np.linspace(0, 4*np.pi, 20))
    # x = (input * 10).astype(int) # Makes it easier to read
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
    d = ((x - radius) ** 2 + (y - radius) ** 2 < 0.01 + radius ** 2).astype(int)
    return d


def localMaxima(arr, radius=1, thr=None):
    """ Calculate local maxima in a 2D array
    """
    strel = disk(radius)
    local_max = (filters.maximum_filter(arr, footprint=strel) == arr)

    if thr is not None:
        local_max[arr < thr] = 0
    return np.where(local_max)


def subpixelmax(A, mpos, verbose=0):
    """ Calculate maximum position with subpixel accuracy

    For each position specified by mpos this method fits a parabola through 3 points and calculates the
    maximum position of the parabola.

    Args:
        A (1D array): Input data
        mpos (array with integer indices): Positions in the array A with maxima
        verbose (int): Verbosity level

    Returns:
        subpos (array): Array with subpixel positions
        subval (array): Values at maximum positions
    """

    A = np.array(A)
    mpos = np.array(mpos)
    if np.array(mpos).size == 0:
        # corner case
        subpos = copy.copy(mpos)
        return subpos, []

    dsize = A.size
    val = A[mpos]

    mp = np.maximum(mpos - 1, 0)
    pp = np.minimum(mpos + 1, dsize - 1)

    valm = A[mp]  # value to the left
    valp = A[pp]  # value to the right

    cy = val
    ay = (valm + valp) / 2 - cy
    by = ay + cy - valm

    if np.any(ay == 0):
        shift = 0 * ay
    else:
        shift = -by / (2 * ay)  # Maxima of quadradic

    if verbose:
        print('subpixelmax: mp %d, pp %d\n' % (mp, pp))
        print('subpixelmax: ap %.3f, by %.3f , cy %.3f\n' % (ay, by, cy))

    subpos = mpos + shift

    subval = ay * shift * shift + by * shift + cy

    if verbose:
        print('subpixelmax1d: shift %.3f\n', shift)

    return subpos, subval


def rescaleImage(im, imextent, mvx=None, mvy=None, verbose=0, interpolation=None, fig=None):
    """ Scale image to make pixels at specified resolution

    Args:
      im (array): input image
      imextent (list of 4 floats): coordinates of image region (x0, x1, y0, y1)
      mvx, mvy (float or None): number of units per pixel requested. If None then keep unchanged

    Returns:
       ims (array): transformed image
       H (array): transformation matrix from units to pixels. H is the homogeneous transform from original to
                  scaled image
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
    mvx0 = dxmv / float(dx - 1)  # current unit/pixel
    mvy0 = dymv / float(dy - 1)

    if mvy is None:
        mvy = mvy0
    if mvx is None:
        mvx = mvx0

    if im.dtype == np.int64 or im.dtype == np.int32:
        # opencv cannot handle int32 or int64 in resize
        im = im.astype(np.float32)
    # scale factors
    fw = np.abs(float(mvx0) / mvx)
    fh = np.abs(float(mvy0) / mvy)
    if verbose:
        print('rescaleImage: scale factorf x %.4f, factor y %.4f' % (fw, fh))
        print('rescaleImage: result unit/pixel x %.4f y %.4f' % (mvx, mvy))

    # scale in steps for the horizontal direction
    if fw < .5:
        fwx = fw
        fac = 1
        ims = im
        while fwx < .5:
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
        [-.5, -.5]).dot(np.diag([fw, fh, 1]).dot(pgeometry.pg_transl2H([.5, .5])))

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(im, interpolation='nearest')
        plt.subplot(1, 2, 2)
        plt.imshow(ims, interpolation='nearest')
        plt.title('scaled')
    return ims, H, (mvx, mvy, fw, fh)


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


# %%


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


# %%


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
    if fig is not None:
        plt.figure(fig)
        hh = pylab.arrow(pp[0, 0], pp[0, 1], ww[0], ww[
            1], linewidth=4, fc="k", ec="k", head_width=sigma / 2, head_length=sigma / 2)
        hh.set_alpha(0.8)

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
    valr = pgeometry.rot2D(np.pi / 2).dot(val.reshape((2, 1)))
    sidesteps = np.arange(-6, 6.01, 3).reshape((-1, 1)) * \
        np.matrix(valr.reshape((1, 2)))
    pts = ptx + .5 * np.array(sidesteps)
    ptsf = ptxf + resizefactor * sidesteps
    valx = np.array([getValuePixel(flow, p) for p in ptsf])

    a = pgeometry.directionMean(valx)
    val = np.array([np.sin(a), np.cos(a)]).flatten()
    val *= np.sign(val[1] - val[0])

    if verbose:
        print('findCoulombDirection: final: %s' % str(val))

    if fig is not None:
        showCoulombDirection(ptx, val, im=im, dd=None, fig=fig)
    return val


# %% Visualization


def extent2fullextent(extent0, im):
    """ Convert extent to include half pixel border """
    nx = im.shape[1]
    ny = im.shape[0]
    dx = (extent0[1] - extent0[0]) / (nx - 1)
    dy = (extent0[3] - extent0[2]) / (ny - 1)
    extent = [extent0[0] - dx / 2, extent0[1] + dx /
              2, extent0[2] - dy / 2, extent0[3] + dy / 2]
    return extent


def getValuePixel(imx, pt):
    """ Return interpolated pixel value in an image

    Args:
        imx (numpy array): input image
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
        vv = np.zeros(imx.shape[2])
        for ii in range(imx.shape[2]):
            vv[ii] = cv2.getRectSubPix(imx[:, :, ii], (1, 1), (ptf[0], ptf[1]))
    return vv


def smoothImage(im, k=3):
    """ Super simple image smoothing

    Args:
        im (array): input image
        k (int): kernel size

    Returns:
        ims (array): smoothed image

    Example:
        ims = smoothImage(np.random.rand( 30,40))
    """
    ndim = len(im.shape)
    kernel = np.ones((k,) * ndim) / k ** ndim
    ims = scipy.ndimage.convolve(im, kernel, mode='nearest')
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
    cv2.drawContours(
        mask, contours, contourIdx=contourIdx, color=1, thickness=-1)

    yy, xx = np.meshgrid(range(im.shape[1]), range(im.shape[0]))
    xyw = np.array(
        [(im * mask * yy).sum(), (im * mask * xx).sum()]) / (mask * im).sum()
    if fig is not None:
        yx = np.array([(mask * xx).sum(), (mask * yy).sum()]) / mask.sum()
        plt.figure(11)
        plt.clf()
        plt.imshow(im, interpolation='nearest')
        plt.plot(yx[1], yx[0], '.m', markersize=12)
        plt.plot(xyw[0], xyw[1], '.g', markersize=17)
    return xyw


def boxcar_filter(signal: np.ndarray, kernel_size: Union[np.ndarray, Tuple[int]]) -> np.ndarray:
    """ Perform boxcar filtering on an array.
    At the edges, the edge value is replicated beyond the edge as needed by the size of the kernel.
    This is the 'nearest' mode of scipy.ndimage.convolve. For details, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html?highlight=mode


    Args:
        signal: An array containing the signal to be filtered.
        kernel_size: Multidimensional size of the filter box. Must have the same number of dimensions as the signal.

    Returns:
        Array containing the filtered signal.
    """

    signal = np.asarray(signal)
    if not isinstance(kernel_size, np.ndarray):
        kernel_size = np.array(kernel_size, dtype=np.int_)

    if len(kernel_size) != len(signal.shape):
        raise RuntimeError('Number of dimensions of kernel (%d) not equal to number of dimension of input signal (%d)' %
                           (len(kernel_size), len(signal.shape)))
    if np.any(kernel_size <= 0):
        raise RuntimeError('Kernel sizes must be > 0')

    if signal.dtype.kind in ('i', 'u'):
        filtered_signal = signal.astype(np.float64)
    else:
        filtered_signal = signal

    if np.prod(kernel_size) == 1:
        filtered_signal = np.array(signal)
    else:
        boxcar_kernel = np.ones(kernel_size, dtype=np.float64) / np.float64(np.prod(kernel_size))
        filtered_signal = scipy.ndimage.convolve(filtered_signal, boxcar_kernel, mode='nearest')

    return filtered_signal
