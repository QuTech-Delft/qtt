import math
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

try:
    import cv2
except ImportError:
    import qtt.exceptions
    warnings.warn('could not find opencv, not all functionality available',
                  qtt.exceptions.MissingOptionalPackageWarning)

import qtt.pgeometry as pgeometry
from qtt.algorithms.generic import smoothImage
from qtt.algorithms.misc import polyfit2d, polyval2d
from qtt.measurements.scans import fixReversal
from qtt.utilities.tools import diffImage, diffImageSmooth

# %%


def fitBackground(im, smooth=True, fig=None, order=3, verbose=1, removeoutliers=False, returndict=None):
    """ Fit smooth background to 1D or 2D image.

    Args:
        im (array): input image.

    Returns
        vv (array): estimated background.

    """
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
        if verbose:
            print('fitBackGround: inliers %d/%d (std %.2f)' %
                  (gidx.sum(), gidx.size, ww.std()))
        s2d = polyfit2d(xxf[gidx], yyf[gidx], imsf[gidx], order=order)
        vv = polyval2d(xx, yy, s2d)

    if fig is not None:
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
            plt.title('fitBackground: interpolation')
            plt.subplot(3, 1, 3)
            plt.imshow(im - vv, interpolation='nearest')
            plt.axis('image')
            plt.title('fitBackground: difference')

    if returndict is not None:
        warnings.warn('please do not use this feature any more')
        returndict['xx'] = xx
        returndict['yy'] = yy
        returndict['ims'] = ims
    return vv


def cleanSensingImage(im, dy=0, sigma=None, order=3, fixreversal=True, removeoutliers=False, verbose=0):
    """ Clean up image from sensing dot.

    Args:
        im (numpy array) TODO.
        dy (int or str): direction for differentiation.
        order (int):  TODO.
        fixreversal (bool): TODO.
        removeoutliers (bool) TODO.

    Returns:
        ww (image): processed image.
    """
    verbose = int(verbose)
    removeoutliers = bool(removeoutliers)
    im = np.asarray(im)
    if sigma is None:
        imx = diffImage(im, dy=dy, size='same')
    else:
        imx = diffImageSmooth(im, dy=dy, sigma=sigma)
    if order >= 0:
        vv = fitBackground(imx, smooth=True, verbose=verbose, fig=None,
                           order=int(order), removeoutliers=removeoutliers)
        ww = (imx - vv).copy()
    else:
        ww = imx.copy()
    if fixreversal:
        ww = fixReversal(ww, verbose=verbose)
    return ww


def _showIm(ims, fig=1, title=''):
    """ Show image with nearest neighbor interpolation and axis scaling."""
    matplotlib.pyplot.figure(fig)
    matplotlib.pyplot.clf()
    matplotlib.pyplot.imshow(ims, interpolation='nearest')
    matplotlib.pyplot.axis('image')


@pgeometry.static_var("scaling0", np.diag([1., 1, 1]))
def evaluateCross(param, im, verbose=0, fig=None, istep=1, istepmodel=1, linewidth=2,
                  usemask=False, use_abs=False, w=2.5):
    """ Calculate cross matching score.

    Args:
        param (array or list): used by createCross to create image template.
        im (numpy array): TODO.

    Returns:
        cost, patch, cdata, tuple.

    See also:
        createCross.

    """
    samplesize = [int(im.shape[1] * istep / istepmodel), int(im.shape[0] * istep / istepmodel)]
    param = np.array(param)
    aa = param[3:]

    H = evaluateCross.scaling0.copy()
    H[0, 0] = istep / istepmodel
    H[1, 1] = istep / istepmodel

    dsize = (samplesize[0], samplesize[1])
    patch = cv2.warpPerspective(im.astype(np.float32), H, dsize, None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, -1)

    if verbose:
        print('evaluateCross: patch shape %s' % (patch.shape,))
    modelpatch, cdata = createCross(param, samplesize, centermodel=False, istep=istepmodel, verbose=verbose >= 2, w=w)
    (cc, lp, hp, ip, op, _, _, _) = cdata

    if use_abs:
        dd = np.abs(patch) - modelpatch
    else:
        dd = patch - modelpatch

    if usemask:
        # distance centre mask
        imtmp = 10 + 0 * modelpatch.copy()
        imtmp[int(imtmp.shape[1] / 2), int(imtmp.shape[0] / 2)] = 0
        mask = scipy.ndimage.distance_transform_edt(imtmp)
        mask = 1 - .75 * mask / mask.max()

        dd = dd * mask

    cost = np.linalg.norm(dd)

    # area of intersection
    rr = np.array([[0., im.shape[1]], [0, im.shape[0]]])
    ppx = pgeometry.region2poly(np.array([[0, samplesize[0]], [0., samplesize[1]]]))
    ppimage = pgeometry.region2poly(rr)
    pppatch = pgeometry.projectiveTransformation(H, ppimage)
    ppi = pgeometry.polyintersect(ppx.T, pppatch.T).T
    A = pgeometry.polyarea(ppi.T)
    A0 = pgeometry.polyarea(ppx.T)

    # special rules
    if np.abs(A / A0) < .85:
        if verbose:
            print('  add cost for A/A0: A %f A0 %f' % (A, A0))
        cost += 4000
    if aa[0] < 0 or aa[0] > np.pi / 2 - np.deg2rad(5):
        cost += 10000
    if aa[1] < np.pi or aa[1] > 3 * np.pi / 2:
        if verbose:
            print('  add cost for alpha')
        cost += 10000
    if aa[2] < np.pi or aa[2] > 3 * np.pi / 2:
        if verbose:
            print('  add cost for alpha')
        cost += 10000
    if aa[3] < 0 or aa[3] > np.pi / 2:
        if verbose:
            print('  add cost for alpha')
        cost += 10000

    if 1:
        ccim = (np.array(im.shape) / 2 + .5) * istep
        tmp = np.linalg.norm(ccim - param[0:2])
        dcost = 2000 * pgeometry.logistic(tmp, np.mean(ccim), istep)
        if verbose:
            print('  add cost for image cc: %.1f' % dcost)
        cost += dcost
    if pgeometry.angleDiff(aa[0], aa[1]) < np.deg2rad(30):
        if verbose:
            print('  add cost for angle diff')
        cost += 1000
    if pgeometry.angleDiff(aa[2], aa[3]) < np.deg2rad(30):
        if verbose:
            print('  add cost for angle diff')
        cost += 1000

    if pgeometry.angleDiffOri(aa[0], aa[2]) > np.deg2rad(10):
        cost += 10000
    if pgeometry.angleDiffOri(aa[1], aa[3]) > np.deg2rad(10):
        cost += 10000
    if param[2] < 0:
        if verbose:
            print('  add cost for negative param')
        cost += 10000

    if np.abs(param[2]) > 7.:
        if verbose:
            print('  add cost large param[2]')
        cost += 10000

    if np.abs(param[2] - 10) > 8:
        cost += 10000

    if len(param) > 7:
        if np.abs(pgeometry.angleDiff(param[7], np.pi / 4)) > np.deg2rad(30):
            cost += 10000

    if fig is not None:
        _showIm(patch, fig=fig)
        plt.title('Image patch: cost %.1f: istep %.2f' % (cost, istepmodel))
        pgeometry.addfigurecopy(fig=fig)
        plt.plot([float(lp[0]), float(hp[0])], [float(lp[1]), float(hp[1])], '.--m',
                 linewidth=linewidth, markersize=10, label='transition line')
        plt.plot(cc[0], cc[1], '.m', markersize=12)
        for ii in range(4):
            if ii == 0:
                lbl = 'electron line'
            else:
                lbl = None
            plt.plot([op[ii, 0], ip[ii, 0]], [op[ii, 1], ip[ii, 1]], '.-',
                     linewidth=linewidth, color=[0, .7, 0], label=lbl)
            pgeometry.plotLabels(np.array((op[ii, :] + ip[ii, :]) / 2).reshape((2, -1)), '%d' % ii)
        if verbose >= 1:
            _showIm(modelpatch, fig=fig + 1)
            plt.title('Model patch: cost %.1f' % cost)
            _showIm(np.abs(dd), fig=fig + 2)
            plt.title('diff patch: cost %.1f' % cost)
            plt.colorbar()
            plt.show()

    if verbose:
        print('evaluateCross: cost %.4f' % cost)
    return cost, patch, cdata, (H, )
    pass


def createCross(param, samplesize, l=20, w=2.5, lsegment=10, H=100, scale=None,
                lines=range(4), istep=1, centermodel=True, linesegment=True, addX=True, verbose=0):
    """ Create a cross model.
    The parameters are [x, y, width, alpha_1, ..., alpha_4, [rotation of polarization line] ]
    With the optional parameters psi (angle of transition line).
    The x, y, width are in mV. Angles in radians.

    Args:
        param (array): parameters of the model.
        samplesize (int): size of image patch in pixels.
        l, w, lsegment (float): parameters of the model in mV?. lsegment is the length of the 4 addition lines
                w is width of lines in the model.
                l is not used by default.
        istep (float): scan resolution in pixel/mV.
        scale (None): parameter not used any more.
        addX (bool): if True add polarization line to model.
        H (float): intensity of cross.
        linesegment (bool): if True create line segments instead of full lines.

    Returns:
        modelpatch, (cc, lp, hp, ip, opr, w, H, lsegment): return data.

    """
    aa = param[3:7]

    psi = np.pi / 4
    if len(param) > 7:
        psi = param[7]
        if verbose:
            print('createCross: psi set to %.1f [def]' % np.rad2deg(psi))

    if samplesize is None:
        cc = param[0:2].reshape((2, 1))
    else:
        samplesize = np.array(samplesize).flatten()
        if centermodel:
            cc = np.array(samplesize).reshape((2, 1)) / 2 - .5
        else:
            cc = np.array(param[0:2] / istep).reshape((2, 1))

    # lp and hp are the triple points
    lp = cc + pgeometry.rot2D(psi + np.pi / 2).dot(np.array([[param[2] / istep], [0]]))
    hp = cc - pgeometry.rot2D(psi + np.pi / 2).dot(np.array([[param[2] / istep], [0]]))

    op = np.zeros((5, 2))
    opr = np.zeros((4, 2))
    ip = np.zeros((5, 2))
    # loop over all 4 line segments
    for ii, a in enumerate(aa):
        if ii == 0 or ii == 1:
            ip[ii].flat = lp
        else:
            ip[ii].flat = hp
        opr[ii] = ip[ii] + ((lsegment / istep) * pgeometry.rot2D(a).dot(np.array([[1], [0]]))).flat

    if samplesize is not None:
        modelpatch = np.zeros([samplesize.flat[1], samplesize.flat[0]])

        for ii in lines:
            if linesegment:
                x0 = ip[ii]
                x1x = np.array(x0 + lsegment / istep * (pgeometry.rot2D(aa[ii]).dot(np.array([[1], [0]]))).T).flatten()

                lineSegment(modelpatch, x0=x0, x1=x1x, w=w / istep, l=None, H=H)
            else:
                raise Exception('code not used any more?')
                # semiLine(modelpatch, x0=ip[ii], theta=aa[ii], w=w / istep, l=l / istep, H=H)
        if addX:
            lx = np.linalg.norm(hp - lp, ord=2)
            lineSegment(modelpatch, x0=np.array(hp.reshape((2, 1))),
                        x1=np.array(lp.reshape((2, 1))), w=w / istep, l=lx, H=H)

    else:
        modelpatch = None
    modelpatch = np.minimum(modelpatch, H)
    return modelpatch, (cc, lp, hp, ip, opr, w, H, lsegment)

# %%


def fitModel(param0, imx, verbose=1, cfig=None, ksizemv=41, istep=None,
             istepmodel=0.5, cb=None, use_abs=False, model_line_width=2.5):
    """ Fit model of an anti-crossing .

    This is a wrapper around evaluateCross and the scipy optimization routines.

    Args:
        param0 (array): parameters for the anti-crossing model.
        imx (array): input image.

    """
    def costfun(param0): return evaluateCross(param0, imx, fig=None, istepmodel=istepmodel, usemask=False,
                                              istep=istep, use_abs=use_abs, linewidth=model_line_width)[0]

    vv = []

    def fmCallback(plocal, pglobal):
        """ Helper function to store intermediate results """
        vv.append((plocal, pglobal))
    if cfig is not None:
        def cb_funcion(x):
            return fmCallback(x, None)
        cb = cb_funcion

    if 1:
        # simple brute force
        ranges = list(slice(x, x + .1, 1) for x in param0)
        for ii in range(2):
            ranges[ii] = slice(param0[ii] - 13, param0[ii] + 13, 1)
        ranges = tuple(ranges)
        res = scipy.optimize.brute(costfun, ranges)
        paramy = res
    else:
        paramy = param0
    res = scipy.optimize.minimize(costfun, paramy, method='nelder-mead',
                                  options={'maxiter': 1200, 'maxfev': 101400, 'xatol': 1e-8, 'disp': verbose >= 2},
                                  callback=cb)

    if verbose:
        print('fitModel: score %.2f -> %.2f' % (costfun(param0), res.fun))
    return res
# %%


def lineSegment(im, x0, x1=None, theta=None, w=2, l=12, H=200, ml=0):
    """ Plot half-line into image .

    >>> lineSegment(np.zeros( (160,240)), [60,40], [70,40], w=10, l=60)
    >>> lineSegment(np.zeros( (160,240)), [60,40], theta=np.deg2rad(20), w=10, l=60)

    """
    x0 = np.array(x0).flatten()
    if x1 is None:
        thetar = -theta
        dx = l
    else:
        x1 = np.array(x1).flatten()
        theta = x0 - x1
        theta = np.arctan2(theta[1], theta[0]) + np.pi
        thetar = -theta

        dx = np.linalg.norm(x0 - x1)

    xx0, yy0 = np.meshgrid(np.arange(im.shape[1]) - x0[0], np.arange(im.shape[0]) - x0[1])
    xx0 = xx0.astype(np.float32)
    yy0 = yy0.astype(np.float32)

    xxr = math.cos(thetar) * xx0 - math.sin(thetar) * yy0
    yyr = math.sin(thetar) * xx0 + math.cos(thetar) * yy0
    yyr = pgeometry.signedmin(yyr, w / 2.)

    data = H * np.cos((np.pi / w) * yyr) * (pgeometry.smoothstep(xxr, ml, 4)) * (1 - pgeometry.smoothstep(xxr, dx, 4))

    im += data
    return im


def semiLine(im, x0, theta, w, l, H=200, dohalf=True):
    """ Plot half-line into image .

    Args:
        im (array)
        x0 (array): starting point of semi-line.
        theta (float): angle in radians.
        w (float): width.
        l (float): length of line segment.
        H (float): intensity of line segment.
        dohalf (bool): add smoothing?

    >>> im=semiLine(np.zeros( (160,240)), [60,40], theta=np.deg2rad(20), w=10, l=60)
    >>> plt.imshow(im)

    """
    thetar = -theta
    xx0, yy0 = np.meshgrid(np.arange(im.shape[1]) - x0[0], np.arange(im.shape[0]) - x0[1])
    xx0 = xx0.astype(np.float32)
    yy0 = yy0.astype(np.float32)

    xxr = math.cos(thetar) * xx0 - math.sin(thetar) * yy0
    yyr = math.sin(thetar) * xx0 + math.cos(thetar) * yy0
    yyr = pgeometry.signedmin(yyr, w / 2.)

    data = H * np.cos((np.pi / w) * yyr)
    if dohalf:
        data *= (pgeometry.smoothstep(xxr, 0, 10))

    im += data
    return im


def __calcSlope(points):
    """ Calculate slope between two points."""
    q = -np.diff(points, axis=1)
    psi = math.atan2(q[1], q[0])
    slope = q[1] / q[0]

    return psi, slope


def Vtrace(cdata, param, fig=None):
    """ Calculate position of next V-trace from fitted model .

    Args:
        cdata (?): TODO
        param (?): TODO
        fit (None or integer): figure handle.

    """
    cc = cdata[0]
    psi = param[-1]

    q = np.array([10, 0]).reshape((2, 1))
    p1 = cc + pgeometry.rot2D(psi).dot(q)
    p2 = cc + pgeometry.rot2D(np.pi + psi).dot(q)
    pp = np.array(np.hstack((p1, cc, p2)))
    pp = np.array(np.hstack((p1, p2)))
    if fig is not None:
        plt.figure(fig)

        pgeometry.plotPoints(pp, '--k', markersize=20, linewidth=3, label='scan line')
        pgeometry.plotPoints(pp, '.y', markersize=20)
        plt.legend(numpoints=1, fontsize=14, loc=0)
    psi, slope = __calcSlope(pp)
    return pp, cc, slope
