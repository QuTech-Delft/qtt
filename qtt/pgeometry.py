# -*- coding: utf-8 -*-
"""

pgeometry
-------

A collection of usefull functions.

For additional options also see
`numpy <http://numpy.scipy.org/>`_ and `matplotlib <http://matplotlib.sourceforge.net/>`_.

:platform: Unix, Windows

.. doctest::

  >>> choose(6,2)
  15

Additions:
    Copyright 2012-2016   TNO
Original code:
    Copyright 2011 Pieter Eendebak <pieter.eendebak@gmail.com>

@author: eendebakpt
"""


# make python2/3 compatible
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

#%% Load necessary packages
import os
import sys
import tempfile
import math
import numpy as np
import time
import platform
import warnings
import pickle
import re
import logging
import pkgutil
import subprocess

__version__ = '0.5'

#%% Load pyqside or pyqt4
# We want to do this before loading matplotlib


def qtModules(verbose=0):
    """ Return list of Qt modules loaded """
    _ll = sys.modules.keys()
    qq = [x for x in _ll if x.startswith('Py')]
    if verbose:
        print('qt modules: %s' % str(qq))
    return qq

try:
    _applocalqt = None

    try:
        # by default use qtpy to import Qt
        import qtpy
        _haveqtpy = True
    except:
        _haveqtpy = False
        pass

    _ll = sys.modules.keys()
    _pyside = len([_x for _x in _ll if _x.startswith('PySide.QtGui')]) > 0
    _pyqt4 = len([_x for _x in _ll if _x.startswith('PyQt4.QtGui')]) > 0
    _pyqt5 = len([_x for _x in _ll if _x.startswith('PyQt5.QtGui')]) > 0

    if _pyside:
        import PySide.QtCore as QtCore
        import PySide.QtGui as QtGui
        import PySide.QtGui as QtWidgets
        from PySide.QtCore import Slot as Slot
        from PySide.QtCore import QObject
        from PySide.QtCore import Signal
    else:
        if _pyqt4:
            import PyQt4.QtCore as QtCore
            import PyQt4.QtGui as QtGui
            import PyQt4.QtGui as QtWidgets
            from PyQt4.QtCore import pyqtSlot as Slot
            from PyQt4.QtCore import QObject
            from PyQt4.QtCore import pyqtSignal as Signal
            # print('pmatlab: using PyQt4')
        elif _pyqt5:
            import PyQt5.QtCore as QtCore
            import PyQt5.QtGui as QtGui
            import PyQt5.QtWidgets as QtWidgets
            from PyQt5.QtCore import pyqtSlot as Slot
            from PyQt5.QtCore import QObject
            from PyQt5.QtCore import pyqtSignal as Signal
            logging.debug('pgeometry: using PyQt5')
        else:
            if 1:
                import PyQt4.QtCore as QtCore
                import PyQt4.QtGui as QtGui
                import PyQt4.QtGui as QtWidgets
                from PyQt4.QtCore import pyqtSlot as Slot
                from PyQt4.QtCore import QObject
                from PyQt4.QtCore import pyqtSignal as Signal
            else:
                import PySide.QtCore as QtCore
                import PySide.QtGui as QtGui
                import PySide.QtGui as QtWidgets
                from PySide.QtCore import Slot as Slot
                from PySide.QtCore import QObject
                from PySide.QtCore import Signal

    try:
        # the normal creation of a Qt application instance can fail
        # this is an additional mechanism
        import pyqtgraph
        _applocalqt = pyqtgraph.mkQApp()
    except:
        pass
    _applocalqt = QtWidgets.QApplication.instance()
    # print('pgeometry: _applocalqt %s' % _applocalqt )
    if _applocalqt is None:
        _applocalqt = QtWidgets.QApplication([])

    def slotTest(txt):
        """ Helper function for Qt slots """
        class slotObject(QtCore.QObject):

            def __init__(self, txt):
                QObject.__init__(self)
                self.txt = txt

            @Slot()
            def slot(self, v=None):
                if v is None:
                    print('slotTest: %s' % self.txt)
                else:
                    print('slotTest: %s: %s' % (self.txt, str(v)))
        s = slotObject(txt)
        return s.slot

    class signalTest(QObject):

        """ Helper function for Qt signals """
        s = Signal()

        def __init__(self):
            QObject.__init__(self)

        def go(self):
            self.s.emit()

except Exception as ex:
    logging.info('pgeometry: load qt: %s' % ex)
    print(ex)
    print('pgeometry: no Qt found')

#%% Load other modules
try:
    import pylab
    import pylab as p
    # print('after pylab'); qtModules(1)

except Exception as inst:
    print(inst)
    print('could not import pylab, not all functionality available...')
    pass

try:
    import matplotlib.pyplot as plt
    import matplotlib
    # needed for 3d plot points, do not remove!
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except:
        pass
except Exception as inst:
    # print(inst)
    warnings.warn(
        'could not import matplotlib, not all functionality available...')
    plt = None
    pass

try:
    import skimage.filters
except Exception as inst:
    warnings.warn(
        'could not load skimage.filters, not all functionality is available')
    pass


try:
    import cv2
    _haveOpenCV = True
except:
    _haveOpenCV = False
    warnings.warn('could not find OpenCV, not all functionality is available')
    pass


#%%
try:
    import resource

    def memUsage():
        # http://chase-seibert.github.io/blog/2013/08/03/diagnosing-memory-leaks-python.html
        print('Memory usage: %s (mb)' %
              ((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024., ))
except:
    def memUsage():
        print('Memory usage: ? (mb)')

#%% Try numba support
try:
    from numba import jit, autojit
except:
    def autojit(original_function):
        """ dummy autojit decorator """
        def dummy_function(*args, **kwargs):
            return original_function(*args, **kwargs)
        return dummy_function
    pass


#%% Utils

from functools import wraps

def package_versions(verbose=1):   
    
    print('numpy.__version__ %s' % numpy.__version__)
    print('scipy.__version__ %s' % scipy.__version__)
    try:
        import cv2
        print('cv2.__version__ %s' % cv2.__version__)
    except:
        pass
    try:
        import qtpy
        import qtpy.QtCore
        print('qtpy.API_NAME %s'  % (qtpy.API_NAME) )
        print('qtpy.QtCore %s'  % (qtpy.QtCore) )
        print('qtpy.QtCore.__version__ %s'  % (qtpy.QtCore.__version__) )
    except:
        pass
    try:
        import sip
        print('sip %s'  % sip.SIP_VERSION_STR)
    except:
        pass
def freezeclass(cls):
    """ Decorator to freeze a class """
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and not hasattr(self, key):
            print("Class {} is frozen. Cannot set {} = {}"
                  .format(cls.__name__, key, value))
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


def static_var(varname, value):
    """ Helper function to create a static variable """
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("time", 0)
def tprint(string, dt=1, output=False):
    """ Print progress of a loop every dt seconds """
    if (time.time() - tprint.time) > dt:
        print(string)
        tprint.time = time.time()
        if output:
            return True
        else:
            return
    else:
        if output:
            return False
        else:
            return


def partiala(method, **kwargs):
    ''' Function to perform functools.partial on named arguments '''
    def t(x):
        return method(x, **kwargs)
    return t


def createCheckerboard(wh, outfile=None, fac=100, fig=None):
    """ Create a checkerboard image of specific size
    Example:
        >>> B,Br=createCheckerboard( [4,6], fac=1, fig=100)

    """
    B = 225 * np.ones((wh[0] + 2, wh[1] + 2))
    B[1:-1, 1:-1] = 0
    for ii in range(B.shape[0]):
        for jj in range(B.shape[1]):
            if ((ii + jj) % 2 == 0) and B[ii, jj] == 0:
                B[ii, jj] = 255

    if fig:
        plt.figure(10)
        plt.clf()
        plt.gray()
        plt.imshow(B, interpolation='nearest')

    Br = cv2.resize(B, (B.shape[1] * fac, B.shape[0] * fac), interpolation=0)

    if not outfile is None:
        #        cv2.imwrite('/home/eendebakpt/tmp/cb.png', B);
        cv2.imwrite(outfile, Br)
    return B, Br


def setFontSizes(labelsize=20, fsize=17, titlesize=None, ax=None,):
    """ Update font sizes for a plot """
    if ax is None:
        ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)

    for x in [ax.xaxis.label, ax.yaxis.label]:  # ax.title,
        x.set_fontsize(labelsize)

    plt.tick_params(axis='both', which='major', labelsize=fsize)
    # plt.tick_params(axis='both', which='minor', labelsize=8)

    if titlesize is not None:
        ax.title.set_fontsize(titlesize)

    plt.draw()


def plotCostFunction(fun, x0, fig=None, marker='.', scale=1, c=None):
    """

    Example with variation of Booth's function:

    >>> fun = lambda x: 2*(x[0]+2*x[1]-7)**2 + (2*x[0]+x[1]-5)**2
    >>> plotCostFunction(fun, np.array([1,3]), fig=100, marker='-')

    """
    x0 = np.array(x0).astype(float)
    nn = x0.size
    if fig is not None:
        plt.figure(fig)

    scale = np.array(scale)
    if scale.size == 1:
        scale = scale * np.ones(x0.size)
    tt = np.arange(-1, 1, 5e-2)

    for ii in range(nn):
        val = np.zeros(tt.size)
        for jj in range(tt.size):
            x = x0.copy()
            x[ii] += scale[ii] * tt[jj]
            val[jj] = fun(x)
        if c is None:
            plt.plot(tt, val, marker)
        else:
            plt.plot(tt, val, marker, color=c[ii])
    plt.xlabel('Scaled variables')
    plt.ylabel('Value of cost function')


class fps_t:

    """ Class for framerate measurements
    Example usage:

    >>> fps = fps_t(nn=8)
    >>> for kk in range(12):
    ...       fps.addtime( .2*kk )
    >>> fps.show()
    framerate: 5.000
    """

    def __init__(self, nn=40):
        self.n = nn
        self.tt = np.zeros(self.n)
        self.x = np.zeros(self.n)
        self.ii = 0

    def __repr__(self):
        ss = 'fps_t: buffer size %d, framerate %.3f [fps]' % (
            self.n, self.framerate())
        return ss

    def addtime(self, t, x=0):
        """ Add a timestamp to the object """
        self.ii = self.ii + 1
        iim = self.ii % self.n
        # iimn=(self.ii+1)%self.n
        self.tt[iim] = t
        self.x[iim] = x

    def value(self):
        """ Return mean current values """
        return self.x.mean()

    def iim(self):
        return self.ii % self.n

    def framerate(self):
        """ Return the current framerate """
        iim = self.ii % self.n
        iimn = (self.ii + 1) % self.n
        dt = self.tt[iim] - self.tt[iimn]
        if dt == 0:
            return np.NaN
        fps = float(self.n - 1) / dt
        return fps

    def loop(self, s=''):
        self.addtime(time.time())
        self.showloop(s='')

    def showloop(self, dt=2, s=''):
        """ Print current framerate """
        fps = self.framerate()
        if len(s) == 0:
            tprint('loop %d: framerate: %.1f [fps]' % (self.ii, fps), dt=dt)
        else:
            tprint(
                '%s: loop %d: framerate: %.1f [fps]' % (s, self.ii, fps), dt=dt)

    def show(self):
        ''' Print the current framerate '''
        fps = self.framerate()
        print('framerate: %.3f' % fps)


def mkdirc(d):
    """ Similar to mkdir, but no warnings if the directory already exists """
    try:
        os.mkdir(d)
    except:
        pass
    return d


def projectiveTransformation(H, x):
    """ Apply a projective transformation to a kxN array

    >>> y = projectiveTransformation( np.eye(3), np.random.rand( 2, 10 ))
    """
    k = x.shape[0]
    kout = H.shape[0] - 1
    xx = x.transpose().reshape((-1, 1, k))

    if (xx.dtype is np.integer or xx.dtype == 'int64'):
        xx = xx.astype(np.float32)
    if xx.size > 0:
        ww = cv2.perspectiveTransform(xx, H)
        # ww=cv2.transform(xx, H)
        ww = ww.reshape((-1, kout)).transpose()
        return ww
    else:
        # fixme
        return x


def rottra2mat(rot, tra):
    """ create 4x4 matrix from 3x3 rot and 1x3 tra """
    out = np.eye(4)
    out[0:3, 0:3] = rot
    out[0:3, 3] = tra.transpose()
    return out


def breakLoop(wk=None, dt=0.001, verbose=0):
    """ Break a loop using OpenCV image feedback """
    if wk is None:
        wk = cv2.waitKey(1)
    time.sleep(dt)

    wkm = wk % 256
    if wkm == 27 or wkm == ord('q') or wk == 1048689:
        if verbose:
            print('breakLoop: key q pressed, quitting loop')
        return True
    return False

#%% Camera functions


def camera2opengl(mtx, far=100, near=1):
    """ Convert intrinsic camera calibration to opengl projection matrix

    See http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix/
    http://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix

    """

    alpha = mtx[0, 0]
    beta = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    f = far
    n = near
    s = 1
    M = np.array([[alpha / cx, 0, 0, 0], [0, beta / cy, 0, 0],
                 [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)], [0, 0, -1, 0]])
    return M

#%%

import scipy.io
import numpy


def hom(x):
    """ Make points homogeneous """
    nx = x.shape[1]
    return np.vstack((x, np.ones(nx)))


def dehom(x):
    """ Convert homogeneous points to local coordinates """
    return x[0:-1, :] / x[-1, :]


def null(a, rtol=1e-5):
    """ Calculate null space of a matrix """
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol * s[0]).sum()
    return rank, v[rank:].T.copy()


class camera_t():

    """ Class representating a camera """

    def __init__(self, T=np.matrix(np.eye(4)), mtx=None, distcoeff=None, imsize=None, matlabcalibfile=None, im=None, fc=None):
        """ Create a camera structure

        .T : world-to-camera matrix

        Arguments
        ---------
            distcoeff : array
                distortion coefficients: [k1,k2,p1,p2, k3] (as in OpenCV)
            mtx : array
                matrix with internal calibration
            imsize : array, optional
                size of image

        """
        self.T = T  # world-to-camera matrix
        self.mtx = None
        self.distcoeff = np.array(distcoeff)
        self.imsize = imsize

        if not matlabcalibfile is None:
            self.loadMatlabCalibration(matlabcalibfile)
            return

        if im is not None:
            self.idealCamera(im, fc)
            self.im = im
            return

        if T is None:
            raise Exception('T matrix should be 4x4 numpy matrix')
        self.T = np.array(T)
        if mtx is None:
            self.mtx = np.array(np.eye(3))
        else:
            self.mtx = np.array(mtx)
        if distcoeff is None:
            self.distcoeff = np.zeros(5,)
        else:
            self.distcoeff = distcoeff
        if imsize is None:
            self.imsize = np.array([0, 0])
        else:
            self.imsize = imsize

    def resizeCamera(self, factor):
        """ Resize a camera to desired size """
        newimsize = (factor * self.imsize).astype(int)
        newmtx = self.mtx.copy()

        newmtx[0, 0] *= factor  # rescale mtx
        newmtx[1, 1] *= factor

        cc = self.mtx[0:2, -1]
        ccnew = (cc - 0.5) * factor
        newmtx[0:2, -1] = ccnew

        newcam = camera_t(self.T, newmtx, self.distcoeff, newimsize)
        return newcam

    def back_project(self, xim, plane=None, verbose=1):
        """ Project image points to plane in world

        TODO: add distortion to camera backprojection

        Arguments:
            xim (Nx2 array): image points
            plane (4x1 array or None): plane in 3D
            verbose (int): output level


        >>> ecam = camera_t()
        >>> plane = np.array([[0,0,1,1]]).T
        >>> X = ecam.back_project( np.array( [[10,0]] ), plane )
        """
        ecam = self

        if plane is None:
            plane = np.array([0, 0, 1, 0])

        N = xim.shape[0]
        Xproj = np.zeros((N, 3))

        cc = ecam.cameraCentre()

        # TODO: vectorize code
        for i in range(N):
            pt = xim[i, :]
            # TODO: here we assume no distortion
            v = np.linalg.inv(ecam.mtx) .dot(hom(pt.reshape(2, 1)))
            vv = projectiveTransformation(np.matrix(np.linalg.inv(ecam.T)), v)
            # linedir=vv.flatten()-cc

            # linedir=-linedir/np.linalg.norm(linedir)
            # pt2=cc.reshape(3,1)+10*linedir.reshape(3,1)
            Ldual = np.hstack((hom(cc.reshape(3, 1)), hom(vv)))
            r, L = null(Ldual.T)

            AA = np.hstack((L, plane.reshape(4, 1)))
            _, pt = null(AA.T)

            Xproj[i, :] = dehom(pt).flatten()

        return Xproj  # , ( pt, pt2) # np.array( [0,0,0]).reshape( (1,3))

    def idealCamera(self, im, fc=None):
        """ Set the camera to an ideal camera for the specified image """
        self.imsize = np.array([im.shape[1], im.shape[0]])
        cc1 = float(self.imsize[0]) / 2 - .5
        cc2 = float(self.imsize[1]) / 2 - .5
        if fc is None:
            fc = self.imsize[0]
        self.mtx = np.matrix([[fc, 0, cc1], [0, fc, cc2], [0, 0, 1]])
        self.T = np.matrix(np.eye(4))
        self.distcoeff = np.zeros(5,)

    def undistortImage(self, im):
        if im is None:
            return None
        if self.mtx is None:
            imu = im.copy()
            return imu
        else:
            # w,h=self.imsize[0], self.imsize[1]
            # mapx,mapy = cv2.initUndistortRectifyMap(self.mtx,self.distcoeff,None,self.mtx,(w,h),5)
            # imu = cv2.remap(im,mapx,mapy,cv2.INTER_LINEAR)
            imu = cv2.undistort(im, self.mtx, self.distcoeff, None, self.mtx)
            return imu

    def writeCalibration(self, filename, verbose=1):
        with open(filename, 'wt') as fid:
            for m in self.mtx:
                mm = ' '.join(
                    ['%f' % x for x in (np.array(m).flatten()).tolist()])
                fid.write('%s\n' % mm)

            fid.write('%s\n' % ' '.join(['%f' % float(x)
                      for x in np.array(self.distcoeff).flatten()]))
            fid.write('%d %d\n' % (self.imsize[0], self.imsize[1]))
            fid.write('-1 -1\n')

    def loadMatlabCalibration(self, matlabcalibfile, verbose=1):
        """ Load Matlab calibration generated by the Camera Calibration toolbox (Bougueut) """
        m = scipy.io.loadmat(matlabcalibfile)

        if ('cam' in m) and not ('fc' in m):
            try:
                # print('here')
                m = m['cam']
                fc = m['fc'][0, 0]
                cc = m['cc'][0, 0]
                distc = np.array(m['kc'][0, 0])
                imsize = np.array([m['nx'][0, 0], m['ny'][0, 0]]).flatten()
            except:
                raise Exception('error loading calibration file')
                pass
        else:
            # normal calibration file
            fc = m['fc']
            cc = m['cc']
            distc = np.array(m['kc'])
            imsize = np.array([m['nx'], m['ny']]).flatten()
        mtx = np.matrix(np.eye(3))
        mtx[0, 0] = fc[0]
        mtx[1, 1] = fc[1]
        mtx[0:2, -1] = cc

        self.T = np.matrix(np.eye(4))
        self.mtx = np.matrix(mtx)
        self.distcoeff = distc
        self.imsize = imsize
        if verbose:
            print('camera_t: loaded Matlab calibration fc %.3f' %
                  self.focallength())

    def cameraCentre(self):
        """ Returns the centre of the camera in world coordinates """
        return np.array(np.linalg.inv(self.T)[0:3, 3])

    def Twc(self):
        """ Return world-to-camera matrix """
        return self.T

    def heading(self):
        """ Return heading of the camera """
        vd = self.viewDirection()
        heading = np.arctan2(vd[0], vd[1])[0, 0]
        return heading

    def fieldOfView(self, imsize, verbose=1):
        """ Return field of view for a camera """
        imsize = np.array(imsize)
        fov = 2 * np.arctan(((imsize / 2.) / self.focallength()))
        if verbose:
            print('camera: horizontal field of view = %.2f degrees\n' %
                  np.rad2deg(fov[0]))

        return fov

    def rpy(self):
        return RBE2euler(self.T[0:3, 0:3])

    def drawHorizon(self, img, height=0, color=(255, 0, 0), thickness=3):
        """ Draw camera horizon on image """
        mtx = self.mtx
        dist = self.distcoeff
        rvecs, tvecs = self.rvectvec()

        vd = self.viewDirection()
        up = self.upDirection()
        side = np.cross(vd.flat, up.flat)
        cc = self.cameraCentre()
        xx = 10000
        gside = np.cross(vd.flat, [0, 0, 1])

        vv2 = np.linspace(-xx, xx, 100) * gside.reshape((3, 1))
        vv1 = cc + vd * xx
        Xhor = vv1 + vv2
        Xhor[2, :] += height
        xhor = self.projectPoints(Xhor)
        # hidx=pointsinrect(xhor, [[1;1] [1600;1200]]);
        # xhor=xhor(:, hidx);

        imgpts = xhor.astype(int)
        img = img.copy()
        corner = tuple(imgpts[0].ravel())
        for ii, x in enumerate(imgpts):
            xx = tuple(x.ravel())
            r = cv2.line(img, corner, xx, color=color, thickness=thickness)
            corner = xx
        return img

    def drawAxis(self, img, scale=0.23):
        """ Draw coordinate axis on image

        Arguments
        ---------
            img : array
                input image
            scale: float
                scale of axis to plot
        """
        mtx = self.mtx
        dist = self.distcoeff
        rvecs, tvecs = self.rvectvec()
        cc = scale * \
            np.array([[0, 0, 0.], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(
                np.float32)
        imgpts = cv2.projectPoints(cc, rvecs, tvecs, mtx, dist)[0]
        img = img.copy()
        corner = tuple(imgpts[0].ravel())
        r = cv2.line(img, corner, tuple(imgpts[1].ravel()), (255, 0, 0), 5)
        r = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 255, 0), 5)
        r = cv2.line(img, corner, tuple(imgpts[3].ravel()), (0, 0, 255), 5)
        return img

    def upDirection(self):
        """ Return the current up direction
        """
        vd = np.linalg.inv(self.T[0:3, 0:3]).dot(np.array([[0], [-1], [0]]))
        return vd / np.linalg.norm(vd)

    def viewDirection(self):
        """ Return the current viewing direction """
        vd = np.linalg.inv(self.T[0:3, 0:3]).dot(numpy.array([[0], [0], [1]]))
        return vd / np.linalg.norm(vd)

    def viewYDirection(self):
        """ Return the current y-image direction """
        vd = np.linalg.inv(self.T[0:3, 0:3]).dot(np.array([[0], [1], [0]]))
        return vd / np.linalg.norm(vd)

    def viewXDirection(self):
        """ Return the current x-image direction """
        vd = np.linalg.inv(self.T[0:3, 0:3]).dot(np.array([[1], [0], [0]]))
        return vd / np.linalg.norm(vd)

    def focallength(self):
        """ return focal length of the camera """
        return (self.mtx[0, 0] + self.mtx[1, 1]) / 2

    def rvectvec(self):
        """ Return OpenCV pose """
        rvec, tvec = T2opencv(self.T)
        return rvec, tvec

    def __repr__(self):
        try:
            cc = self.cameraCentre()
            vd = self.viewDirection()
            return 'camera_t object: center %s, viewdir %s' % (cc.ravel(), vd.ravel())
        except:
            return 'camera_t object: '

    def projectPoints(ecam, X):
        """ Project a set of points to image coordinates
            The 3D points X should be an Nx3 array.

        """
        rvec, tvec = T2opencv(ecam.T)
        if X.shape[0] == 3 and X.shape[1] != 3:
            # legacy mode
            if X.size > 0:
                xim = cv2.projectPoints(
                    X.transpose(), rvec, tvec, ecam.mtx, ecam.distcoeff)[0]
            else:
                xim = np.zeros((0, 1, 2))
        else:
            xim = cv2.projectPoints(X, rvec, tvec, ecam.mtx, ecam.distcoeff)[0]

        return xim

    def reprojectionError(ecam, X, xim):
        """ Calculate reprojection error for a set of 3D and 2D points

        The error calculated is:

        .. math::
            E = \sqrt{ \sum_j ||PX_j-y_j ||^2 }

        """

        xxl = ecam.projectPoints(X).squeeze().transpose()
        dd = xxl - xim
        err = np.linalg.norm(dd, axis=0)
        cost = np.mean(err)
        return cost, err

    def upCamera(self):
        """ Return camera with same position and viewing direction but with camera y-axis aligned to world z-axis """
        cc = self.cameraCentre()
        vd = self.viewDirection()
        side = np.cross(vd.T, np.array([0, 0, 1])).T
        if np.linalg.norm(side) < 1e-6:
            # side=np.array([[1,0,0]]).T
            side = np.cross(vd.T, np.array([0, -1, 0])).T
        side = side / np.linalg.norm(side)
        side = side / np.linalg.norm(side)
        down = np.cross(vd.T, side.T).T
        down = down / np.linalg.norm(down)
        Tnew = pg_rotation2H(np.hstack((side, down, vd)))
        Tnew[0:3, 3] = cc

        upcam = camera_t(np.linalg.inv(Tnew), self.mtx, self.distcoeff)
        return upcam

    def frontFilterFast(ecam, X, verbose=0, rect=None, neardist=0.01):
        """ Select points which are in front of the camera (and in a certain region)
            Points in a Nx3 np.array and rect is a 2x5 np.array.float32
        """
        mtx = ecam.mtx
        distcoeff = ecam.distcoeff
        rvec, tvec = ecam.rvectvec()
        Twc = ecam.T
        # Tcw = np.linalg.inv(Twc)
        # apply to all points
        xx = projectiveTransformation(Twc, X.T)
        # valid and invalid indices
        ind = (xx[2, :] >= neardist).nonzero()[0]

        # notind = (bools==False).nonzero()[0]
        if ind.size == 0:
            xim = np.zeros((0, 1, 2))
        else:
            xim = cv2.projectPoints(X[ind, :], rvec, tvec, mtx, distcoeff)[0]

        if verbose >= 2:
            plt.figure(100)
            plt.clf()
            plotPoints(rect, '-m')
            plotPoints(xim.T, '.b')
            if verbose >= 3:
                plotPoints(ximu.T, 'or')
            plt.axis('image')

        return xim, ind

    def frontFilter(ecam, X, verbose=0, rect=None, neardist=0.01):
        """ Select points which are in front of the camera (and in a certain region)
            Points in a 3xN np.array.float32 and rect with in a 2x5 np.array.float32
        """

        mtx = ecam.mtx
        distcoeff = ecam.distcoeff
        rvec, tvec = ecam.rvectvec()
        Twc = ecam.T
        # Tcw = Twc.I
        # apply to all points
        xx = projectiveTransformation(Twc, X)
        ximall = cv2.projectPoints(
            X.transpose(), rvec, tvec, mtx, distcoeff)[0]
        ximuall = cv2.projectPoints(X.transpose(), rvec, tvec, mtx, None)[0]

        # do the z buffer test
        bools = xx[2, :] > neardist
        # do the 2d projection test
        if not rect is None:
            for ii, pt in enumerate(ximuall):
                ptx = tuple(pt.flatten())
                v = cv2.pointPolygonTest(rect.T, ptx, False)
                bools[ii] = bools[ii] and v > 0

        # valid and invalid indices
        ind = (bools == True).nonzero()[0]
        notind = (bools == False).nonzero()[0]
        if ind.size == 0:
            ximu = np.zeros((0, 1, 2))
            xim = np.zeros((0, 1, 2))
            ximall = -np.ones((xx.shape[1], 1, 2))
        else:
            ximu = cv2.projectPoints(
                X[:, ind].transpose(), rvec, tvec, mtx, None)[0]
            xim = cv2.projectPoints(
                X[:, ind].transpose(), rvec, tvec, mtx, distcoeff)[0]
            ximall[notind, :, :] = -1

        if verbose >= 2:
            plt.figure(100)
            plt.clf()
            plotPoints(rect, '-m')
            plotPoints(xim.T, '.b')
            if verbose >= 3:
                plotPoints(ximu.T, 'or')
            plt.axis('image')

        return xim, ind, ximall

    def estimatePose(ecam, XX, xim):
        try:
            pnpmode = cv2.EPNP  # opencv3 dev
            # pnpmode = cv2.CV_ITERATIVE # opencv3 dev
            pnpmodeiter = cv2.CV_ITERATIVE  # opencv3 dev
        except:
            pnpmode = cv2.SOLVEPNP_EPNP  # opencv2?
            pnpmodeiter = cv2.SOLVEPNP_ITERATIVE  # opencv3 dev
            pass
#        pnpmode= cv2.SOLVEPNP_ITERATIVE

        mtx = ecam.mtx
        distcoeff = ecam.distcoeff
        # tvec=init_tvec, rvec=init_rvec, useExtrinsicGuess=1 )
        found, rvec, tvec = cv2.solvePnP(XX, xim.reshape(
            (-1, 1, 2)), mtx, distcoeff, None, None, False, pnpmode)
        Tgood = opencv2T(rvec, tvec)  # world-to-camera
        ecamx = camera_t(Tgood, mtx, distcoeff)
        return ecamx



def runcmd(cmd, verbose=0):
    """ Run command and return output """
    output = subprocess.check_output(cmd, shell=True)
    return output

#%% Geometry functions

#%% Conversion between different representations of angles and frames


def angleDiff(x, y):
    """ Return difference between two angles in radians modulo 2* pi

    >>> d=angleDiff( 0.01, np.pi+0.02)
    >>> d=angleDiff( 0.01, 2*np.pi+0.02)
    """
    return np.abs(((x - y + np.pi) % (2 * np.pi)) - np.pi)


def angleDiffOri(x, y):
    """ Return difference between two angles in radians modulo pi

    >>> d=angleDiff( 0.01, np.pi+0.02)
    >>> d=angleDiff( 0.01, 2*np.pi+0.02)
    """
    return np.abs(((x - y + np.pi / 2) % (np.pi)) - np.pi / 2)


def opencvpose2attpos(rvecs, tvecs):
    tvec = np.array(tvecs).flatten()
    rvec = np.array(rvecs).flatten()
    R, tmp = cv2.Rodrigues(rvec)
    att = RBE2euler(R)
    pos = -R.transpose().dot(np.matrix(tvec.reshape((3, 1))))
    return att, pos


def opencv2TX(rvecs, tvecs):
    """ Convert OpenCV pose to homogenous transform """
    T = np.matrix(np.eye(4))
    R = cv2.Rodrigues(rvecs)[0]
    T[0:3, 0:3] = R
    T[0:3, 3:4] = tvecs
    return T


def opencv2T(rvec, tvec):
    """ Convert OpenCV pose to homogenous transform """
    T = np.matrix(np.eye(4))
    T[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    T[0:3, 3] = tvec
    return T


def T2opencv(T):
    """ Convert transformation to OpenCV rvec, tvec pair

    Example
    -------
    >>> rvec, tvec = T2opencv(np.eye(4))

    """
    rvec = cv2.Rodrigues(T[0:3, 0:3])[0]
    tvec = T[0:3, 3]
    return rvec, tvec


def frame2T(f):
    """ Convert frame into 4x4 transformation matrix """
    T = np.matrix(np.eye(4))
    T[0:3, 0:3] = euler2RBE(f[3:7])
    T[0:3, 3] = f[0:3].reshape(3, 1)
    return T


def euler2RBE(theta):
    """ Convert Euler angles to rotation matrix

      Example
      -------
      >>> np.set_printoptions(precision=4, suppress=True)
      >>> euler2RBE( [0,0,np.pi/2] )
      matrix([[ 0., -1.,  0.],
              [ 1.,  0.,  0.],
              [-0.,  0.,  1.]])

    """
    cr = math.cos(theta[0])
    sr = math.sin(theta[0])
    cp = math.cos(theta[1])
    sp = math.sin(theta[1])
    cy = math.cos(theta[2])
    sy = math.sin(theta[2])

    out = np.matrix([cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy,
                     cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy, -sp, sr * cp, cr * cp])
    return out.reshape((3, 3))


def RBE2euler(Rbe):
    """ Convert rotation matrix to Euler angles """
    out = np.zeros([3, 1])
    out[0, 0] = math.atan2(Rbe[2, 1], Rbe[2, 2])
    out[1, 0] = -math.asin(Rbe[2, 0])
    out[2, 0] = math.atan2(Rbe[1, 0], Rbe[0, 0])
    return out

#%% Helper functions


def pg_rotation2H(R):
    """ Convert rotation matrix to homogenous transform matrix """
    X = np.matrix(np.eye((R.shape[0] + 1)))
    X[0:-1, 0:-1] = R
    return X


def directionMean(vec):
    """ Calculate the mean of a set of directions

    The initial direction is determined using the oriented direction. Then a non-linear optimization is done.

    >>> vv=np.array( [[1,0],[1,0.1], [-1,.1]])
    >>> a=directionMean(vv)

    """
    def dist(a, vec):
        phi = np.arctan2(vec[:, 0], vec[:, 1])
        x = a - phi
        x = np.mod(x + np.pi / 2, np.pi) - np.pi / 2
        cost = np.linalg.norm(x)
        return cost
    Nfeval = 1

    def callbackF(Xi):
        global Nfeval
        print(Xi)
        print('{0:4d}   {1: 3.6f}'.format(Nfeval, Xi[0], dist(Xi[0], vec)))
        Nfeval += 1

    m = vec.mean(axis=0)
    a0 = np.arctan2(m[0], m[1])
    ff = lambda a: dist(a, vec)

    r = scipy.optimize.minimize(
        ff, a0, callback=None, options=dict({'disp': False}))
    a = r.x
    return a


def circular_mean(weights, angles):
    """ Calculate circular mean of a set of 2D vectors """
    x = y = 0.
    for angle, weight in zip(angles, weights):
        x += math.cos(math.radians(angle)) * weight
        y += math.sin(math.radians(angle)) * weight

    mean = math.degrees(math.atan2(y, x))
    return mean


def dir2R(d, a=None):
    """ Convert direction to rotation matrix

    Note: numerically not stable near singular points!

    Arguments:
        d (numpy array of size 3): direction to rotation to a
        a (numpy array of size 3): target direction
    Returns:
        R (3x3 numpy array): matrix R such that R*a = d

    Example:

    >>> d = np.array([0, 1, 0]); a = np.array([0, -1, 0])
    >>> R = dir2R(d, a)

    Pieter Eendebak <pieter.eendebak@tno.nl>
    """

    # set target vector
    if a is None:
        a = np.array([0, 0, 1])

    # normalize
    b = d.reshape((3, 1)) / np.linalg.norm(d)
    a = a.reshape((3, 1))

    c = np.cross(a.flat, b.flat)

    if np.linalg.norm(c) < 1e-12 and a.T.dot(b) < .01:
        #  deal with singular case
        if(np.linalg.norm(a[1:]) < 1e-4):
            R0 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        else:
            R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        a = R0.dot(a)
        bt = (a + b) / np.linalg.norm(a + b)
        R = np.eye(3) - 2 * a.dot(a.T) - 2 * \
                   (bt.dot(bt.T)).dot(np.eye(3) - 2 * a.dot(a.T))
        R = R.dot(R0)
    else:
        bt = (a + b) / np.linalg.norm(a + b)

        R = np.eye(3) - 2 * a.dot(a.T) - 2 * \
                   (bt.dot(bt.T)).dot(np.eye(3) - 2 * a.dot(a.T))

    return R


@static_var("b", np.matrix(np.zeros((2, 2))))
def rot2D(phi):
    """ Return 2x2 rotation matrix from angle

    Arguments
    ---------
    phi : float
        Angle in radians
    Returns
    -------
        R : array
            The 2x2 rotation matrix

    Examples
    --------
    >>> R = rot2D(np.pi)

    """
    r = rot2D.b.copy()
    c = math.cos(phi)
    s = math.sin(phi)
    r.itemset(0, c)
    r.itemset(1, -s)
    r.itemset(2, s)
    r.itemset(3, c)
    return r


def pg_rotx(phi):
    """ Rotate around the x-axis with angle """
    c = math.cos(phi)
    s = math.sin(phi)
    R = np.eye(3)
    R[1, 1] = c
    R[2, 1] = s
    R[1, 2] = -s
    R[2, 2] = c
    return np.matrix(R)

if platform.node() == 'woelmuis':
    # transition code to convert all elements to np.array type (and use python 3.5 @ operator)
    # np.array is faster, and with the @ operator we get cleaner code

    def frame2T(f):
        """ Convert frame into 4x4 transformation matrix """
        T = np.array(np.eye(4))
        T[0:3, 0:3] = euler2RBE(f[3:7])
        T[0:3, 3] = f[0:3].reshape(3, 1)
        return T

    @static_var("b", np.array(np.zeros((2, 2))))
    def rot2D(phi):
        """ Return 2x2 rotation matrix from angle

        Arguments
        ---------
        phi : float
            Angle in radians
        Returns
        -------
            R : array
                The 2x2 rotation matrix

        Examples
        --------
        >>> R = rot2D(np.pi)

        """
        r = rot2D.b.copy()
        c = math.cos(phi)
        s = math.sin(phi)
        r.itemset(0, c)
        r.itemset(1, -s)
        r.itemset(2, s)
        r.itemset(3, c)
        return r

    def pg_rotx(phi):
        """ Rotate around the x-axis with angle """
        c = math.cos(phi)
        s = math.sin(phi)
        R = np.zeros((3, 3))
        R.flat = [1, 0, 0, 0, c, -s, 0, s, c]
        return R


def imshowz(im, *args, **kwargs):
    """ Show image with interactive z-values """
    plt.imshow(im, *args, **kwargs)

    sz = im.shape
    numrows, numcols = sz[0], sz[1]

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = im[row, col]
            try:
                if len(z) == 1:
                    return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
                else:
                    return 'x=%1.4f, y=%1.4f, z=%s' % (x, y, str(z))
            except:
                return 'x=%1.4f, y=%1.4f, z=%s' % (x, y, str(z))
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax = plt.gca()
    ax.format_coord = format_coord


def pg_scaling(scale, cc=None):
    """ Create scaling with specified centre


    Example
    -------
    >>> pg_scaling( [1.,2])
    array([[ 1.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  1.]])

    """
    scale = np.array(scale)
    scale = np.hstack((scale, 1))
    H = np.diag(scale)
    if cc is not None:
        cc = np.array(cc).flatten()
        H = pg_transl2H(cc) * H * pg_transl2H(-cc)

    return H


def pg_transl2H(tr):
    """ Convert translation to homogeneous transform matrix

    >>> pg_transl2H( [1,2])
    matrix([[ 1.,  0.,  1.],
            [ 0.,  1.,  2.],
            [ 0.,  0.,  1.]])

    """
    sh = np.array(tr)
    H = np.eye(sh.size + 1)
    H[0:-1, -1] = sh.flatten()
    H = np.matrix(H)
    return H


def setregion(im, subim, pos, mask=None, clip=False):
    """ Set region in Numpy image

    Arguments
    ---------
        im : Numpy array
            image to fill region in
        subim : Numpy array
            subimage
        pos: array
            position to place image
        mask (None or array): mask to use for the subimage
        clip (bool): if True clip the subimage where necessary to fit
    """
    h = subim.shape[0]
    w = subim.shape[1]
    x1 = int(pos[0])
    y1 = int(pos[1])
    x2 = int(pos[0]) + w
    y2 = int(pos[1]) + h
    if clip:
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, im.shape[1])
        y2 = min(y2, im.shape[0])
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
    # print('x1 %d, x2 %d, w %d' % (x1, x2, w))
    if mask is None:
        if len(im.shape) == len(subim.shape):
            im[y1:y2, x1:x2, ...] = subim[0:h, 0:w]
        else:
            im[y1:y2, x1:x2, ...] = subim[0:h, 0:w, np.newaxis]
    else:

        if len(im.shape) > len(mask.shape):
            im[y1:y2, x1:x2] = im[y1:y2, x1:x2] * \
                (1 - mask[:, :, np.newaxis]) + (subim * mask[:, :, np.newaxis])
        else:
            if len(im.shape) == len(subim.shape):
                im[y1:y2, x1:x2, ...] = im[y1:y2, x1:x2, ...] * \
                    (1 - mask[:, :]) + (subim * mask[:, :])
            else:
                im[y1:y2, x1:x2, ...] = im[y1:y2, x1:x2, ...] * \
                    (1 - mask[:, :]) + (subim[:, :, np.newaxis] * mask[:, :])

    return im


def frontFilter(X, rvec, tvec, mtx, verbose=0, distcoeff=None, rect=None, neardist=0.01, returnall=False):
    """ Select points which are in front of the camera (and in a certain region)
        Points in a 3xN np.array.float32 and rect with in a 2x5 np.array.float32
    """
    T = opencv2T(rvec, tvec)

    # apply to all points
    xx = projectiveTransformation(T, X)
    ximuall = cv2.projectPoints(X.transpose(), rvec, tvec, mtx, None)[0]

    # do the z buffer test
    bools = xx[2, :] > neardist
    # do the 2d projection test
    if not rect is None:
        for ii, pt in enumerate(ximuall):
            ptx = tuple(pt.flatten())
            v = cv2.pointPolygonTest(rect.T, ptx, False)
            bools[ii] = bools[ii] and v > 0

    # valid and invalid indices
    ind = (bools == True).nonzero()[0]
    notind = (bools == False).nonzero()[0]
    if ind.size == 0:
        ximu = np.zeros((0, 1, 2))
        xim = np.zeros((0, 1, 2))
    else:
        ximu = cv2.projectPoints(
            X[:, ind].transpose(), rvec, tvec, mtx, None)[0]
        xim = cv2.projectPoints(
            X[:, ind].transpose(), rvec, tvec, mtx, distcoeff)[0]

    if verbose >= 2:
        plt.figure(100)
        plt.clf()
        plt.imshow(rgb)
        plotPoints(xim.T, '.b')
        plotPoints(ximu.T, 'or')

    if returnall:
        if ind.size == 0:
            ximall = -np.ones((xx.shape[1], 1, 2))
        else:
            ximall = cv2.projectPoints(
                X.transpose(), rvec, tvec, mtx, distcoeff)[0]
            ximall[notind, :, :] = -1
        return xim, ind, ximall
    else:
        return xim, ind

# import matplotlib
# matplotlib.use('TkAgg')


def region2poly(rr):
    """ Convert a region (bounding box xxyy) to polygon """
    if type(rr) == tuple or type(rr) == list:
        # x,y,x2,y2 format
        rr = np.array(rr).reshape((2, 2)).transpose()
        poly = np.array([rr[:, 0:1], np.array([[rr[0, 1]], [rr[1, 0]]]), rr[
                        :, 1:2], np.array([[rr[0, 0]], [rr[1, 1]]]), rr[:, 0:1]]).reshape((5, 2)).T
        return poly
        # todo: eliminate transpose
    # poly=np.array( (2, 5), dtype=rr.dtype)
    # poly.flat =rr.flat[ [0,1,1,0, 0, 2,2,3,3,2] ]
    poly = rr.flat[[0, 1, 1, 0, 0, 2, 2, 3, 3, 2]].reshape((2, 5))

    # poly = np.array([rr[:, 0:1], np.array([[rr[0, 1]], [rr[1, 0]]]), rr[ :,
    # 1:2], np.array([[rr[0, 0]], [rr[1, 1]]]), rr[:, 0:1]]).reshape((5, 2)).T
    return poly


def plotLabels(xx, *args, **kwargs):
    """ Plot labels next to points

    Example:
    >>> xx=np.random.rand(2, 10)
    >>> fig=plt.figure(10); plt.clf()
    >>> _ = plotPoints(xx, '.b'); _ = plotLabels(xx)
    """

    if len(np.array(xx).shape) == 1 and xx.shape[0] == 2:
        xx = xx.reshape((2, 1))
    if xx.shape[0] > 2 and xx.shape[1] == 2:
        xx = xx.T
    if len(args) == 0:
        v = range(0, xx.shape[1])
        lbl = ['%d' % i for i in v]
    else:
        lbl = args[0]
        if type(lbl) == int:
            lbl = [str(lbl)]
        elif type(lbl) == str:
            lbl = [str(lbl)]
    # plt.text(xx[0:], xx[1,:], lbl, **kwargs)
    nn = xx.shape[1]
    ax = plt.gca()
    # print(nn)
    # print(lbl)
    th = [None] * nn
    for ii in range(nn):
        # print('-- %d' % ii)
        ww = str(lbl[ii])
        # print(ww)
        th[ii] = ax.annotate(str(lbl[ii]), xx[:, ii], **kwargs)
    return th


def plot2Dline(line, *args, **kwargs):
    """ Plot a 2D line in a matplotlib figure

    line: 3x1 array

    >>> plot2Dline([-1,1,0], 'b')
    """
    if np.abs(line[1]) > .001:
        xx = plt.xlim()
        xx = np.array(xx)
        yy = (-line[2] - line[0] * xx) / line[1]
        plt.plot(xx, yy, *args, **kwargs)
    else:
        yy = np.array(plt.ylim())
        xx = (-line[2] - line[1] * yy) / line[0]
        plt.plot(xx, yy, *args, **kwargs)


# plotLabels( np.zeros( (2,3)))

#%%

def scaleImage(image, display_min=None, display_max=None):
    """ Scale any image into uint8 range

    Args:
        image (numpy array): input image
        display_min (float): value to map to min output range
        display_max (float): value to map to max output range
    Returns:
        image (numpy array): the scaled image

    Example:
        >>> im=scaleImage(255*np.random.rand( 30,40), 40, 100)
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


def auto_canny(image, sigma=0.33):
    """ Canny edge detection with automatic parameter detection

    Code from http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

    >>> imc=auto_canny(np.zeros( (200,300)).astype(np.uint8))

    Arguments
    ---------
        image : array
            input image
    Returns
    -------
        edged : array
            detected edges

    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

#%% Plotting functions


def plotPoints(xx, *args, **kwargs):
    """ Plot 2D or 3D points


    Arguments:
        xx - array of points
        \*args - arguments passed to the plot function of matplotlib

    Example:
    >>> plotPoints(np.random.rand(2,10), '.-b')

    """
    if xx.shape[0] == 2:
        h = plt.plot(xx[0, :], xx[1, :], *args, **kwargs)
    elif xx.shape[0] == 3:
        h = plt.plot(xx[0, :], xx[1, :], xx[2, :], *args, **kwargs)
    if xx.shape[0] == 1:
        h = plt.plot(xx[0, :], *args, **kwargs)
    else:
        h = None
    return h


def orthogonal_proj(zfront, zback):
    """ see http://stackoverflow.com/questions/23840756/how-to-disable-perspective-in-mplot3d """
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    return numpy.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, a, b],
                        [0, 0, -1e-9, zback]])


def plotPoints3D(xx, *args, **kwargs):
    """ Plot 3D points

    Arguments
    ---------
    xx: 3xN array
        the 3D data points

    Example
    -------
    >> ax=plotPoints3D(np.random.rand(3, 1) ,'.r', markersize=10, fig=12)

    """

    fig = kwargs.get('fig', None)
    verbose = kwargs.get('verbose', 0)
    if 'fig' in kwargs.keys():
        kwargs.pop('fig')
    if 'verbose' in kwargs.keys():
        kwargs.pop('verbose')

    if verbose:
        print('plotPoints3D: using fig %s' % fig)
        print('plotPoints3D: using args %s' % args)
    # pdb.set_trace()
    if fig is None:
        ax = p.gca()
    else:
        fig = p.figure(fig)
    # ax = p3.Axes3D(fig)
        ax = fig.gca(projection='3d')
        # ax = fig.gca(projection='rectilinear')
        # ax = p3.Axes3D(fig)
    # ax=p.gca()
    # r=ax.plot3D(np.ravel(xx[0,:]),np.ravel(xx[1,:]),np.ravel(xx[2,:]),
    # *args, **kwargs)
    r = ax.plot(np.ravel(xx[0, :]), np.ravel(xx[1, :]),
                np.ravel(xx[2, :]), *args, **kwargs)
    # ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    #   fig.add_axes(ax)
    # p.show()
    p.draw()
    return ax

#%%


def polyarea(p):
    """ Return signed area of polygon

    Arguments
    ---------
        p : Nx2 numpy array or list of vertices
            vertices of polygon
    Returns
    -------
        area : float
            area of polygon

    >>> polyarea( [ [0,0], [1,0], [1,1], [0,2]] )
    1.5
    """
    if len(p) <= 1:
        return 0
    if isinstance(p, numpy.ndarray):
        val = 0
        for x in range(len(p)):
            x0 = p[x, 0]
            y0 = p[x, 1]
            xp = x + 1
            if xp >= len(p):
                xp = 0
            x1 = p[xp, 0]
            y1 = p[xp, 1]
            val += 0.5 * (x0 * y1 - x1 * y0)
        return val

    def polysegments(p):
        """ Helper functions """
        if isinstance(p, list):
            return zip(p, p[1:] + [p[0]])
        else:
            return zip(p, np.vstack((p[1:], p[0:1])))
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for ((x0, y0), (x1, y1)) in polysegments(p)))


try:
    import Polygon as polygon3

    def polyintersect(x1, x2):
        """ Intersection of two polygons

        >>> x1=np.array([(0, 0), (1, 1), (1, 0)] )
        >>> x2=np.array([(1, 0), (1.5, 1.5), (.5, 0.5)])
        >>> x=polyintersect(x1, x2)
        >>> _=plt.figure(10); plt.clf()
        >>> plotPoints(x1.T, '.-r' )
        >>> plotPoints(x2.T, '.-b' )
        >>> plotPoints(x.T, '.-g' , linewidth=2)

        """

        p1 = polygon3.Polygon(x1)
        p2 = polygon3.Polygon(x2)
        p = p1 & p2
        x = np.array(p)
        x = x.reshape((-1, 2))
        return x
except:
    try:
        import shapely
        import shapely.geometry
    except Exception as inst:
        pass

    def polyintersect(x1, x2):
        """ Intersection of two polygons

        >>> x1=np.array([(0, 0), (1, 1), (1, 0)] )
        >>> x2=np.array([(1, 0), (1.5, 1.5), (.5, 0.5)])
        >>> x=polyintersect(x1, x2)
        >>> _=plt.figure(10); plt.clf()
        >>> plotPoints(x1.T, '.-r' )
        >>> plotPoints(x2.T, '.-b' )

        """

        p1 = shapely.geometry.Polygon(x1)
        p2 = shapely.geometry.Polygon(x2)
        p = p1.intersection(p2)
        if p.is_empty:  # len(p)==0:
            return np.zeros((0, 2))
        x = np.array(list(p.exterior.coords))
        return x

#%%


def opencv_draw_points(bgr, imgpts, drawlabel=True, radius=3, color=(255, 0, 0), thickness=-1, copyimage=True):
    """ Draw points on image

    Arguments
    ---------
        bgr : numpy array
            image to draw points into
        impts : array
            locations of points to plot

    """
    if copyimage:
        out = bgr.copy()
    else:
        out = bgr
    fscale = .5 + .5 * (radius * 0.2)
    fthickness = int(fscale + 1)
    for i, pnt in enumerate(imgpts):  # enumerate(imgpts):
        tpnt = tuple(pnt.ravel())
        # print('radius: %f, th %f' % (radius, thickness))
        cv2.circle(out, tpnt, radius, color, thickness)
        if(drawlabel):
            cv2.putText(
                out, '%d' % (i + 1), tpnt, cv2.FONT_HERSHEY_SIMPLEX, fscale, color, fthickness)
    return out


def enlargelims(factor=1.05):
    """ Enlarge the limits of a plot
    Example:
      >>> enlargelims(1.1)

    """
    xl = plt.xlim()
    d = (factor - 1) * (xl[1] - xl[0]) / 2
    xl = (xl[0] - d, xl[1] + d)
    plt.xlim(xl)
    yl = plt.ylim()
    d = (factor - 1) * (yl[1] - yl[0]) / 2
    yl = (yl[0] - d, yl[1] + d)
    plt.ylim(yl)


def finddirectories(p, patt):
    """ Get a list of files """
    lst = os.listdir(p)
    rr = re.compile(patt)
    lst = [l for l in lst if re.match(rr, l)]
    lst = [l for l in lst if os.path.isdir(os.path.join(p, l))]
    return lst

def _walk_calc_progress(progress, root, dirs):
    """ Helper function """
    prog_start, prog_end, prog_slice = 0.0, 1.0, 1.0

    current_progress = 0.0
    parent_path, current_name = os.path.split(root)
    data = progress.get(parent_path)
    if data:
        prog_start, prog_end, subdirs = data
        i = subdirs.index(current_name)
        prog_slice = (prog_end - prog_start) / len(subdirs)
        current_progress = prog_slice * i + prog_start

        if i == (len(subdirs) - 1):
            del progress[parent_path]

    if dirs:
        progress[root] = (current_progress, current_progress+prog_slice, dirs)

    return current_progress

def findfilesR(p, patt, show_progress=False):
    """ Get a list of files (recursive)

    Args:
        
        p (string): directory
        patt (string): pattern to match
        show_progress (bool)
    Returns:
        lst (list of str)               
    """
    lst = []
    rr = re.compile(patt)
    progress = {}
    for root, dirs, files in os.walk(p, topdown=True):
         frac=_walk_calc_progress(progress, root, dirs)
         if show_progress:
           tprint('findfilesR: %s: %.1f%%' % (p, 100*frac))
         lst += [os.path.join(root, f) for f in files if re.match(rr, f)]
    return lst


def signedsqrt(val):
    """ Signed square root function

    >>> signedsqrt([-4.,4,0])
    array([-2.,  2.,  0.])
    >>> signedmin(-10, 5)
    -5
    """
    val = np.sign(val) * np.sqrt(np.abs(val))
    return val


def signedmin(val, w):
    """ Signed minimum value function

    >>> signedmin(-3, 5)
    -3
    >>> signedmin(-10, 5)
    -5
    """
    val = np.minimum(val, abs(w))
    val = np.maximum(val, -abs(w))
    return val


def smoothstep(x, x0=0, alpha=1):
    """ Smooth step function

    >>> t=np.arange(0,600,1.)
    >>> _ = plt.plot(t, smoothstep(t, 300, alpha=1./100),'.b')

    """
    x = alpha * (x - x0)
    f = ((x / np.sqrt(1 + x * x)) + 1) / 2
    return f


def logistic(x, x0=0, alpha=1):
    """ Simple logistic function

    >>> t=np.arange(0,600,1.)
    >>> _ = plt.plot(t, logistic(t, 300, alpha=1./100),'.b')

    """
    f = 1 / (1 + np.exp(-2 * alpha * (x - x0)))
    return f


def findfiles(p, patt, recursive=False):
    """ Get a list of files """
    if recursive:
        return findfilesR(p, patt)
    lst = os.listdir(p)
    rr = re.compile(patt)
    lst = [l for l in lst if re.match(rr, l)]
    return lst

#%%


def blur_measure(im, verbose=0):
     """ Calculate bluriness for an image

     Args:
         im (array): input image
     """

     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
     # compute the variance of laplacian
     fm = cv2.Laplacian(gray, cv2.CV_64F).var()
     if verbose:
         print('calculate_blur: %.3f' % fm)
     return fm
 
def gaborFilter(ksize, sigma, theta, Lambda=1, psi=0, gamma=1, cut=None):
    """ Create a Gabor filter of specified size


    Input
    -----
    ksize : integer
        kernel size in pixels
    sigma, theta, Lambda, psi: float
        parameters of Gabor function
    cut: boolean
        if True cut off the angular component after specified distance (in radians)

    Output
    ------

    g : array
        constructed kernel

    Example
    -------

    >>> g = gaborFilter(ksize=15, sigma=2,theta=2,Lambda=1, gamma=1)

    """
    h = ((ksize - 1) // 2)
    x, y = np.meshgrid(range(-h, h + 1), range(-h, h + 1))
    sigma_x = sigma
    # print('gamma %s' % gamma)
    sigma_y = float(sigma) / gamma
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    xt = 2 * np.pi / Lambda * x_theta
    if cut is not None:
        pass
        xt = np.minimum(xt, cut)
        xt = np.maximum(xt, -cut)

    gb = np.exp(-.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(xt + psi)
    return gb

#%%

import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology


def detect_local_minima(arr, thr=None):
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    Args:
        arr (array): input array

    """
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710

    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr == 0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    if thr is not None:
        detected_minima[arr > thr] = 0
    return np.where(detected_minima)


#%% Matlab compatibility functions

__t00 = 0


def tic():
    """ Start timer """
    global __t00
    __t00 = time.time()
    return __t00


def toc(t0=None):
    """ Stop timer

    Returns:
        time elapsed (in seconds) since the start of the timer

    See also: :func:`tic`
    """
    if t0:
        dt = time.time() - t0
    else:
        dt = time.time() - __t00
    return dt


def fullpath(*args):
    """ Return full path from a list """
    p = os.path.join(*args)
    return p


def ginput(n=1, drawmode='', **kwargs):
    """ Select points from figure

    Press middle mouse button to stop selection

    Arguments:
        n - number of points to select
        drawmode - style to plot selected points
        kwargs : arguments passed to plot function
    """
    xx = np.zeros((2, 0))
    for ii in range(0, n):
        x = pylab.ginput(1)
        if len(x) == 0:
            # x = x[:, 0:ii]
            break
        x = np.array(x).T
        xx = np.hstack((xx, x))
        if drawmode is not None:
            plt.plot(xx[0, :].T, xx[1, :].T, drawmode, **kwargs)
            plt.draw()
    return xx


def save(pkl_file, *args):
    """ Save objects to file

    Arguments
    ---------
    pkl_file : string
         filename
    \*args : anything
         Python objects to save
    """

    # save data to disk
    # pdb.set_trace()
    output = open(pkl_file, 'wb')
    pickle.dump(args, output, protocol=2)
    output.close()


def load(pkl_file):
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


def cd(dd=''):
    """ Change current working directory """
    w = os.getcwd()
    if len(dd) < 1:
        return w
    os.chdir(dd)
    return


def choose(n, k):
    """ Binomial coefficients
    Return the n!/((n-k)!k!)

    Arguments:
        n -- Integer
        k -- Integer

    Returns:
        The bionomial coefficient n choose k

    Example:
      >>> choose(6,2)
      15

    """
    ntok = 1
    for t in range(min(k, n - k)):
        ntok = ntok * (n - t) // (t + 1)
    return ntok


def closefn():
    """ Destructor function for the module """
    return
    # global _applocalqt
import atexit
atexit.register(closefn)

# print('hi there')

#%%
import warnings


def deprecation(message):
    """ Issue a deprecation warning message """
    warnings.warn(message, DeprecationWarning, stacklevel=2)


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1, ploton=False):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in xrange(niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa)**2.) / step[0]
            gE = np.exp(-(deltaE / kappa)**2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa)**2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa)**2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return imgout


#%%

try:
    import PIL
    from PIL import ImageFont
    from PIL import Image
    from PIL import ImageDraw

    def writeTxt(im, txt, pos=(10, 10), fontsize=25, color=(0, 0, 0), fonttype='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'):
        """ Write text on image using PIL """
        font = ImageFont.truetype(fonttype, fontsize)
        im1 = Image.fromarray(im)
        # Drawing the text on the picture
        draw = ImageDraw.Draw(im1)
        draw.text(pos, txt, color, font=font)
        return np.array(im1)
except:
    def writeTxt(*args, **kwargs):
        """ Dummy function """
        warnings.warn('writeTxt: could not find PIL')
        return None
    pass


#%% Copy mplimage to clipboard

try:
    _usegtk = 0
    try:
        import matplotlib.pyplot
        _usegtk = 0
    except:
        import pygtk
        pygtk.require('2.0')
        import gtk
        _usegtk = 1
        pass

    import cv2
    # cb=QtGui.QClipboard

    #%
    def mpl2clipboard(event=None, verbose=1, fig=None):
        """ Copy current Matplotlib figure to clipboard """
        if verbose:
            print('copy current Matplotlib figure to clipboard')
        if fig is None:
            fig = matplotlib.pyplot.gcf()
        else:
            print('mpl2clipboard: figure %s' % fig)
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)
        im = np.roll(buf, 3, axis=2)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        if _usegtk:
            r, tmpfile = tempfile.mkstemp(suffix='.png')
            cv2.imwrite(tmpfile, im)
            image = gtk.gdk.pixbuf_new_from_file(tmpfile)
            clipboard = gtk.clipboard_get()
            clipboard.set_image(image)
            clipboard.store()
        else:
            cb = QtWidgets.QApplication.clipboard()
            r, tmpfile = tempfile.mkstemp(suffix='.bmp')
            cv2.imwrite(tmpfile, im)
            qim = QtGui.QPixmap(tmpfile)
            cb.setPixmap(qim)

            if 0:
                im = im[:, :, 0:3].copy()
                qim = QtWidgets.QImage(
                    im.data, im.shape[0], im.shape[1], QtWidgets.QImage.Format_RGB888)
                cb.setImage(qim)


except:
    def mpl2clipboard(event=None, verbose=1, fig=None):
        """ Copy current Matplotlib figure to clipboard

        Dummy implementation
        """
        if verbose:
            print('copy current Matplotlib figure to clipboard not available')

    pass


def addfigurecopy(fig=None):
    """ Add callback to figure window

    By pressing the 'c' key figure is copied to the clipboard

    """
    if fig is None:
        Fig = plt.gcf()
    else:
        Fig = plt.figure(fig)
    ff = lambda xx, figx=Fig: mpl2clipboard(fig=figx)
    Fig.canvas.mpl_connect('key_press_event', ff)  # mpl2clipboard)

#%%


class plotCallback:

    def __init__(self, func=None, xdata=None, ydata=None, scale=[1, 1], verbose=0):
        """ Object to facilitate matplotlib figure callbacks

        Args:
            func (function): function to be called
            xdata, ydata (arrays): datapoints to respond to
            verbose (int): output level
        Returns:
            pc (object): plot callback

        Example:
            >>> xdata=np.arange(4); ydata = np.random.rand( xdata.size)/2 + xdata
            >>> f = lambda plotidx, *args, **kwargs: print('point %d clicked' % plotidx)
            >>> pc = plotCallback(func=f, xdata=xdata, ydata=ydata)
            >>> fig = plt.figure(1); plt.clf(); _ = plt.plot(xdata, ydata, '.-b')
            >>> cid = fig.canvas.mpl_connect('button_press_event', pc)

        """

        self.func = func
        self.xdata = xdata
        self.ydata = ydata
        self.verbose = verbose
        if scale is None:
            # automatically determine scale
            scale=[1/(1e-8+np.ptp(xdata)), 1/(1e-8+ np.ptp(ydata) ) ] 
        self.scale = scale

    def __call__(self, event):
        if self.verbose:
            print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  (event.button, event.x, event.y, event.xdata, event.ydata))
            print('callback function: %s' % self.func)

        # pick data point
        idx = None

        try:
            if self.xdata is not None:
                xdata = np.array(self.xdata)
                ydata = np.array(self.ydata)
                pt = np.array(event.xdata, event.ydata)
                pt = np.array([event.xdata, event.ydata])
                xx = np.vstack((xdata.flat, ydata.flat)).T
                dd = xx - pt
                dd = np.multiply(np.array(self.scale).reshape((1, 2)), dd)
                d = np.linalg.norm(dd, axis=1)
                idx = np.argmin(d)
                distance = d[idx]
                if self.verbose:
                    print('point %d: distance %.3f' % (idx, distance))
            else:
                if self.verbose:
                    print('no xdata')

            # call the function
            self.func(plotidx=idx, button=event.button)
        except Exception as e:
            print(e)
        if self.verbose:
            print('plot callback complete')

    def connect(self, fig):
        if isinstance(fig, int):
            fig =plt.figure(fig)
        cid = fig.canvas.mpl_connect('button_press_event', self)


def cfigure(*args, **kwargs):
    """ Create Matplotlib figure with copy to clipboard functionality

    By pressing the 'c' key figure is copied to the clipboard

    """
    if 'facecolor' in kwargs:
        fig = plt.figure(*args, **kwargs)
    else:
        fig = plt.figure(*args, facecolor='w', **kwargs)
    ff = lambda xx, figx=fig: mpl2clipboard(fig=figx)
    fig.canvas.mpl_connect('key_press_event', ff)  # mpl2clipboard)
    return fig

#%%

try:
    def monitorSizes(verbose=0):
        """ Return monitor sizes """
        _qd = QtWidgets.QDesktopWidget()
        if sys.platform == 'win32' and _qd is None:
            import ctypes
            user32 = ctypes.windll.user32
            wa = [
                [0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]]
        else:
            # tmp=QtWidgets.QApplication.startingUp()
            # logging.debug('starting QApplication: startingUp %d' % tmp)
            _applocalqt = QtWidgets.QApplication.instance()

            if _applocalqt is None:
                _applocalqt = QtWidgets.QApplication([])
                _qd = QtWidgets.QDesktopWidget()
            else:
                _qd = QtWidgets.QDesktopWidget()

            nmon = _qd.screenCount()
            wa = [_qd.screenGeometry(ii) for ii in range(nmon)]
            wa = [[w.x(), w.y(), w.width(), w.height()] for w in wa]

        if 0:
            # import gtk # issues with OpenCV...
            window = gtk.Window()
            screen = window.get_screen()

            nmon = screen.get_n_monitors()
            wa = [screen.get_monitor_geometry(ii) for ii in range(nmon)]
            wa = [[w.x, w.y, w.width, w.height] for w in wa]

        if verbose:
            for ii, w in enumerate(wa):
                print('monitor %d: %s' % (ii, str(w)))
        return wa
except:
    def monitorSizes(verbose=0):
        """ Dummy function for monitor sizes """
        return [[0, 0, 1600, 1200]]
    pass

#%%


def getWindowRectangle():
    """ Return current window rectangle """
    x, y, w, h = None, None, None, None
    mngr = plt.get_current_fig_manager()
    be = matplotlib.get_backend()
    if be == 'WXAgg':
        (x, y) = mngr.canvas.manager.window.GetPosition(x, y)
        (w, h) = mngr.canvas.manager.window.GetSize()
    elif be == 'TkAgg':
        print('getWindowRectangle: not implemented...')
        #_=mngr.canvas.manager.window.wm_geometry("%dx%d+%d+%d" % (w,h,x,y))
    elif be == 'module://IPython.kernel.zmq.pylab.backend_inline':
        pass
    else:
        # assume Qt canvas
        g = mngr.canvas.manager.window.geometry()
        x, y, w, h = g.left(), g.top(), g.width(), g.height()
        # mngr.window.setGeometry(x,y,w,h)
    return (x, y, w, h)


def setWindowRectangle(x, y=None, w=None, h=None, mngr=None, be=None):
    """ Position the current Matplotlib figure at the specified position
    Usage: setWindowRectangle(x,y,w,h)
    """
    if y is None:
        y = x[1]
        w = x[2]
        h = x[3]
        x = x[0]
    if mngr is None:
        mngr = plt.get_current_fig_manager()
    be = matplotlib.get_backend()
    if be == 'WXAgg':
        mngr.canvas.manager.window.SetPosition((x, y))
        mngr.canvas.manager.window.SetSize((w, h))
    elif be == 'TkAgg':
        _ = mngr.canvas.manager.window.wm_geometry("%dx%d+%d+%d" % (w, h, x, y))
    elif be == 'module://IPython.kernel.zmq.pylab.backend_inline':
        pass
    else:
        # assume Qt canvas
        mngr.canvas.manager.window.move(x, y)
        mngr.canvas.manager.window.resize(w, h)
        mngr.canvas.manager.window.setGeometry(x, y, w, h)
        # mngr.window.setGeometry(x,y,w,h)

try:
    # http://forums.xkcd.com/viewtopic.php?f=11&t=99890
    import msvcrt

    def getkey():
        """ Cross-platform get key function """
        if msvcrt.kbhit():
            k = msvcrt.getch()
            return k
        return None
except:
    pass


def raiseWindow(fig):
    """ Raise a matplotlib window to to front """
    plt.figure(fig)  # plt.show()
    w = pylab.get_current_fig_manager().window
    w.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
    w.show()

#%%


def ca():
    """ Close all open windows """
    plt.close('all')

#%%


@static_var('monitorindex', -1)
def tilefigs(lst, geometry=[2, 2], ww=None, raisewindows=False, tofront=False, verbose=0):
    """ Tile figure windows on a specified area

    Arguments
    ---------
        lst : list
                list of figure handles or integers
        geometry : 2x1 array
                layout of windows

    """
    mngr = plt.get_current_fig_manager()
    be = matplotlib.get_backend()
    if ww is None:
        ww = monitorSizes()[tilefigs.monitorindex]

    w = ww[2] / geometry[0]
    h = ww[3] / geometry[1]

    # wm=plt.get_current_fig_manager()

    if isinstance(lst, int):
        lst = [lst]

    if verbose:
        print('tilefigs: ww %s, w %d h %d' % (str(ww), w, h))
    for ii, f in enumerate(lst):
        if isinstance(f, matplotlib.figure.Figure):
            fignum = f.number
        else:
            fignum = f
        if not plt.fignum_exists(fignum):
            if verbose >= 2:
                print('tilefigs: fignum: %s' % str(fignum))
            continue
        fig = plt.figure(fignum)
        iim = ii % np.prod(geometry)
        ix = iim % geometry[0]
        iy = np.floor(float(iim) / geometry[0])
        x = ww[0] + ix * w
        y = ww[1] + iy * h
        if verbose:
            print('ii %d: %d %d: f %d: %d %d %d %d' %
                  (ii, ix, iy, fignum, x, y, w, h))
            if verbose >= 2:
                print('  window %s' % mngr.get_window_title())
        if be == 'WXAgg':
            fig.canvas.manager.window.SetPosition((x, y))
            fig.canvas.manager.window.SetSize((w, h))
        if be == 'WX':
            fig.canvas.manager.window.SetPosition((x, y))
            fig.canvas.manager.window.SetSize((w, h))
        if be == 'agg':
            fig.canvas.manager.window.SetPosition((x, y))
            fig.canvas.manager.window.resize(w, h)
        if be == 'Qt4Agg' or be == 'QT4' or be == 'QT5Agg':
            # assume Qt canvas
            try:
                fig.canvas.manager.window.move(x, y)
                fig.canvas.manager.window.resize(w, h)
                fig.canvas.manager.window.setGeometry(x, y, w, h)
                # mngr.window.setGeometry(x,y,w,h)
            except Exception as e:
                print('problem with window manager: ', )
                print(be)
                print(e)
                pass
        if raisewindows:
            mngr.window.raise_()
        if tofront:
            plt.figure(f)

#%%


def robustCost(x, thr, method='L1'):
    """ Robust cost function

    method : string
        method to be used. use 'show' to show the options

    Example
    -------
    >>> robustCost([2,3,4],thr=2.5)
    array([ 2. ,  2.5,  2.5])
    >>> robustCost(2, thr=1)
    1
    >>> methods=robustCost(np.arange(-5,5,.2), thr=2, method='show')
    """
    if method == 'L1':
        y = np.minimum(np.abs(x), thr)
    elif method == 'L2' or method == 'square':
        y = np.minimum(x * x, thr)
    elif method == 'BZ':
        alpha = thr * thr
        epsilon = np.exp(-alpha)
        y = -np.log(np.exp(-x * x) + epsilon)
    elif method == 'BZ0':
        # print('BZ0')
        alpha = thr * thr
        epsilon = np.exp(-alpha)
        y = -np.log(np.exp(-x * x) + epsilon) + np.log(1 + epsilon)
    elif method == 'cauchy':
        b2 = thr * thr
        d2 = x * x
        y = np.log(1 + d2 / b2)
    elif method == 'cg':
        delta = x
        delta2 = delta * delta
        w = 1 / thr  # ratio of std.dev
        w2 = w * w
        A = .1  # fraction of outliers
        y = -np.log(A * np.exp(-delta2) + (1 - A) * np.exp(-delta2 / w2) / w)
        y = y + np.log(A + (1 - A) * 1 / w)
    elif method == 'huber':
        d2 = x * x
        d = 2 * thr * np.abs(x) - thr * thr
        y = d2
        idx = np.abs(y) >= thr * thr
        y[idx] = d[idx]
    elif method == 'show':
        plt.figure(10)
        plt.clf()
        mm = ['L1', 'L2', 'BZ', 'cauchy', 'huber', 'cg']
        for m in mm:
            plt.plot(x, robustCost(x, thr, m), label=m)
        plt.legend()
        # print('robustCost: %s'  % mm)
        y = mm
    else:
        raise Exception('no such method')
    return y

# robustCost(np.arange(-5,5,.1), 2, 'show')

#%%


class timeConverter:

    """ Class to convert time between current time and cyclic log times """
    tstart = None
    tstop = None
    starttimeNow = None
    deltaTime = None
    hh = 0

    def __init__(self, tstart, tstop, startnow):
        self.tstart = tstart
        self.tstop = tstop
        self.loopTime = tstop - tstart
        self.starttimeNow = startnow
        self.deltaTime = startnow - tstart

    def __repr__(self):
        s = 'timeConverter: looptime %.1f [s]' % self.loopTime
        return s

    def current2cyclic(self, t):
        return ((t - self.deltaTime) - self.tstart) % self.loopTime + self.tstart

    def fraction(self, t):
        s = (t - self.starttimeNow)
        f = (s % self.loopTime) / self.loopTime
        return f

    def loopindexlog(self, logtime):
        loopidx = math.floor(float(logtime - self.tstart) / self.loopTime)
        return loopidx

    def loopindex(self, tnow):
        loopidx = math.floor(float(tnow - self.starttimeNow) / self.loopTime)
        return loopidx

    def cyclic2current(self, clt, currenttime, verbose=0):
        tau = currenttime - clt - self.deltaTime
        k = tau / self.loopTime
        loopidx = round(k)
        current = clt + loopidx * self.loopTime + self.deltaTime
        return current

    def passLog(tc, clt, tnow):
        """ Determine whether at current time (tnow) we have passed an entry in the log """

        # logtimenow = tc.cyclic2current(clt, tnow)

        return tc.cyclic2current(clt, tnow) < tnow

    def current2logtime(self, t):
        return t - self.deltaTime

    def cyclic2logtime(self, clt, currenttime, verbose=0):
        current = self.cyclic2current(clt, currenttime, verbose)
        return current - self.deltaTime

    def logtime2current(self, logtime):
        return logtime + self.deltaTime

    def getidx(tc, tnow, logtimes):
        tcylic = tc.current2cyclic(tnow)

        N = logtimes.size
        idx = np.interp(tcylic, logtimes, np.arange(N))
        idx = int(idx)
        idx = np.minimum(idx, N - 1)
        idx = np.maximum(idx, 0)
        return idx

#%% Testing area


def testTimeconverter():
    """ Testing function for the TimeConverter class """
    tc = timeConverter(0, 100, -20)

    tthenlin = np.arange(0, 500, 0.1)
    tcyclic = tthenlin % tc.loopTime
    tnow = tthenlin.copy()
    tthen = tthenlin.copy()
    for ii in range(tnow.size):
        tnow[ii] = tc.cyclic2current(tcyclic[ii], tnow[ii], 0)
        tthen[ii] = tc.cyclic2logtime(tcyclic[ii], tnow[ii], 0)

    plt.figure(100)
    plt.clf()
    plt.plot(tthenlin, tcyclic, '-r')
    plt.plot(tthenlin, tthen, '.b')
    plt.plot(tthenlin, tnow, '.g')
    plt.xlabel('Logtime')
    plt.ylabel('Current time')
    plot2Dline([-1, 0, tc.starttimeNow], 'g')

# testTimeconverter()

#%% Wrapper to read video files


class videoreader_t:

    def __init__(self, vidfile):
        try:
            import avireader
            self._haveavireader = True
        except:
            self._haveavireader = False
        if self._haveavireader:
            self.cap = avireader.videoreader(vidfile)
        else:
            self.cap = cv2.VideoCapture(vidfile)
        self.resize = False

    def isOpened(self):
        if self._haveavireader:
            return True
        else:
            return self.cap.isOpened()

    def get_vid_properties(self):
        if self._haveavireader:
            nframes = self.cap.nframes()
            fheight = self.cap.height()
            fwidth = self.cap.width()
            fps = self.cap.framerate()
            return nframes, fps, fheight, fwidth
        else:
            return get_vid_properties(self.cap)

    def read(self, index=None):
        if self._haveavireader:
            if index is None:
                im = self.cap.read()
                r = None
            else:
                return None, self.cap.read(idx=index)
        else:
            if index is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            r, im = self.cap.read()

        if self.resize:
            im = cv2.resize(im, None, fx=.25, fy=.25)
        # cap = cv2.VideoCapture(vidfile)
        return r, im
    def close():
        pass

#%% Camera control


class uvccamera:

    """ Class representing a UVC camera

    Implementation is Linux only...
    """
    camid = None
    verbose = 0

    def __init__(self, camid):
        self.camid = camid

    def commandoutput(self, cmd):
        # print(cmd)
        # print(cmd.split(' '))
        r = subprocess.check_output(cmd, shell=True)
        # r=subprocess.check_output(cmd.split(' '), shell=False)
        r = r.decode('ASCII').strip()
        return r

    def command(self, cmd):
        if platform.os.name == 'posix':
            r = os.system(cmd)
        else:
            print('uvc camera setting not supported on your OS')

    def devicename(self):
        return '/dev/video%d' % self.camid

    def __repr__(self):
        s = 'uvccamera: device %s' % self.devicename()
        return s

    def editControls(self):
        cmd = 'guvcview -o -d %s &' % self.devicename()
        self.command(cmd)

    def ledOff(self):
        cmd = 'uvcdynctrl -d %s --set="LED1 Mode" 0' % self.devicename()
        self.command(cmd)

    def setFocus(self, fc=None):
        camname = self.devicename()
        if fc is None:
            cmd = 'uvcdynctrl -d %s --set="Focus, Auto" 1' % camname
            return
        cmd = 'uvcdynctrl -d %s --set="Focus, Auto" 0' % camname
        self.command(cmd)
        fcx = fc + 130
        cmd = 'uvcdynctrl -d %s --set="Focus (absolute)" %d' % (camname, fcx)
        if self.verbose:
            print('  ' + cmd)
        self.command(cmd)
        time.sleep(.35)
        cmd = 'uvcdynctrl -d %s --set="Focus (absolute)" %d' % (camname, fc)
        if self.verbose:
            print('  ' + cmd)
        self.command(cmd)

    def listProperties(self):
        # cmd = 'uvcdynctrl -c -d %s' % self.devicename()
        # self.command(cmd)
        import subprocess
        lst = subprocess.check_output(['uvcdynctrl', '-c', '-d', '%s' % self.devicename()])
        lst = lst.decode('ASCII').split('\n')
        lst = [l.strip() for l in lst[1:]]

        lst = [l for l in lst if len(l) > 1]

        for l in lst:
            # l='Brightness'
            time.sleep(.02)
            # r=self.commandoutput('uvcdynctrl -d /dev/video0 --get="%s"' % l )
            r = self.getProperty(l)

            # r=self.commandoutput('uvcdynctrl -d /dev/video0 --get="%s"' % l)
            # cmd = 'uvcdynctrl -d %s --get="%s"' % ('/dev/video0', 'Brightness')
            # r=self.commandoutput(cmd)
            print('prop %s: %s' % (l, r))

    def getProperty(self, prop):
        camname = self.devicename()
        cmd = 'uvcdynctrl -d %s --get="%s"' % (camname, prop)
        r = self.commandoutput(cmd)
        return r

    def setProperty(self, prop, value):
        camname = self.devicename()
        cmd = 'uvcdynctrl -d %s --set="%s" %s' % (camname, prop, value)
        if self.verbose:
            print('uvccamera: set property:  ' + cmd)
        self.command(cmd)

    def ledBlink(self, freq=10):
        camname = self.devicename()
        cmd = 'uvcdynctrl -d %s --set="LED1 Mode" 2' % camname
        if self.verbose:
            print('ledBlink: ' + cmd)
        r = os.system(cmd)
        cmd = 'uvcdynctrl -d %s --set="LED1 Frequency" %d' % (camname, freq)
        # print('  ' + cmd)
        self.command(cmd)

    def listControls(self):
        cmd = 'uvcdynctrl -c -d %s' % self.devicename()
        # self.command(cmd)
        r = subprocess.check_output(cmd.split(' ')).decode('ASCII')
        print(r)

    @staticmethod
    def listCameras():
        # cmd = 'uvcdynctrl -l'
        if platform.os.name == 'posix':
            r = subprocess.check_output(['uvcdynctrl', '-l']).decode('ASCII')
            print(r)
            # r = os.system(cmd)
        else:
            print('uvc camera setting not supported on your OS')


#%%

def findImageHandle(fig, verbose=0, otype=matplotlib.image.AxesImage):
    """ Search for specific type of object in Matplotlib figure """
    cc = fig.get_children()
    if verbose:
        print('findImageHandle: %s: %d children' % (str(fig), len(cc)))
    for c in cc:
        if isinstance(c, otype):
            return c
        p = findImageHandle(c, verbose=verbose, otype=otype)
        if p is not None:
            return p
        if verbose >= 2:
            print(type(c))
        # return p
    return None


#%% Make nice plots
#

def niceplot(ax, fig=None, despine=True, verbose=0, figurebg=True, tightlayout=True, legend=None, almost_black='#222222', tickdirection='out', debug=0, figpad=1.0):
    """ Create a good looking plot

    The code
        - removes spines
        - makes legend and spines lighter
        - makes legend lighter

    See also: http://blog.olgabotvinnik.com/prettyplotlib/

    Arguments
    ---------
        ax : axis handle
            Handle for axis to process
        fig : integer
            Figure handle to use


    """

    if debug > 16:
        plt.pause(0.001)
        plt.draw()
        plt.show()
        return

    if isinstance(fig, int):
        fig = plt.figure(fig)

    # Remove top and right axes lines ("spines")
    if verbose >= 2:
        print('niceplot: ax %s, fig %s' % (str(ax), str(fig)))
    if verbose:
        print('niceplot: remove spines')

    spines_to_keep = ['bottom', 'left']
    if despine:
        spines_to_remove = ['top', 'right']
        for spine in spines_to_remove:
            ax.spines[spine].set_visible(False)
    else:
        spines_to_keep += ['top', 'right']
    if debug > 10:
        plt.pause(0.001)
        plt.draw()
        plt.show()
        return

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if verbose:
        print('niceplot: reduce spine intensity')

    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(almost_black)

    ax.tick_params(axis='both', direction=tickdirection)

    if debug > 6:
        return

    if legend is not None:
        if verbose:
            print('niceplot: adjust legend')

        # Remove the line around the legend box, and instead fill it with a light grey
        # Also only use one point for the scatterplot legend because the user will
        # get the idea after just one, they don't need three.
        light_grey = np.array([float(241) / float(255)] * 3)
        rect = legend.get_frame()
        rect.set_facecolor(light_grey)
        middle_grey = np.array([float(151) / float(255)] * 3)
        rect.set_edgecolor(middle_grey)
        # rect.set_linewidth(0.0)

        # Change the legend label colors to almost black, too
        texts = legend.texts
        for t in texts:
            t.set_color(almost_black)

        # ttx=legend.get_texts()
        #[v.set_color(almost_black) for v in ttx]

    # fig.tight_layout(pad=0.1)
    if fig is not None and tightlayout:
        fig.tight_layout(pad=figpad)

    if figurebg and (fig is not None):
        fig.set_facecolor('w')

    plt.draw()
    plt.show()


def otsu(im, fig=None):
    """ Calculate threshold on data using Otsu's method

    Arguments
    ---------
    im : array
        data to be processed
    fig : number, optional
        If set to a number show results in a histogram

    Returns
    -------
    thr : float
        The threshold value

    Examples
    --------

    >>> thr = otsu(np.random.rand( 2000), fig=100)

    """
    thr = skimage.filters.threshold_otsu(im)

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        hist, bin_edges = np.histogram(im.flatten(), bins=36)
        bwidth = np.mean(np.diff(bin_edges))
        plt.bar(bin_edges[:-1], hist, width=bwidth)

        plt.xlabel('Value')
        plot2Dline([-1, 0, thr], '--g', linewidth=2, label='Otsu')
        plt.title('Otsu: threshold')
        plt.xlim(min(bin_edges), max(bin_edges))
    return thr

#%%


def histogram(x, nbins=30, fig=1):
    """
    >>> _=histogram(np.random.rand(1,100))
    """
    nn, bin_edges = np.histogram(x, bins=nbins)
    bwidth = np.mean(np.diff(bin_edges))

    if fig:
        plt.figure(fig)
        plt.clf()
        h = plt.bar(bin_edges[:-1:], nn, color='b', width=bwidth)
        plt.ylabel('Frequency')
        return nn, bin_edges, h
    return nn, bin_edges, None


#%%
def decomposeProjectiveTransformation(H, verbose=0):
    """ Decompose projective transformation
    H is decomposed as H = Hs*Ha*Hp with

 Hs = [sR t]
      [0  1]

 Ha = [K 0]
      [0 1]

 Hp = [I 0]
      [v' eta]

 If H is 3-dimensional, then R = [ cos(phi) -sin(phi); sin(phi) cos(phi)];

 For more information see "Multiple View Geometry", paragraph 1.4.6.

    >>> Ha, Hs, Hp, rest = decomposeProjectiveTransformation( np.eye(3) )
    """
    H = np.matrix(H)
    k = H.shape[0]
    km = k - 1

    eta = H[k - 1, k - 1]
    Hp = np.matrix(np.vstack((np.eye(km, k), H[k - 1, :])))
    A = H[0:km, 0:km]
    t = H[0:km, -1]
    v = H[k - 1, 0:km].T

    eps = 1e-10
    if np.abs(np.linalg.det(A)) < 4 * eps:
        print('decomposeProjectiveTransformation: part A of matrix is (near) singular')

    sRK = A - np.matrix(t).dot(np.matrix(v.T))
    # upper left block of H*inv(Hp)
    R, K = np.linalg.qr(sRK)
    K = np.matrix(K)
    R = np.matrix(R)

    s = (np.abs(np.linalg.det(K)))**(1. / km)
    K = K / s

    if k == 2 and K[0, 0] < 0:  # in 3-dimensional case normalize sign
        K = np.diag([-1, 1]) * K
        R = R.dot(np.diag([-1, 1]))
    else:
        # primitive...
        sc = np.sign(np.diag(K))
        K = np.diag(sc) * K
        R = R * np.diag(sc)
    br = np.hstack((np.zeros((1, km)), np.ones((1, 1))))
    Hs = np.matrix(np.vstack((np.hstack((s * R, t.reshape((-1, 1)))), br)))
    Ha = np.matrix(np.vstack((np.hstack((K, np.zeros((km, 1)))), br)))

    phi = np.arctan2(R[1, 0], R[0, 0])

    if verbose:
        print('decomposeProjectiveTransformation: size %d' % k)
    rest = (s, phi, t, v, )
    return Ha, Hs, Hp, rest

#%% Geometry


def points_in_polygon(pts, pp):
    """

    Arguments:
        pt (Nx2 array): point
        pp (Nxk array): polygon
    """
    rr = np.zeros(len(pts))
    for i, pt in enumerate(pts):
        r = cv2.pointPolygonTest(np.array(pp).astype(np.float32), (pt[0], pt[1]), measureDist=False)
        rr[i] = r
    return rr


def point_in_polygon(pt, pp):
    """

    Arguments:
        pt (2x1 array): point
        pp (2xN array): polygon
    """
    r = cv2.pointPolygonTest(pp, (pt[0], pt[1]), measureDist=False)
    return r


def minAlg_5p4(A):
    """ Algebraic minimization function

    Function computes the vector x that minimizes ||Ax|| subject to the
    condition ||x||=1.
    Implementation of Hartley and Zisserman A5.4 on p593 (2nd Ed)

    Usage:   [x,V] = minAlg_5p4(A)
    Arguments:
             A (numpy array) : The constraint matrix, ||Ax|| to be minimized
    Returns:
              x - The vector that minimizes ||Ax|| subject to the
                  condition ||x||=1

    """

    # Compute the SVD of A
    (U, D, V) = np.linalg.svd(A)

    # Take last vector in V
    x = V[-1, :]
    return x


def fitPlane(X):
    """ Determine plane going through a set of points

    >>> X=np.array([[1,0,0 ], [0,1,0], [1,1,0], [2,2,0]])
    >>> t=fitPlane(X)

    """

    AA = np.vstack((X.T, np.ones(X.shape[0]))).T
    t = minAlg_5p4(AA)
    return t

#%% Point clouds

try:
    from plyfile import PlyData, PlyElement
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg

    class MyView(gl.GLViewWidget):

        def pan(self, dx, dy, dz, relative=False):
            """
            Moves the center (look-at) position while holding the camera in place.

            If relative=True, then the coordinates are interpreted such that x
            if in the global xy plane and points to the right side of the view, y is
            in the global xy plane and orthogonal to x, and z points in the global z
            direction. Distances are scaled roughly such that a value of 1.0 moves
            by one pixel on screen.

            """
            if not relative:
                self.opts['center'] += QtGui.QVector3D(dx, dy, dz)
            else:
                cPos = self.cameraPosition()
                cVec = self.opts['center'] - cPos
                dist = cVec.length()  # distance from camera to center
                xDist = dist * 2. * np.tan(0.5 * self.opts['fov'] * np.pi / 180.)  # approx. width of view at distance of center point
                xScale = xDist / self.width()
                zVec = QtGui.QVector3D(0, 0, 1)
                xVec = QtGui.QVector3D.crossProduct(zVec, cVec).normalized()
                # xVec = QtGui.QVector3D(0,1,0)
                # yVec = QtGui.QVector3D.crossProduct(xVec, zVec).normalized()
                yVec = QtGui.QVector3D(0, 0, 1)
                self.opts['center'] = self.opts['center'] + xVec * xScale * dx + yVec * xScale * dy + zVec * xScale * dz
            self.update()

    class pointcloud_t:

        def __init__(self):

            self.X = np.array((0, 3))
            self.color = None
            self.faces = None

        def transform(self, T):
            pc0 = pointcloud_t()
            pc0.X = projectiveTransformation(T, self.X.T).T
            pc0.color = self.color
            pc0.faces = self.faces
            return pc0

        def nVertices(self):
            """ Return the number of vertices in the cloud """
            return self.X.shape[0]

        def __repr__(self):
            return 'pointcloud_t: %d vertices' % self.X.shape[0]

        def loadPly(self, plyfile):
            pd = PlyData.read(plyfile)

            self.X = np.vstack((pd['vertex']['x'], pd['vertex']['y'], pd['vertex']['z'])).T
            try:
                self.color = np.vstack((pd['vertex']['diffuse_red'], pd['vertex']['diffuse_green'], pd['vertex']['diffuse_blue'])).T
            except:
                pass
            return pd

        @staticmethod
        def addPoints(glview, X, color=None, size=.1):
            nn = X.shape[0]
            if color is None:
                col = np.zeros((nn, 4))
                col[:, 1] = 1
                col[:, -1] = 1
            else:
                if color.size == 4:
                    col = np.repeat(color.reshape(1, 4), nn, axis=0)
                else:
                    col = color
            sizedata = size * np.ones(nn)
            sp1 = gl.GLScatterPlotItem(pos=X, size=sizedata, color=col, pxMode=False)
            glview.addItem(sp1)
            return sp1

        @staticmethod
        def drawLine(glview, c1, c2, color=None, nn=20, width=5):
            X = np.array([c1.flatten(), c2.flatten()])
            nn = 2
            if color is None:
                col = np.zeros((nn, 4))
                col[:, 1] = 1
                col[:, -1] = 1
            else:
                if color.size == 4:
                    col = np.repeat(color.reshape(1, 4), nn, axis=0)
                else:
                    col = color
            # sizedata=size*np.ones( nn)
            sp1 = gl.GLLinePlotItem(pos=X, color=col, width=width)  # pxMode=False)
            glview.addItem(sp1)
            return sp1

        @staticmethod
        def show_cloud(pc=None, addgrid=False):
            _ = pg.mkQApp()
            w = gl.GLViewWidget()
            w.opts['distance'] = 20
            w.show()
            w.setWindowTitle('pointcloud_t')

            if addgrid:
                g = gl.GLGridItem()
                w.addItem(g)

            if pc is not None:
                nn = pc.nVertices()
                sz = float(np.mean(np.std(pc.X, axis=0)) / 400.)

                size = sz * np.ones(pc.X.shape[0])
                color = .25 * np.ones((nn, 4))
                if pc.color is None:
                    color = None
                else:
                    color[:, 0:3] = .5 * (pc.color / 255.)
                pointcloud_t.addPoints(w, pc.X, color, size=size)
            return w

        @staticmethod
        def clearPlot(w):
            while len(w.items) > 0:
                w.removeItem(w.items[0])


except:
    # no pointcloud functionality available
    pass


#%% Debugging

import threading


class ThreadObject:

    def __init__(self, name, worker=None, wait_time=.1, **kwargs):
        ''' Class to run code in the background with a specified time delay '''
        self.name = name
        self.worker = worker
        self.wait_time = wait_time
        self.active = 1
        self.mythread = threading.Thread(target=self.worker)
        self.pverbose = 0

        self.fps = fps_t(10)
        self.loopidx = 0

    def switch(self):
        if self.active:
            self.active = 0
        else:
            self.run()

    def abort(self):
        self.active = 0

    def stop(self):
        self.abort()

    def start(self, worker=None):
        if worker is None:
            worker = self.worker
        try:
            self.mythead.terminate()
            self.active = 1
        except:
            pass
        self.mythread = threading.Thread(target=self.run)
        self.mythread.start()
    def run(self):
        ii = 0
        self.active = 1
        if self.pverbose:
            print('%s: run start' % self.name)
        while True:
            if self.pverbose:
                print('%s: run %d ' % (self.name, ii))
            logging.info('pid %d: %s: loop %d' % (os.getpid(), self.name, ii), tag=self.name, dt=5)
            self.fps.addtime(time.time())
            ii = ii + 1

            if self.worker is not None:
                if self.pverbose:
                    print('%s: run %d: call worker ' % (self.name, ii))
                self.worker()
            time.sleep(self.wait_time)

            if not self.active:
                logging.info('%s: aborting loop' % self.name)
                break

        if 1:
            if self.pverbose:
                print('%s: run done' % self.name)

    def __repr__(self):
        return ('%s: active %d' % (self.__class__, self.active))


def modulepath(m):
    package = pkgutil.get_loader(m)
    if package is None:
        return None
    return package.get_filename()


def checkmodule(mname, verbose=1):
    import imp
    q = imp.find_module(mname)
    import importlib
    q = importlib.util.find_spec(mname)
    if verbose:
        print(q)
    return q


def test_geometry(verbose=1, fig=None):
    im = np.zeros((200, 100, 3))
    subim = np.ones((40, 30,))
    im = setregion(im, subim, [0, 0])
    im = np.zeros((200, 100, 3))
    subim = np.ones((40, 30,))
    im = setregion(im, subim, [95, 0], clip=True)
    if fig:
        plt.figure(fig)
        plt.clf()
        plt.imshow(im, interpolation='nearest')

def unittest(verbose=1):
    """ Unittest function for module """
    import doctest
    if verbose >= 2:
        print('pgeometry: running unittest')
    _ = euler2RBE([0, 1, 2])
    doctest.testmod()

#%% Run tests from documentation
if __name__ == "__main__":
    """ Dummy main for doctest
    Run python pgeometry.py -v to test the module
    """
    import doctest
    doctest.testmod()


#%% Testing zone
