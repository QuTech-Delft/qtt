""" Legacy functions (do not use) """
import copy

import numpy as np
import scipy
import matplotlib
import sys
import os
import logging
import cv2
import time
import math
import pickle
import warnings

import qcodes
# explicit import
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot
from qtt.algorithms.images import straightenImage

import qtt.data
from qtt.data import loadExperimentData
import qtt.algorithms.onedot
from qtt.measurements.scans import scanjob_t
import matplotlib.pyplot as plt
import datetime

from qtt.measurements.scans import sample_data_t, enforce_boundaries

# %%

from qtt.data import dataset2Dmetadata, dataset2image

from qtt.algorithms.onedot import onedotGetBalanceFine
from qtt.measurements.scans import fixReversal
from qtt.data import load_data, show2D
from qtt.utilities.tools import diffImage, diffImageSmooth, rdeprecated
from qtt.algorithms.generic import smoothImage
#from qtt.measurements.scans import scanPinchValue


from qtt import pgeometry as pmatlab
from qtt.pgeometry import plotPoints, tilefigs

warnings.warn('please do not this import this module')

# %%

try:
    import graphviz
except:
    pass
import matplotlib.pyplot as plt


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1 Sep 2018')
def showDotGraph(dot, fig=10):
    dot.format = 'png'
    outfile = dot.render('dot-dummy', view=False)
    print(outfile)

    im = plt.imread(outfile)
    plt.figure(fig)
    plt.clf()
    plt.imshow(im)
    plt.tight_layout()
    plt.axis('off')


# %%


@rdeprecated(txt='Method will be removed in future release of qtt', expire='7-1-2018')
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


# %%


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1-7-2018')
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
    qtt.utilities.tools.mkdirc(os.path.join(resultsdir, 'pictures'))
    imfilerel = os.path.join('pictures', imfile0)

    if fig is not None:
        plt.figure(fig)
    if tight:
        plt.savefig(imfile, dpi=dpi, bbox_inches='tight', pad_inches=tight)
    else:
        plt.savefig(imfile, dpi=dpi)
    return imfilerel, imfile


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1-7-2019')
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


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1 Sep 2018')
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


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1-1-2019')
def writeBatchData(outputdir, tag, timestart, timecomplete):
    tt = datetime.datetime.now().strftime('%d%m%Y-%H%m%S')
    with open(os.path.join(outputdir, '%s-%s.txt' % (tag, tt)), 'wt') as fid:
        fid.write('Tag: %s\n' % tag)
        fid.write('Time start: %s\n' % timestart)
        fid.write('Time complete: %s\n' % timecomplete)
        fid.close()
        print('writeBatchData: %s' % fid.name)

# %%


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1 Sep 2018')
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


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1 Sep 2018')
def filterGabor(im, theta0=-np.pi / 8, istep=1, widthmv=2, lengthmv=10, gammax=1, cut=None, verbose=0, fig=None):
    """
    Filter image with Gabor

    step is in pixel/mV

    Parameters
    ----------

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

    gfilter = pmatlab.gaborFilter(ksize, sigma=sigmax, theta=theta0, Lambda=cwidth,
                                  psi=0, gamma=sigmax / sigmay, cut=cut)
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


# %%


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1 Sep 2018')
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

    def reduced_cmap(step): return np.array(cmap(step)[0:3])
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


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1 Sep 2018')
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

# %%


from qtt.algorithms.misc import polyval2d, polyfit2d

from qtt.utilities.imagetools import fitBackground as fitBackgroundTmp
from qtt.utilities.imagetools import cleanSensingImage

fitBackground = qtt.utilities.tools.deprecated(fitBackgroundTmp)


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1 Sep 2018')
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


# %%
from qtt.algorithms.misc import point_in_poly, points_in_poly, fillPoly


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1 Sep 2018')
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
            print('getPinchvalues: gate %s : %.2f' % (g, adata['pinchoff_point']))
        od['pinchvalues'][jj] = adata['pinchoff_point']
    return od


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1 Sep 2018')
def createDoubleDotJobs(two_dots, one_dots, resultsdir, basevalues=dict(), sdinstruments=[], fig=None, verbose=1):
    """ Create settings for a double-dot from scans of the individual one-dots """
    raise Exception('function was removed from qtt')


# %%

@rdeprecated(txt='Method will be removed in future release of qtt', expire='1-1-2019')
def printGateValues(gv, verbose=1):
    s = ', '.join(['%s: %.1f' % (x, gv[x]) for x in sorted(gv.keys())])
    return s


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1-1-2019')
def getODbalancepoint(od):
    bp = od['balancepoint']
    if 'balancepointfine' in od:
        bp = od['balancepointfine']
    return bp


@rdeprecated(txt='Method will be removed in future release of qtt', expire='1-6-2018')
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
