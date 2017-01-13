# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:40:53 2015

@author: eendebakpt
"""

#%% Load packages
from __future__ import division

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot
import scipy
import copy
import skimage.filters

try:
    from skimage import morphology
except:
    pass        

_linetoolswarn=False

import numpy as np

try:
    import shapely
    import shapely.geometry
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    from shapely.geometry import LineString
except:
    if not _linetoolswarn:
        #warnings.warn('module shapely not found')
        _linetoolswarn=True
try:
    from descartes.patch import PolygonPatch
except:
    if not _linetoolswarn:
        #warnings.warn('module descartes not found')
        _linetoolswarn=True

from qtt import pmatlab
from qtt.pmatlab import *
import cv2

try:
    import igraph
except:
    if not _linetoolswarn:
        pass


from qtt.legacy import cmap_discretize
from qtt.algorithms.generic import scaleImage, smoothImage, localMaxima

try:
    import networkx as nx
except:
    print('networkx module is not available')
    pass


#%% Try numba support
try:
    from numba import autojit  # ,jit
except:
    def autojit(original_function):
        def dummy_function(*args, **kwargs):
            return original_function(*args, **kwargs)
        return dummy_function

    pass

#%% Functions


# plocal: xl, theta_l, w, h
# pglobal: x,y,theta

def showIm(ims, fig=1, title=''):
    """ Show image with nearest neighbor interpolation and axis scaling """
    matplotlib.pyplot.figure(fig)
    matplotlib.pyplot.clf()
    matplotlib.pyplot.imshow(ims, interpolation='nearest')
    matplotlib.pyplot.axis('image')
    # scaleCmap(ims)


def l2g(plocal, pglobal, dy=0):
    """ Convert pair of local-global coordinates to global """
    pp = pglobal.copy()
    # pp[0]+=plocal[0]
    pp[2] += plocal[1]
    pp[0] += math.cos(-pp[2]) * plocal[0] - math.sin(-pp[2]) * dy
    pp[1] += math.sin(-pp[2]) * plocal[0] + math.cos(-pp[2]) * dy
    #pp[0:2] += ( rot2D(-pp[2])*np.array( [[plocal[0]], [dy]] ) ).flat
    return pp


@autojit
def lg2data(pl, pg, score):
    pg = l2g(pl, pg)
    dx = np.hstack((pg, pl[2], pl[3], score))
    return dx
#%%


def dummy():
    print('plt: %s' % str(plt))
    print('matplotlib: %s' % str(matplotlib))

    plt.figure(10)
    return


def getBlobPosition(ims, label_im, idx):
    """ Get starting position from blob """
    cms = scipy.ndimage.measurements.center_of_mass(
        ims, labels=label_im, index=idx)
    xstart0 = np.array(cms).reshape((2, 1))[[1, 0], :]
    ww = (label_im == idx).nonzero()
    ww = np.vstack((ww[1], ww[0])).T

    dd = ww - xstart0.T
    jj = np.argmin(np.linalg.norm(dd, axis=1))
    xstart = ww[jj, :].reshape((2, 1))

    return xstart


def plotCrosssectionModel(patch, xfit, pglobal):
    """ Plot cross section of rectangular patch """
    ksize = patch.shape[1]
    xxfine = np.arange(0, ksize, 0.1)
    xxfine = np.arange(0, ksize, 1)
    tmp, fff = lineModel(
        xfit[2:4], xx=xxfine - (ksize / 2. - .5) - xfit[0], ksize=ksize, fig=None)

    samplesize = (patch.shape[1], patch.shape[0])
    xx = np.arange(ksize)
    vv = patch.mean(axis=0)
    [xxm, yym] = np.meshgrid(
        np.arange(samplesize[0]), np.arange(samplesize[1]))
    plt.plot(xx, vv, '.b', label='Mean of data', markersize=14)
    plt.plot(xxm.flatten(), patch.flatten(), '.c')
    #plt.plot(xx, ff0,'--m', label='Model')
    plt.plot(xxfine, fff, '.-r', label='Fitted model')
    plt.legend(loc=0)
    plt.title('Cross section')


def getpatch(ims, pp, samplesize, fig=None):
    """ Return image patch from parameters 
    """

    #pp = l2g(plocal, pglobal)
    patch = sampleImage(ims, pp, samplesize=samplesize, fig=fig)
    return patch

#@autojit





def sampleImage(im, pp, samplesize, fig=None, clearfig=True, nrsub=1):
    """ Sample image patch

    The patch is sampled and displayed if fig is not None. The image patch is returned

    Arguments
    ---------
    im : numpy array
         The input image
    pp : list
        line parameters
    samplesize : int
        size of patch to sample
    fig :
    clearfig :
    nrsub :

    """

    H = createH(samplesize, pp)

    # H=pg_transl2H(1*c)*pg_rotation2H(rot2D(theta))*pg_transl2H(-cc) # image
    # to patch

    dsize = (samplesize[0], samplesize[1])

    #patch=cv2.warpPerspective(im.astype(float32), H, dsize)
    #patch=cv2.warpPerspective(im.astype(float32), H, dsize, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, -1)
    patch = cv2.warpPerspective(im.astype(
        np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)
    if not fig is None:
        cc = pp[0:2].reshape((2, 1))

        rr = np.array([[0., 0], dsize]).T
        rr = region2poly(rr)
        rrs = np.array([[dsize[0] * .2, 0], [dsize[0] * .8, dsize[1]]]).T
        rrs = region2poly(rrs)
        rrim = projectiveTransformation(H.I, rr)
        rrims = projectiveTransformation(H.I, rrs)
        ff = np.array([[dsize[0] / 2., 0]]).T
        ffim = projectiveTransformation(H.I, ff)
        plt.figure(fig)
        if clearfig:
            plt.clf()
        plt.subplot(nrsub, 2, 1)
        plt.imshow(im)
        plt.axis('image')
        plt.title('Input image')
        plotPoints(cc, '.r', markersize=14)
        plotPoints(rrim, 'b')
        plotPoints(ffim, '.b', markersize=14)
        plotPoints(rrims, '--b')
        ax = plt.subplot(nrsub, 2, 2)
        plt.imshow(patch, interpolation='nearest')
        plt.axis('off')
        # ax.invert_yaxis()
        plt.title('sampleImage')

    return patch


def mergeBackTrace(lm, lmb):
    """ Merge line model and back model """

    plb = lmb.data[::-1, :]
    #plb[:,0]=-    plb[:,0]
    plb[:, 2] += np.pi
    lm.data = np.vstack((plb, lm.data))
    #lm.pglobal=np.vstack( (lmb.pglobal[::-1,:], lm.pglobal))
    #lm.score=np.vstack( (lmb.score[::-1,:], lm.score))
    lm.cidx = lm.cidx + lmb.nPoints()
    return lm


def distancePointResults(p, results, mindist=2):
    dmin = 1e8
    for ii, r in enumerate(results):
        d = r.distance(p)
        #print('distance %.2f' % d)
        dmin = min(d, dmin)
    #print('distance %.2f' % dmin)
    return dmin

#%%


import math


def splitedge(net, p, verbose=1):
    """ Split edge at specified point """
    results0 = net.g.es['linedata']
    ww = [rx.distance(p) for rx in results0]
    ri = np.argmin(ww)
    if verbose:
        print('splitedge: edge %d' % ri)

    r0 = results0[ri]

    if r0.nPoints() == 2:
        pass

    ls = LineString(r0.getXY())
    q = ls.project(Point(p), normalized=True)
    Xi = ls.interpolate(q, normalized=True)

    npoints = len(ls.coords.xy[0])
    vi = (npoints - 1) * q

    vi1 = math.floor(vi)
    vi2 = math.ceil(vi)
    if verbose:
        print('splitedge: edge %d: linestring %d %d' % (ri, vi1, vi2))

    dx0 = r0.data[vi1, :]
    dx = np.array([Xi.x, Xi.y, dx0[2], dx0[3], dx0[4], -1])
    d1 = np.vstack((r0.data[0:vi2, :], dx))
    d2 = np.vstack((dx, r0.data[vi2:, :]))

    ee = net.getEdgeVertices(ri)

    q0 = net.g.vs[ee[0]]['xy']
    q1 = net.g.vs[ee[1]]['xy']
    dist1 = np.linalg.norm(q0 - d1[0, 0:2])
    dist2 = np.linalg.norm(q1 - d1[0, 0:2])

    if dist1 < dist2:
        vx1, vx2 = ee
    else:
        vx2, vx1 = ee

    return d1, d2, Xi, ri, vx1, vx2

#%% Classes


def lineModel(p1, xx=None, ksize=31, fig=None, clearfig=True):
    """ Model for a line to fit

    Arguments
    ---------
        p1: list
            width, height   
        p2: list
            argument not used...
        ksize: integer
            kernel size

    """
    # model:

    w = p1[0]
    h = p1[1]
    if xx is None:
        xx = np.arange(-(ksize - 1) / 2, (ksize + 1) / 2)

    tmp = np.pi * xx / (w)
    tmp[np.abs(tmp) > np.pi] = np.pi
    ff = (h / 2.) * (1 + np.cos(tmp))

    if not fig is None:
        plt.figure(fig)
        if clearfig:
            plt.clf()
        plt.plot(xx, ff, '.-r')
        plt.title('lineModel: w %.1f, h %.1f' % (w, h))

    return xx, ff


try:
    from matplotlib.patches import Polygon as mplPolygon

    from shapely.geometry.polygon import Polygon
    from shapely.geometry import Point
    from descartes.patch import PolygonPatch
    from shapely.geometry import LineString

except:
    pass


def mergeLineModels(lm, lm2):
    lm0 = linemodel()
    if lm.nPoints() == 0:
        lm0 = lm2
        return lm0
    dd = lm.data[-1, 0:2] - lm2.data[0, 0:2]
    if np.any(np.abs(dd) > 5):
        raise Exception('cannot append non-connected linemodels')
    lm0.data = np.vstack((lm.data, lm2.data))
    return lm0


class linemodel:

    """ Model that represents a tracked line 
    """
    #plocal=np.zeros( (0,4))
    #pglobal=np.zeros( (0,3))
    #score=np.zeros( (0,1))
    data = np.zeros((0, 6))  # x,y,theta,w,h
    cidx = -1

    im = None

    def __init__(self, im=None):
        self.im = im

    def __repr__(self):
        s = ('linemodel: %d points' % self.nPoints())
        return s

    def append(lm, lm2):
        """ Append linemodel to current linemodel """

        dd = lm.data[-1, 0:2] - lm2.data[0, 0:2]
        if np.any(np.abs(dd) > 5):
            raise Exception('cannot append non-connected linemodels')
        lm.data = np.vstack((lm.data, lm2.data))
        return

    def reverse(self):
        lm = linemodel()
        lm.data = copy.copy(self.data)
        lm.data[::-1, ...] = lm.data
        lm.data[:, 2] += np.pi
        return lm

    def traceBack(lm):
        """ Return reversed object of a line model """
        lmb = linemodel(im=ims)
        pl, pg = lm.getPointLG(0)
        pl[1] += np.pi
        pl[0] = -pl[0]
        lmb.addPointLG(pl, pg)
        return lmb

    def nPoints(self):
        return self.data.shape[0]

    def setPoint(self, ii, dx):
        if lm.nPoints() == ii:
            self.addPoint(dx)
        else:
            self.data[ii, :] = dx
            # self.cidx=self.cidx+1

    def selectPoints(self, idx):
        self.data = self.data[idx, ...]
        self.cidx = self.nPoints() - 1

    def addPoint(self, dx):
        self.data = np.vstack((self.data, dx.flat))
        self.cidx = self.cidx + 1

    def addPointLG(self, pl, pg, score=-1):
        self.addPoint(lg2data(pl, pg, score=score))

        #self.plocal=np.vstack( (self.plocal, pl.flat ))
        #self.pglobal=np.vstack( (self.pglobal, pg.flat))
        #self.score=np.vstack( (self.score, score))
        self.cidx = self.cidx + 1

    def getPointX(self, idx):
        pt = l2g(self.plocal[idx].copy(), self.pglobal[idx].copy(), dy=0)
        return pt

    def getPointLG(self, idx):
        d = self.getPoint(idx)
        pl = np.array([0, 0, d[3], d[4]])
        pg = d[0:3]
        return pl, pg

    def getPoint(self, idx):
        return self.data[idx].copy()

    def getWidth(self):
        return self.data[:, 3]

    def getHeight(self):
        return self.data[:, 4]

    def currentIndex(self):
        return self.nPoints() - 1

    def getScore(self):
        return self.data[:, 5]

    def getLength(self):
        z = self.getXY()

        z = np.linalg.norm(np.diff(z, axis=0), axis=1)

        return np.hstack((np.zeros((1,)), z.cumsum()))

    def predictPoint(lm, stepsize=6, ii=None):
        if ii is None:
            ii = lm.nPoints() - 1
        plocal, pglobal = lm.getPointLG(ii)

        pp = l2g(plocal, pglobal, dy=stepsize)

        # pglobal[0:2] += ( rot2D(pglobal[2])*np.array( [[plocal[0]], [stepsize]] ) ).flat      # update position
        # pglobal[2]+=plocal[1]    # update angle
        plocal[0:2] = 0   # reset
        return plocal, pp

    def distance(lm, p):
        """ Return distance to shapely Point """
        p = shapely.geometry.Point(p)
        if lm.nPoints() > 1:
            L = shapely.geometry.LineString(lm.getXY())
        else:
            L = shapely.geometry.Point(lm.getXY().flatten())
        d = p.distance(L)
        return d

    def imagePoly(lm):
        reg = np.array([[0, 0], lm.im.shape]).T
        pp = pmatlab.region2poly(reg)
        return pp

    def draw(lm, fig=30):
        fig = plt.figure(fig)
        plt.clf()
        ax = fig.add_subplot(111)

        if lm.im is None:
            pp = None
        else:
            pp = lm.imagePoly()
            p = Polygon(pp.T)

        # plt.plot(pp[0,:],pp[1,:],'.-b')
        #plot_coords(ax, p.exterior )
        COLOR = {True:  '#6699cc', False: '#ff3333'}

        def v_color(ob):
            return COLOR[ob.is_valid]

        if not lm.im is None:
            plt.imshow(lm.im)
            ob = p.exterior
            x, y = ob.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)
        else:
            ax.invert_yaxis()
            # pm=mplPolygon(pp.T)

            #patch = PolygonPatch(pm, facecolor=v_color(pm), edgecolor=v_color(pm), alpha=0.5, zorder=2)
            # ax.add_patch(patch)

        plotPoints(lm.getXY().T, '.r')
        pmatlab.enlargelims()

    def drawLine(lm,  marker='.', color='r'):
        pmatlab.plotPoints(lm.getXY().T, marker, color=color)

    def drawLineEndpoints(lm,  marker='.', color='r', markersize=18):
        plotPoints(
            lm.getXY()[0].reshape((2, 1)), marker, color=color, markersize=18)
        plotPoints(
            lm.getXY()[-1].reshape((2, 1)), marker, color=color, markersize=18)

    def drawCrossSection(self, idx, fig=101, samplesize=29):
        pp = self.getPoint(idx)
        patch = getpatch(self.im, pp, samplesize, fig=None)
        plocal, pglobal = self.getPointLG(idx)
        plocal[0] = 0
        plocal[1] = 0

        plt.figure(fig)
        plt.clf()
        plotCrosssectionModel(patch, plocal, pglobal)
        plt.title('Cross section', fontsize=15)

    def plotWH(lm, fig=300):
        pfig = plt.figure(fig)
        plt.clf()
        ll = lm.getLength()
        ww = lm.getWidth()
        hh = lm.getHeight()
        plt.subplot(1, 2, 1)
        ax1 = plt.gca()
        plt.plot(ll, ww, '.b')
        plt.xlabel('Arc length')
        plt.title('Width')
        #ax2 = ax1.twinx()
        plt.subplot(1, 2, 2)
        plt.plot(ll, hh, '.r')
        plt.xlabel('Arc length')
        plt.ylabel('Height')
        plt.title('Height')
        plt.show()

    def getXY(self, idx=None):
        if idx is None:
            return self.data[:, 0:2]
            # nn=self.nPoints()
            #xxyy=np.zeros( ( nn, 2)  )
            # for ii in range(nn):
            #    xy=self.getXY(ii)
            #    xxyy[ii,:]=xy
            # return xxyy
        #plocal, pglobal=self.getPoint(idx)

        #pp=l2g(plocal, pglobal)
        return self.data[idx, 0:2]
        # return pp[0:2]


class mynet:
    g = None
    r = []
    im = None

    def __init__(self, image=None):
        self.g = igraph.Graph()
        self.im = image
        return

    def __repr__(self):
        return 'mynet: graph %d vertices, %d edges' % (self.g.vcount(), self.g.ecount())

    def getEdgeVertices(self, ei):
        return self.g.get_edgelist()[ei]

    def cycleBasis(net):
        """ Return cycle basis for network """
        # https://networkx.github.io/documentation/latest/reference/generated/networkx.algorithms.cycles.simple_cycles.html#networkx.algorithms.cycles.simple_cycles

        G = nx.Graph()
        edges = net.g.get_edgelist()

        for e in edges:
            G.add_edge(e[0], e[1])

        # cc=nx.simple_cycles(G)
        cc = nx.cycle_basis(G)
        return cc

    def addvertex(self, xy):
        self.g.add_vertex()
        idx = self.g.vcount() - 1
        self.g.vs[idx]['xy'] = xy
        return self.g.vcount() - 1

    def add_edge(self, v1, v2, edata=None):
        self.g.add_edge(v1, v2)
        self.g.es[self.g.ecount() - 1]['linedata'] = edata
        return

    def getEdgeId(net, v1, v2):
        ei = net.g.get_eid(v1, v2)
        return ei

    def drawEdge(net, ei, *args, **kwargs):
        edata = net.g.es[ei]['linedata']
        xy = net.g.vs['xy']
        if edata is None:
            p, q = xy[e[0]], xy[e[1]]
            zz = np.array([p, q])
            plt.plot(zz[:, 0], zz[:, 1], '-b', *args, **kwargs)
        else:
            zz = edata.getXY()
            plt.plot(zz[:, 0], zz[:, 1], '-b', *args, **kwargs)

    def draw(net, showim=False, fig=100):
        if showim:
            showIm(self.im, fig=100)

        for xy in net.g.vs['xy']:
            plt.plot(xy[0], xy[1], '.r', markersize=15)
        xy = net.g.vs['xy']
        for ei, e in enumerate(net.g.get_edgelist()):
            net.drawEdge(ei)


def addOuter(net, ims):
    """ Add outer boundaries to net structure """
    sz = ims.shape
    net.addvertex([0, 0])
    net.addvertex([sz[1] - 1, 0])
    net.add_edge(0, 1)
    net.addvertex([sz[1] - 1, sz[0] - 1])
    net.add_edge(1, 2)
    net.addvertex([0, sz[0] - 1])
    net.add_edge(2, 3)
    net.g.add_edge(3, 0)

    el = net.g.get_edgelist()
    xx = net.g.vs['xy']
    for ii in range(net.g.ecount()):
        e = el[ii]
        ww = np.array([xx[e[0]], xx[e[1]]])
        lm = linemodel()
        # fixme: set other props
        lm.data = np.hstack((ww, np.zeros((2, 4))))
        net.g.es[ii]['linedata'] = lm


def polygonCentroids(cc, xy):
    rp = np.zeros((len(cc), 2))
    for ix, c in enumerate(cc):
        # print(c)
        pp = shapely.geometry.Polygon(xy[c, :])
        rp[ix] = np.array(pp.centroid)
    return rp

#%%


def getLinePolygon(net, c, verbose=1):
    al = net.g.get_adjlist()
    ael = net.g.get_adjedgelist()

    vertices = np.array([net.g.vs['xy'][ii] for ii in c])

    ed = []
    for kk in range(len(c)):
        ee = (c[kk], c[(kk + 1) % len(c)])

        ei = ael[ee[0]][al[ee[0]].index(ee[1])]
        if verbose:
            print('getLinePolygon: edge: %d -> %d' % (ee[0], ee[1]))
        # print(ee)
        # print(elist[ei])

        q1 = np.linalg.norm(vertices[kk] - net.g.es[ei]['linedata'].getXY(0))
        q2 = np.linalg.norm(vertices[kk] - net.g.es[ei]['linedata'].getXY(-1))
        if verbose:
            print('   distances %.1f %.1f ' % (q1, q2))
        if q1 > q2:
            lmx = copy.copy(net.g.es[ei]['linedata']).reverse()
            lmx = linemodel()
            lmx.data = copy.copy(net.g.es[ei]['linedata'].data)
            lmx = lmx.reverse()
            #ed.append( lmx )
        else:
            lmx = net.g.es[ei]['linedata']
        ed.append(lmx)
    lm0 = linemodel()
    for lm in ed:
        lm0 = mergeLineModels(lm0, lm)
    return lm0, ed

#lm0, ed=getLinePolygon(net, cc[0])


#%%
def plotNetPatches(net, bidx=None, cc=None, drawvertices=False, drawedges=False):

    if cc is None:
        cc = net.cycleBasis()
    if bidx is None:
        bidx = range(len(cc))

    cm = pylab.get_cmap('jet')
    mycmap = cmap_discretize(cm, N=6, m=6)

    elist = net.g.get_edgelist()

    # simple sorting
    xy = np.array(net.g.vs['xy'])

    ccx = polygonCentroids(cc, xy)
    bidx = np.argsort(ccx[:, 0])

    for ij, jj in enumerate(bidx):
        print('plotNetPatches: patch %d' % jj)
        c = cc[jj]
        print(c)
        if jj != 2:
            pass
            # continue

        ww = np.array([net.g.vs['xy'][ii] for ii in c])
        ed = []

        if drawvertices:
            wwt = ww.transpose()
            print('vertices')
            print(wwt)
            plotPoints(wwt, '.r', markersize=28)
            plotLabels(wwt)

        al = net.g.get_adjlist()
        ael = net.g.get_adjedgelist()
        for kk in range(len(c)):
            ee = (c[kk], c[(kk + 1) % len(c)])

            ei = ael[ee[0]][al[ee[0]].index(ee[1])]
            # print(ee)
            # print(elist[ei])

            q = np.linalg.norm(ww[kk] - net.g.es[ei]['linedata'].getXY(0))
            print('distance %.1f ' % q)
            if q > 10.:
                lmx = copy.copy(net.g.es[ei]['linedata']).reverse()
                lmx = linemodel()
                lmx.data = copy.copy(net.g.es[ei]['linedata'].data)
                lmx = lmx.reverse()
                #ed.append( lmx )
            else:
                lmx = net.g.es[ei]['linedata']
            ed.append(lmx)
            tmp = lmx.getXY(0).reshape((2, 1))
            #plotLabels(tmp, '%d start' % kk)
            tmp = lmx.getXY(-1).reshape((2, 1))
            tmp[1] += .03
            #plotLabels(tmp, '%d end' % kk)
            print(tmp)
            if kk > 21:
                break
        print('cycle %d: len %d' % (jj, len(ed)))

        if drawedges:
            mycmapE = cmap_discretize(cm, N=4, m=4)

            for ii, x in enumerate(ed):
                plotPoints(x.getXY().T, '-', color=mycmapE(ii), linewidth=4)

        ww = np.vstack(tuple([x.getXY() for x in ed]))

        #plotLabels(ww[-1,:].reshape( (2,1)), '%d end' % jj)
        col = mycmap(jj)

        poly = mplPolygon(ww, facecolor=mycmap(ij % mycmap.N), alpha=.3)
        plt.gca().add_patch(poly)
        plt.draw()

#%%
if 0:
    showIm(ims, fig=31)  # net.draw()
    plotNetPatches(net, bidx=[1], drawvertices=True)
    enlargelims()

    #%%
    showIm(ims, fig=40)  # net.draw()
    net.draw(fig=40)

    net.drawEdge(7, 13)


#%%


def Faces(edges, embedding, verbose=1):
    """ Method to detect faces in a planar graph

    edges: is an undirected graph as a set of undirected edges
    embedding: is a combinatorial embedding dictionary. Format: v1:[v2,v3], v2:[v1], v3:[v1] clockwise ordering of neighbors at each vertex.)
    """

    # Establish set of possible edges
    edgeset = set()
    # edges is an undirected graph as a set of undirected edges
    for edge in edges:
        edge = list(edge)
        edgeset |= set([(edge[0], edge[1]), (edge[1], edge[0])])

    # Storage for face paths
    faces = []
    path = []
    for edge in edgeset:
        path.append(edge)
        edgeset -= set([edge])
        break  # (Only one iteration)

    # Trace faces
    print('start: edgeset')
    print(edgeset)
    ii = 0
    while (len(edgeset) > 0):
        ii = ii + 1
        if verbose >= 2:
            print('Faces: len edgeset %d' % len(edgeset))
        if ii > 200:
            print('error: ii too large %d' % ii)
            break
       # print('trace ')

        neighbors = embedding[path[-1][-1]]
        if verbose >= 2:
            print('###\npath[-1]: %s ' % str(path[-1]))
            print('neighbors: %s' % str(neighbors))
        next_node = neighbors[
            (neighbors.index(path[-1][-2]) + 1) % (len(neighbors))]
        if verbose >= 2:
            print('  next_node: %s' % str(next_node))
        tup = (path[-1][-1], next_node)
        if tup == path[0]:
            faces.append(path)
            path = []
            for edge in edgeset:
                path.append(edge)
                edgeset -= set([edge])
                break  # (Only one iteration)
        else:
            path.append(tup)
            edgeset -= set([tup])
    if (len(path) != 0):
        faces.append(path)
    return iter(faces)

#%%

def polyarea(p):
    """ Return signed area of polygon

    Arguments
    ---------
        p : 2xN array or list of vertices
            vertices of polygon
    Returns
    -------
        area : float
            area of polygon
    
    >>> polyarea( [ [0,0], [1,0], [1,1], [0,2]] )
    1.5
    """
    def polysegments(p):
        if isinstance(p, list):
            return zip(p, p[1:] + [p[0]])
        else:
            return zip(p, np.vstack((p[1:], p[0:1])))
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for ((x0, y0), (x1, y1)) in polysegments(p)))

#%%
def closePolygonList(c):
    if c[0] != c[-1]:
        c.append(c[0])
    return c


def removeOuterFace(cc, xy, verbose=1):
    """ Remove the outer face in a cycle basis based on the area """
    cc = copy.deepcopy(cc)
    A = np.zeros(len(cc))
    for ii, c in enumerate(cc):
        # print(c)
        c = closePolygonList(copy.deepcopy(c))
        p = xy[c]
        A[ii] = polyarea(p)
        if verbose >= 2:
            print('removeOuterFace: %s: %.2f' % (c, A[ii]))

    ind = np.argmax(A)
    print('removeOuterFace: pop %d: %s' % (ind, cc[ind]))
    cc[ind] = None

    cc.pop(ind)

    return cc


def polygon_centroid(p):
    """ Return centroid of polygon, can also be computed with shapely """
    p = xy[c]
    A = -linetools.polyarea(p)

    X = 0
    Y = 0
    for ii in range(len(c)):
        i = c[ii]
        iip = (ii + 1) % len(c)
        ip = c[iip]
        Xa = (xy[i, 0] + xy[ip, 0]) * \
            (xy[i, 0] * xy[ip, 1] - xy[ip, 0] * xy[i, 1]) / (6 * A)
        Ya = (xy[i, 1] + xy[ip, 1]) * \
            (xy[i, 0] * xy[ip, 1] - xy[ip, 0] * xy[i, 1]) / (6 * A)
        print('ii %d: %d %d: %.2f %.2f' % (ii, i, ip, Xa, Ya))
        X += Xa
        Y += Ya
    return np.array([X, Y])


#%%

def createCross(param, samplesize, l=20, w=2.5, lsegment=10, H=100, scale=None, lines=range(4), istep=1, centermodel=True, linesegment=True, addX=True, verbose=0):
    """ Create a cross model

    The parameters are [x, y, width, alpha_1, ..., alpha_4, [rotation of polarization line] ]
    With the optional parameters psi (angle of transition line)
    
    param : array of floats
        parameters of the model
    l, w, lsegment : float
        parameters of the model
        

    """
    aa = param[3:7]
    
    psi=np.pi/4
    if len(param)>7:
        psi=param[7]
        if verbose:
            print('createCross: psi set to %.1f [def]' % np.rad2deg(psi))
    # aa=[0,0,0,0]

    if samplesize is None:
        cc = param[0:2].reshape((2, 1))
        if scale is None:
            scale = 50
    else:
        if scale is None:
            scale = np.mean(samplesize)
        samplesize = np.array(samplesize).flatten()
        if centermodel:
            cc = np.array(samplesize).reshape((2, 1)) / 2 - .5
        else:
            cc=np.array(param[0:2]/istep).reshape( (2,1))
            
    lp = cc + pmatlab.rot2D(psi+ np.pi / 2).dot(np.array([[param[2]/istep], [0]]))
    hp = cc - pmatlab.rot2D(psi+ np.pi / 2).dot(np.array([[param[2]/istep], [0]]))

    op = np.zeros((5, 2))
    opr = np.zeros((4, 2))
    ip = np.zeros((5, 2))
    for ii, a in enumerate(aa):
        if ii == 0 or ii == 1:
            ip[ii].flat = lp
        else:
            ip[ii].flat = hp
        #a += -np.pi/4
        op[ii] = ip[ii] + ((scale) * pmatlab.rot2D(a).dot(np.array([[1], [0]])) ).flat
        opr[ii] = ip[ii] + ((lsegment/istep) * pmatlab.rot2D(a).dot(np.array([[1], [0]])) ).flat

    
    if samplesize is not None:
        modelpatch = np.zeros([samplesize.flat[1], samplesize.flat[0]])

        for ii in lines:
            if linesegment:
                x0=ip[ii]
                x1x=np.array(x0+lsegment/istep*(pmatlab.rot2D(aa[ii]).dot( np.array( [[1],[0]]) ) ).T).flatten()
                
                lineSegment(modelpatch, x0=x0, x1=x1x, w=w/istep, l=None, H=H)
            else:
                semiLine(modelpatch, x0=ip[ii], theta=aa[ii], w=w/istep, l=l/istep, H=H)
        if addX:
            lx=np.linalg.norm(hp-lp, ord=2)
            #print(hp); print(lp)
            #print(lx)
            lineSegment(modelpatch, x0=np.array(hp.reshape( (2,1))), x1=np.array(lp.reshape( (2,1))), w=w/istep, l=lx, H=H)
            
    else:
        modelpatch = None
    modelpatch=np.minimum(modelpatch, H)
    return modelpatch, (cc, lp, hp, ip, opr, w, H, lsegment)

#%%
def lineSegment(im, x0, x1=None, theta=None, w=2, l=12, H=200, ml=0):
    """ Plot half-line into image 

    >>> lineSegment(np.zeros( (160,240)), [60,40], [70,40], w=10, l=60)
    >>> lineSegment(np.zeros( (160,240)), [60,40], theta=np.deg2rad(20), w=10, l=60)

    """
    x0=np.array(x0).flatten()
    if x1 is None:
        thetar = -theta
        dx=l
    else:
        x1=np.array(x1).flatten()
        theta = x0-x1
        theta=np.arctan2( theta[1], theta[0])   +np.pi 
        thetar = -theta
    
        dx=np.linalg.norm(x0-x1)
    xx0, yy0 = np.meshgrid(np.arange(im.shape[1])-x0[0], np.arange(im.shape[0])-x0[1])
    xx0=xx0.astype(np.float32)
    yy0=yy0.astype(np.float32)

    xxr = math.cos(thetar) * xx0 - math.sin(thetar) * yy0
    yyr = math.sin(thetar) * xx0 + math.cos(thetar) * yy0
    yyr = pmatlab.signedmin(yyr, w / 2.)

    data = H * np.cos( (np.pi/w) * yyr ) * (pmatlab.smoothstep(xxr, ml, 4)) *(1-pmatlab.smoothstep(xxr, dx, 4))

    im += data
    return im

#%%    
def semiLine(im, x0, theta, w, l, H=200, dohalf=True):
    """ Plot half-line into image 

    >>> im=semiLine(np.zeros( (160,240)), [60,40], theta=np.deg2rad(20), w=10, l=60)
    >>> plt.imshow(im)

    """
    thetar = -theta
    #xx, yy = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    #xx0 = xx - x0[0]
    #yy0 = yy - x0[1]
    xx0, yy0 = np.meshgrid(np.arange(im.shape[1])-x0[0], np.arange(im.shape[0])-x0[1])
    xx0=xx0.astype(np.float32)
    yy0=yy0.astype(np.float32)

    xxr = math.cos(thetar) * xx0 - math.sin(thetar) * yy0
    yyr = math.sin(thetar) * xx0 + math.cos(thetar) * yy0
    yyr = pmatlab.signedmin(yyr, w / 2.)

    data = H * np.cos( (np.pi/w) * yyr )
    if dohalf:
        data *= (pmatlab.smoothstep(xxr, 0, 10))

    im += data
    return im


def createH_old(samplesize, pp):
    c = (np.array(samplesize) / 2. - .5).reshape((2, 1))
    cc = pp[0:2].reshape((2, 1))
    theta = pp[2]
    # H=pg_transl2H(1*c)*pg_rotation2H(rot2D(theta))*pg_transl2H(-cc) # image to patch
    # image to patch, written out
    H = np.matrix(np.eye((3)))
    H[0, :] = [math.cos(theta), -math.sin(theta), -
               math.cos(theta) * cc[0] + math.sin(theta) * cc[1] + c[0]]
    H[1, :] = [math.sin(theta), math.cos(
        theta), -math.sin(theta) * cc[0] - math.cos(theta) * cc[1] + c[1]]
    return H

@pmatlab.static_var("HH", np.matrix(np.eye(3)))
def createH(samplesize, pp, scale=1):
    """ Create H matrix to transform image to patch coordinates """
    cx = (np.array(samplesize) / 2. - .5).reshape((2, 1))
    cc = pp[0:2].reshape((2, 1))
    theta = 0  # pp[2]
    # image to patch, written out
    H = createH.HH.copy()
    c = math.cos(theta)
    s = math.sin(theta)
    
    H.itemset(0, scale*c)
    H.itemset(1, scale*-s)
    H.itemset(2, scale*(-c * cc[0] + s * cc[1]) + cx[0] )
    H.itemset(3, scale*s)
    H.itemset(4, scale*c)
    H.itemset(5, scale*(-s * cc[0] - c * cc[1]) + cx[1] )
    #H[0,:] = [c, -s, -c*cc[0]+s*cc[1] + cx[0] ]
    #H[1,:] = [s, c, -s*cc[0]-c*cc[1] + cx[1]]
    return H

#%%
def findCrossTemplate(imx, ksize=31, fig=None, istep=2, verbose=1, widthmv=6, lenmv=20., sepmv=2.0, dy=5):
    """ Find crosses in image using template match
    Arguments
    ---------
        istep : float
            sampling rate in mV/pixel
        widthmv, lenmv, sepmv : float
            parameters of the cross model
    Returns
    -------
        ptsim : array
            fitted points
        rr : numpy array
            response of the filter
        results : dict
            more results

            
 """
    samplesize = np.array([ksize, ksize + dy])
    param = [None, None, sepmv / istep, 3 * np.pi /
             8, -7 * np.pi / 8, 11 * np.pi / 8, np.pi / 8]
    modelpatch, cdata = createCross(param, samplesize, w=widthmv/istep, l=lenmv/istep, lsegment=lenmv/istep, H=100)

    imtmp = pmatlab.setregion(scaleImage(imx), scaleImage(modelpatch), [0, 0])

    #rr=cv2.matchTemplate(imx, modelpatch.astype(np.float32), method=cv2.TM_SQDIFF)
    rr = cv2.matchTemplate(scaleImage(imx), scaleImage(
        modelpatch.astype(imx.dtype)), method=cv2.TM_CCORR_NORMED)
    #rr=cv2.matchTemplate(scaleImage(imx), scaleImage(modelpatch.astype(np.float32)), method=cv2.TM_SQDIFF); rr=-rr
    rr = smoothImage(rr)

    thr = .65 * rr.max() + .35 * rr.mean()
    #pts = pmatlab.detect_local_minima(-rr, thr=-thr)
    pts = localMaxima(rr, thr=thr, radius=10/istep)
    pts = np.array(pts)
    pts = pts[[1, 0], :]

    ptsim = pts + ((samplesize - 1.) / 2).reshape((2, 1))

    if verbose:
        print('findCrossTemplate: threshold: %.1f, %d local maxima' % (thr, pts.shape[1]) )

    if fig is not None:
        showIm(imtmp, fig=fig)
        #pt+=samplesize[::-1].reshape( (2,1))
        plt.plot(ptsim[0], ptsim[1], '.m', markersize=22)
        showIm(rr, fig=fig+1)
        plt.colorbar()
        plt.title('Template and image')
        plt.plot(pts[0], pts[1], '.m', markersize=22)
        plt.title('Template match')

        tilefigs([fig, fig + 1])

    return ptsim, rr, dict({'modelpatch': modelpatch})

@pmatlab.static_var("scaling0", np.diag([1.,1,1]) )
def evaluateCross(param, im, verbose=0, fig=None, istep=1, istepmodel=1, linewidth=2, usemask=False):
    """ Calculate cross matching score
    
    Returns
    -------
        cost, patch, cdata, tuple
        
    """
    samplesize=[int(im.shape[1]*istep/istepmodel), int(im.shape[0]*istep/istepmodel)]
    param=np.array(param)
    aa=param[3:]
        
    #pp=np.array(param[0:2]) # /istep
    #H=createH(samplesize, pp, scale=istep/istepmodel)
    #H=pmatlab.pg_scaling([istep/istepmodel, istep/istepmodel])
    H=evaluateCross.scaling0.copy()
    H[0,0]=istep/istepmodel
    H[1,1]=istep/istepmodel
    
    #patch=linetools.sampleImage(im, pp, samplesize, fig=11, clearfig=True, nrsub=1)
    dsize=(samplesize[0], samplesize[1])
    patch=cv2.warpPerspective(im.astype(np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)
  
    if verbose:
        print('evaluateCross: patch shape %s'  % (patch.shape,))
    modelpatch, cdata=createCross(param, samplesize, centermodel=False, istep=istepmodel, verbose=verbose>=2)
    (cc,lp,hp,ip,op,_,_,_)=cdata

    dd=patch-modelpatch

    if usemask:
        # near model mask
        #mask = (modelpatch>1).astype(int)
        
        # distance centre mask
        imtmp=10+0*modelpatch.copy()
        imtmp[int(imtmp.shape[1]/2),int(imtmp.shape[0]/2)]=0
        mask=scipy.ndimage.distance_transform_edt(imtmp)
        mask=1-.75*mask/mask.max()
    

        dd=dd*mask
        
    #dd=(dd*cf)
    cost=np.linalg.norm(dd)    
    
    # area of intersection
    rr=np.array([[0.,im.shape[1]],[0,im.shape[0]]])
    ppx=pmatlab.region2poly(np.array([[0,samplesize[0]],[0.,samplesize[1]]]) )
    ppimage=pmatlab.region2poly(rr)
    pppatch=pmatlab.projectiveTransformation(H, ppimage)        
    ppi=pmatlab.polyintersect(ppx.T, pppatch.T).T
    A=pmatlab.polyarea(ppi.T)    
    A0=pmatlab.polyarea(ppx.T)
    
    # special rules
    if A/A0<.85:
        cost+=4000
    if aa[0]<0 or aa[0]>np.pi/2-np.deg2rad(5):
        cost+=10000
    if aa[1]<np.pi or aa[1]>3*np.pi/2:
        pass
        cost+=10000
    if aa[2]<np.pi or aa[2]>3*np.pi/2:
        pass
        cost+=10000
    if aa[3]<0 or aa[3]>np.pi/2:
        pass
        cost+=10000

    if pmatlab.angleDiff(aa[0], aa[1]) < np.deg2rad(30):
        cost+=1000
    if pmatlab.angleDiff(aa[2], aa[3]) < np.deg2rad(30):
        cost+=1000

    if pmatlab.angleDiffOri(aa[0], aa[2]) > np.deg2rad(45):
        cost+=10000
    if pmatlab.angleDiffOri(aa[1], aa[3])  > np.deg2rad(45):
        cost+=10000
    if param[2]<0:
        cost+=10000

    if np.abs( param[2] ) > 7.:
            #print('x deviation!')
            cost+=10000

    if np.abs( param[2]-10 ) >  8:
            #print('x deviation!')
            cost+=10000
        
    if len(param)>7:
        if np.abs( angleDiff(param[7], np.pi/4) ) >  np.deg2rad(30):
            #print('psi deviation!')
            cost+=10000
        
    
    if not fig is None:
        
            #lsegment=cdata[7]
            showIm(patch, fig=fig); plt.title('Image patch: cost %.1f: istep %.2f' % (cost, istepmodel))
            pmatlab.addfigurecopy(fig=fig)
            plt.plot( [float(lp[0]), float(hp[0])], [float(lp[1]), float(hp[1]) ], '.--m', linewidth=linewidth, markersize=10, label='transition line')
            plt.plot( cc[0], cc[0], '.m', markersize=12)
            for ii in range(4):
                if ii==0:
                    lbl = 'electron line'
                else:
                    lbl = None                    
                plt.plot( [op[ii,0], ip[ii,0]], [op[ii,1],ip[ii,1]], '.-', linewidth=linewidth, color=[0,.7,0], label=lbl)
                pmatlab.plotLabels( np.array( (op[ii,:]+ ip[ii,:])/2 ).reshape( (2,-1) ), '%d' % ii )
            showIm(modelpatch, fig=fig+1); plt.title('Model patch: cost %.1f' % cost)
            showIm(np.abs(dd), fig=fig+2); plt.title('diff patch: cost %.1f' % cost); plt.colorbar()
            plt.show()
            #plt.pause(1e-6)
            #tilefigs([fig, fig+1], [2,2])
            
    if verbose:
        print('evaluateCross: cost %.4f' % cost)
        #print('evaluateCross: param %s' % (str(param), ))
    return cost, patch, cdata, (H, )
    pass

    
def evaluateCrossX(param, im, ksize, verbose=0, fig=None, istep=1, istepmodel=1, linewidth=2, usemask=False):
    """ Calculate cross matching score
    
    """
    if isinstance(ksize, list):
        samplesize=[ksize[0], ksize[1]]
    else:
        samplesize=[ksize, ksize]
        
    pp=np.array(param[0:2]) /istep
    aa=param[3:]

    H=createH(samplesize, pp, scale=istep/istepmodel)
    #patch=linetools.sampleImage(im, pp, samplesize, fig=11, clearfig=True, nrsub=1)
    dsize=(samplesize[0], samplesize[1])
    patch=cv2.warpPerspective(im.astype(np.float32), H, dsize, None, (cv2.INTER_NEAREST), cv2.BORDER_CONSTANT, -1)
  
    modelpatch, cdata=createCross(param, samplesize, istep=istepmodel, verbose=0)
    (cc,lp,hp,ip,op)=cdata

    # near model mask
    #mask = (modelpatch>1).astype(int)
    
    # distance centre mask
    imtmp=10+0*modelpatch.copy()
    imtmp[int(imtmp.shape[1]/2),int(imtmp.shape[0]/2)]=0
    mask=scipy.ndimage.distance_transform_edt(imtmp)
    mask=1-.75*mask/mask.max()


    dd=patch-modelpatch
    if usemask:
        dd=dd*mask
        
    #dd=(dd*cf)
    cost=np.linalg.norm(dd)    
    
    # area of intersection
    rr=np.array([[0.,im.shape[1]],[0,im.shape[0]]])
    ppx=pmatlab.region2poly(np.array([[0,samplesize[0]],[0.,samplesize[1]]]) )
    ppimage=pmatlab.region2poly(rr)
    pppatch=pmatlab.projectiveTransformation(H, ppimage)        
    ppi=pmatlab.polyintersect(ppx.T, pppatch.T).T
    A=pmatlab.polyarea(ppi.T)    
    A0=pmatlab.polyarea(ppx.T)
    
    # special rules
    if A/A0<.85:
        cost+=4000
    if aa[0]<0 or aa[0]>np.pi/2-np.deg2rad(5):
        cost+=10000
    if aa[1]<np.pi or aa[1]>3*np.pi/2:
        pass
        cost+=10000
    if aa[2]<np.pi or aa[2]>3*np.pi/2:
        pass
        cost+=10000
    if aa[3]<0 or aa[3]>np.pi/2:
        pass
        cost+=10000

    if pmatlab.angleDiff(aa[0], aa[1]) < np.deg2rad(30):
        cost+=1000
    if pmatlab.angleDiff(aa[2], aa[3]) < np.deg2rad(30):
        cost+=1000

    if pmatlab.angleDiffOri(aa[0], aa[2]) > np.deg2rad(45):
        cost+=10000
    if pmatlab.angleDiffOri(aa[1], aa[3])  > np.deg2rad(45):
        cost+=10000
    if param[2]<0:
        cost+=10000

    if np.abs( param[2] ) > 7.:
            #print('x deviation!')
            cost+=10000

    if np.abs( param[2]-10 ) >  8:
            #print('x deviation!')
            cost+=10000
        
    if len(param)>7:
        if np.abs( angleDiff(param[7], np.pi/4) ) >  np.deg2rad(30):
            #print('psi deviation!')
            cost+=10000
        
    
    if not fig is None:
            showIm(patch, fig=fig); plt.title('Image patch: cost %.1f: istep %.2f' % (cost, istepmodel))
            pmatlab.addfigurecopy(fig=fig)
            plt.plot( [float(lp[0]), float(hp[0])], [float(lp[1]), float(hp[1]) ], '.--m', linewidth=linewidth, markersize=10, label='transition line')
            plt.plot( cc[0], cc[0], '.m', markersize=12)
            for ii in range(4):
                if ii==0:
                    lbl = 'electron line'
                else:
                    lbl = None                    
                plt.plot( [op[ii,0], ip[ii,0]], [op[ii,1],ip[ii,1]], '.-', linewidth=linewidth, color=[0,.7,0], label=lbl)
                pmatlab.plotLabels( np.array( (op[ii,:]+ ip[ii,:])/2 ).reshape( (2,-1) ), '%d' % ii )
            showIm(modelpatch, fig=fig+1); plt.title('Model patch: cost %.1f' % cost)
            showIm(np.abs(dd), fig=fig+2); plt.title('diff patch: cost %.1f' % cost); plt.colorbar()
            plt.show()
            #plt.pause(1e-6)
            #tilefigs([fig, fig+1], [2,2])
            
    if verbose:
        print('evaluateCross: cost %.4f' % cost)
    return cost, patch, cdata, (H, )
    pass

def fitModel(param0, imx, docb=False, verbose=1, cfig=None, ksizemv=41, istep=None, istepmodel=.5, cb=None):
    """ Fit model of an anti-crossing """
    samplesize=[int(ksizemv/istepmodel), int(ksizemv/istepmodel)]

    costfun = lambda param0: evaluateCrossX(param0, imx, samplesize, fig=None, istepmodel=istepmodel, istep=istep)[0]
    costfun = lambda param0: evaluateCross(param0, imx, fig=None, istepmodel=istepmodel, istep=istep)[0]

    #costfun = lambda x0: costFunction(x0, pglobal, ims)
    vv=[]
    def fmCallback(plocal, pglobal):
        vv.append( (plocal, pglobal)) 
    if cfig is not None:
        cb=lambda x: fmCallback(x, None)
        #cb= lambda param0: evaluateCross(param0, imx, ksize, fig=cfig)[0]
        #cb = lambda param0: print('fitModel: cost %.3f' % evaluateCross(param0, imx, ksize, fig=None)[0] )

    if 1:
        # simple brute force
        ranges=list( [ slice(x,x+.1,1) for x in param0] )
        for ii in range(2):
            ranges[ii]=slice( param0[ii]-13, param0[ii]+13, 1) 
        ranges=tuple(ranges)    
        res = scipy.optimize.brute(costfun, ranges)
        paramy=res
    else:
        paramy=param0
    res = scipy.optimize.minimize(costfun, paramy, method='nelder-mead',  options={'maxiter': 1200, 'maxfev': 101400, 'xtol': 1e-8, 'disp': verbose>=2}, callback=cb)
    #res = scipy.optimize.minimize(costfun, res.x, method='Powell',  options={'maxiter': 3000, 'maxfev': 101400, 'xtol': 1e-8, 'disp': verbose>=2}, callback=cb)
    
    #for kk in range(1000):
    #    xxx=evaluateCross(param0, imx, fig=None, istepmodel=istepmodel, istep=istep)[0]
        
    if verbose:
        print('fitModel: score %.2f -> %.2f' % (costfun(param0), res.fun) )
    return res  


def Vtrace(cdata, param, fig=None):
    """ Calculate position of next V-trace from fitted model """
    cc=cdata[0]
    psi=param[-1]
    
    q=np.array([10,0]).reshape( (2,1))
    p1=cc+pmatlab.rot2D(psi).dot(q)
    p2=cc+pmatlab.rot2D(np.pi+psi).dot(q)
    pp=np.array(np.hstack( (p1,cc,p2)))
    pp=np.array(np.hstack( (p1,p2)))
    if fig is not None:
        plt.figure(25)

        pmatlab.plotPoints(pp, '--k', markersize=20, linewidth=3, label='scan line')
        pmatlab.plotPoints(pp, '.y', markersize=20)
        try:
            plt.legend(numpoints=1, fontsize=14, loc=0)
        except:
            # fontsize does not work with older matplotlib versions...
            pass
    psi,slope=calcSlope(pp)
    return pp, cc, slope

def calcSlope(pp):
    q=-np.diff(pp, axis=1)
    psi = math.atan2(q[1], q[0])
    slope=q[1]/q[0]

    return psi, slope    
    
    
#%%

#%%
@pmatlab.static_var("scaling0", np.diag([1.,1,1]) )
def costFunctionLine(pp, imx, istep, maxshift=12, verbose=0, fig=None, maxangle=np.deg2rad(70), ksizemv=12, dthr=8, dwidth=3, alldata=None, px=None):
    """ Cost function for line fitting
    
        pp (list or array): line parameters
        imx (numpy array): image to fit to
        istep (float)
        px (array): translational offset to operate from
    
    """
    istepmodel=.5
    samplesize=[int(imx.shape[1]*istep/istepmodel), int(imx.shape[0]*istep/istepmodel)]

    LW=2 # [mV]
    LL=15 # [mV]
    
    H=costFunctionLine.scaling0.copy()
    H[0,0]=istep/istepmodel
    H[1,1]=istep/istepmodel
    
    #patch=linetools.sampleImage(im, pp, samplesize, fig=11, clearfig=True, nrsub=1)
    dsize=(samplesize[0], samplesize[1])
    patch=cv2.warpPerspective(imx.astype(np.float32), H, dsize, None, (cv2.INTER_LINEAR), cv2.BORDER_CONSTANT, -1)
    pm0=np.array(pp[0:2]).reshape( (1,2))/istepmodel # [pixel]
    if px is None:
        pxpatch=[patch.shape[1]/2, patch.shape[0]/2 ]
    else:
        pxpatch = (float(istep)/istepmodel) * np.array(px)
    pm=pm0+pxpatch
    #modelpatch, cdata=createCross(param, samplesize, centermodel=False, istep=istepmodel, verbose=0)

    lowv=np.percentile(imx, 1)
    highv=np.percentile(imx, 95)
    theta=pp[2]
    
    if verbose:
            print('costFunctionLine: sample line patch: lowv %.1f, highv %.1f' % (lowv, highv))
            #print(px)
    linepatch=lowv+np.zeros( (samplesize[1], samplesize[0] )  )  
    lineSegment(linepatch, pm, theta=pp[2], w=LW/istepmodel, l=LL/istepmodel, H=highv-lowv, ml=-6/istepmodel)
    #plt.figure(99); plt.clf(); plt.imshow(lineseg, interpolation='nearest'); plt.colorbar()
    #plt.figure(99); plt.clf(); plt.imshow(linepatch-lineseg, interpolation='nearest'); plt.colorbar()
    #plt.figure(99); plt.clf(); plt.imshow(linepatch, interpolation='nearest'); plt.colorbar()
    dd=patch-(linepatch)
    cost=np.linalg.norm(dd)
    cost0=cost
    
    if 1:
        ddx0=np.linalg.norm(pm0) # [pixel]
        ddx=np.linalg.norm(pm0) # [pixel]
        if verbose:
            print('costFunctionLine: calculate additonal costs: dist %.1f [mV]' % (ddx*istepmodel) )
        
        ddx=pmatlab.smoothstep(ddx, dthr/istepmodel,dwidth/istepmodel)
        if verbose>=2: 
            print('  ddx: %.3f, thr %.3f'  % (ddx, dthr/istepmodel))
        cost+=100000*ddx
    #cost = sLimits(cost, plocal, pm, maxshift, maxangle)
    
    if fig is not None:
        pmatlab.cfigure(fig); plt.clf()
        plt.imshow(patch, interpolation='nearest'); plt.title('patch: cost %.2f, dist %.1f' % (cost, ddx0*istep ))
        plt.colorbar()
        pm=pm.flatten()
        #plt.plot(pm0.flatten()[0], pm0.flatten()[1], 'dk', markersize=12, label='initial starting point?')
        plt.plot(pm[0], pm[1], '.g', markersize=24, label='fitted point')
        plt.plot(pxpatch[0], pxpatch[1], '.m', markersize=18, label='offset for parameters')


        qq=np.array(pm.reshape(2,1)+ (LL/istepmodel)*pmatlab.rot2D(theta).dot(np.array([[1,-1],[0,0]])))
        
        plt.plot(qq[0,:], qq[1,:], '--k', markersize=24, linewidth=2)

        #print(pm)
        plt.axis('image')
#       plt.colorbar()
        
        pmatlab.cfigure(fig+1); plt.clf()
        plt.imshow(linepatch, interpolation='nearest'); plt.title('line patch')
        plt.plot(px[0], px[1], '.m', markersize=24)
        plt.axis('image')
        plt.colorbar()
        pmatlab.tilefigs([fig,fig+1])
        
        if verbose>=2:
            pmatlab.cfigure(fig+2); plt.clf()
            xx=np.arange(0, 20, .1)
            xxstep=istepmodel*pmatlab.smoothstep(xx/istepmodel, dthr/istepmodel,(1/dwidth)/istepmodel)
            plt.plot(xx, xxstep, '.-b' , label='distance step')
            plt.xlabel('Distance [mV]')
            plt.legend()
            
    if verbose:
        print('costFucntion: cost: base %.2f -> final %.2f' % (cost0, cost))
        if verbose>=2:
            ww=np.abs(dd).mean(axis=0)
    
            print('costFunction: dd %s ' % ww)

    return cost    
    
    
if __name__=='__main__':
        res.x=res.x+[.0,.0,.15]
        pp=res.x
        verbose=2
        c=costFunctionLine(pp, imx, istep, verbose=verbose, fig=fig, px=px); plt.figure(fig); plt.xlabel(cgate); plt.ylabel(igate); #plt.close(fig+1)
        plt.colorbar()
        
#%%

import scipy
from scipy.optimize import minimize
from qtt.deprecated.linetools import costFunctionLine
figl=100



def fitLine(im, param0=None, fig=None):
    """ Fit a line local to a model """
    if param0 is None:        
        param0=[0,0,.5*np.pi] # x,y,theta, 
    istep=.5
    verbose=1
    cb=None
    imx=-np.array(alldata.diff_dir_xy)
    px=[imx.shape[1]/2, imx.shape[0]/2 ]

    #qtt.data.dataset2Dmetadata(alldata)

    costfun=lambda x : costFunctionLine(x, imx, istep, verbose=0, px=px, dthr=7, dwidth=4)
    res = minimize(costfun, param0, method='powell',  options={'maxiter': 3000, 'maxfev': 101400, 'xtol': 1e-8, 'disp': verbose>=2}, callback=cb)
    
    cgate=alldata.diff_dir_xy.set_arrays[1].name; igate=alldata.diff_dir_xy.set_arrays[0].name
    #res.x=param0
    c=costFunctionLine(res.x, imx, istep, verbose=1, fig=figl, px=px); plt.figure(figl); plt.xlabel(cgate); plt.ylabel(igate);
    
                
if __name__=='__main__':
        param0=[0,0,.5*np.pi] # x,y,theta, 
        
        fitdata = fitLine(im, param0=None, fig=None)
        
