#import qtpy
#print(qtpy.API_NAME)
import copy

import numpy as np
import scipy
import matplotlib
import sys, os
import logging
import qcodes

# explicit import
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot

import qtt.data
from qtt.data import loadExperimentData
from qtt.algorithms import onedotGetBalance
from qtt.algorithms.onedot import onedotGetBalanceFine
from qtt.scans import pinchoffFilename
from qtt.data import load_data, show2D

# should be removed later
#from pmatlab import tilefigs

from qtt import pmatlab
from qtt.pmatlab import plotPoints, tilefigs
import matplotlib.pyplot as plt

#%%

def getTwoDotValues(td, ods, basevaluestd=dict({}), verbose=1):
    """ Return settings for a 2-dot, based on one-dot settings """
    # basevalues=dict()

    if verbose >= 2:
        print('getTwoDotValues: start: basevalues td: ')
        print(basevaluestd)

    bpleft = getODbalancepoint(ods[0])
    bpright = getODbalancepoint(ods[1])

    tddata=dict()
    
    if td['gates'][2] == td['gates'][3]:
        ggg = [None] * 3
        ggL = ods[0]['gates']
        ggR = ods[1]['gates']
        
        p1 = basevaluestd[ggL[1]]
        p2 = basevaluestd[ggR[1]]
        
        val = [bpleft[1, 0], p1, bpleft[0, 0]]
        leftval = val[0]
        ggg[0] = ggL[0]
        ggg[1] = ggL[2]
        for g, v in zip(ggL, val):
            basevaluestd[g] = v
        val = [bpright[1, 0], p2, bpright[0, 0]]
        rightval = val[2]
        for g, v in zip(ggR, val):
            basevaluestd[g] = v
        ggg[2] = ggR[2]


        g = ods[0]['gates'][2]
        v1 = bpleft[0, 0]
        v2 = bpright[1, 0]
        v = (v1 + v2) / 2
        if verbose:
            print(
                'getTwoDotValues: one-dots share a gate: %s: %.1f, %.1f [mV]' % (g, v1, v2))
        basevaluestd[g] = float(v)

        tddata['gates']=[ggg[0], ggL[1],ggg[1], ggR[1],ggg[2]]
        tddata['gatevaluesleft']=[bpleft[1, 0], basevaluestd[ggL[1]], bpleft[0, 0]]
        tddata['gatevaluesright']=[ bpright[1, 0], basevaluestd[ggR[1]], bpright[0, 0]]

        fac = .10
        fac = 0
        facplunger=.1
        
        cc = [-rightval * fac, -facplunger*rightval, -(leftval + rightval) * fac / 2, -facplunger*leftval, -leftval * fac]
        print('getTwoDotValues: one-dots share a gate: %s: compensate %s' %
              (str(tddata['gates']), str(cc)))
        for ii,g in enumerate(tddata['gates']):
            basevaluestd[g] += float(cc[ii])
            # basevalues[ggg[ii]]+=10

        tddata['v']=[v1,v2,v]
        tddata['gatecorrection']=cc
        tddata['gatevalues']=[basevaluestd[gx] for gx in tddata['gates']]
        tddata['ods']=ods
    else:
        gg = ods[0]['gates']
        val = [ods[0]['balancepoint'][1, 0], 0, ods[0]['balancepoint'][0, 0]]
        for g, v in zip(gg, val):
            basevaluestd[g] = v
        gg = ods[1]['gates']
        val = [ods[1]['balancepoint'][1, 0], 0, ods[0]['balancepoint'][0, 0]]
        for g, v in zip(gg, val):
            basevaluestd[g] = float(v)

    # make sure all values are nice floats (not scalar numpy arrays)
    for k in basevaluestd:
         basevaluestd[k] = float(basevaluestd[k])

    if verbose >= 2:
        print('getTwoDotValues: return basevalues: ')
        print(basevaluestd)

    return basevaluestd, tddata


#%%

def showODresults(od, dd2d, fig=200, imx=None, ww=None):
    ''' Show results of a 1-dot fit ? '''
    balancepoint = od['balancepoint']
    ptv0 = od['balancepoint0']
    if not fig:
        return

    tmp = show2D(dd2d, fig=fig)

    _ = show2D(dd2d, fig=fig + 1)
    plt.title('result')
    plt.axis('image')
    plotPoints(balancepoint, '.m', markersize=18)
    plotPoints(od['balancefit'], '--c')

    plotPoints(ptv0, 'or', markersize=10, mew=2.5, fillstyle='none')

    if not ww is None:
        plt.figure(fig + 2)
        plt.clf()
        plt.imshow(imx)
        plt.title('polygon')

        plt.figure(fig + 3)
        plt.clf()
        plt.imshow(imx == ww)
        plt.title('difference')
        plt.axis('image')
        #plotPoints(pt, '.m', markersize=18)

    tilefigs([fig, fig + 1, fig + 2], [3, 2])
#%%

def point_in_poly(x, y, poly):
    ''' Return true if a point is contained in a polygon '''
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def points_in_poly(points, poly_verts):
    ''' Determine whether points are contained in a polygon or not
    '''
    nn = points.shape[0]
    rr = np.zeros((nn,))
    for ii in range(nn):
        rr[ii] = point_in_poly(points[ii, 0], points[ii, 1], poly_verts)

    rr = rr.astype(np.bool)
    return rr
    
def fillPoly(im, poly_verts, color=None):
    """ Fill a polygon in an image with the specified color

    Replacement for OpenCV function cv2.fillConvexPoly

    Arugments:
        im (array)
        poly_verts (array): polygon vertices
        color (array or float)
    """
    ny, nx = im.shape[0], im.shape[1]

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((y, x)).T

    npts = poly_verts.size / 2
    poly_verts = poly_verts.reshape((npts, 2))
    poly_verts = poly_verts[:, [1, 0]]

    try:
        pp = Path(poly_verts)
        r = pp.contains_points(points)
    except:
        # slow version...
        r = points_in_poly(points, poly_verts)
        pass
    im.flatten()[r] = 1
    #grid = points_inside_poly(points, poly_verts)
    grid = r
    grid = grid.reshape((ny, nx))

    return grid
    

def getPinchvalues(od, xdir):
    """ Get pinch values from recorded data """
    gg = od['gates']
    od['pinchvalues'] = -800 * np.ones(3)
    for jj, g in enumerate(gg):
        #pp='%s-sweep-1d-%s.pickle' % (od['name'], g)
        pp = pinchoffFilename(g, od=None)
        pfile = os.path.join(xdir, pp)
        
        print('getPinchvalues: gate %s'  % g)
        dd, metadata=qtt.data.loadDataset(pfile)
        #dd = load_data(pfile)

        # print(dd.keys())

        adata = qtt.algorithms.analyseGateSweep(dd, fig=0, minthr=100, maxthr=800, verbose=0)
        od['pinchvalues'][jj] = adata['pinchvalue']
    return od

def createDoubleDotJobs(two_dots, one_dots, resultsdir, basevalues=dict(), fig=None, verbose=1):
    """ Create settings for a double-dot from scans of the individual one-dots """
    # one_dots=get_one_dots(full=1)
    xdir = os.path.join(resultsdir, 'one_dot')

    jobs = []
    for jj, td in enumerate(two_dots):
        print('\n#### analysing two-dot: %s' % str(td['gates']))

        try:
            od1 = 'dot-' + '-'.join(td['gates'][0:3])
            od1 = [x for x in one_dots if x['name'] == od1][0]
            od2 = 'dot-' + '-'.join(td['gates'][3:6])
            od2 = [x for x in one_dots if x['name'] == od2][0]
        except Exception as e:
            print('createDoubleDotJobs: no one-dot data available for %s' %
                  td['name'])
            print(e)
            continue
            pass
        
        if verbose>=2:
            print('get balance point data')
        ods = []
        try:
            for ii, od in enumerate([od1, od2]):

                dstr = '%s-sweep-2d' % (od['name'])
                dd2d = loadExperimentData(resultsdir, tag='one_dot', dstr=dstr)

                if verbose>=2:
                    print('  at getPinchvalues')

                od = getPinchvalues(od, xdir)

                if fig:
                    fign = 1000 + 100 * jj + 10 * ii
                    figm = fig + 10 * ii
                else:
                    fign = None
                    figm = None

                if verbose>=2:
                    print('  at onedotGetBalance')
                od, ptv, pt0, ims, lv, wwarea = onedotGetBalance(od, dd2d, verbose=1, fig=fign)

                dstrhi = '%s-sweep-2d-hires' % (od['name'])
                tmphi = loadExperimentData(resultsdir, tag='one_dot', dstr=dstrhi)
                alldatahi=tmphi['dataset']
                if verbose>=2:
                    print('  at onedotGetBalanceFine')
                if (alldatahi is not None) and True:
                    ptv, fimg, _ = onedotGetBalanceFine( dd=alldatahi, verbose=1, fig=None)
                    od['balancepointfine'] = ptv
                    od['setpoint'] = ptv + 10

                if verbose>=2:
                    print('  at fillPoly')
                imx = 0 * wwarea.copy().astype(np.uint8)
                tmp = fillPoly(imx, od['balancefit'])
                #cv2.fillConvexPoly(imx, od['balancefit'],color=[1] )

                showODresults(od, dd2d, fig=figm, imx=imx, ww=wwarea)
                if 0:
                    plt.close(fig + 10 * ii + 0)
                    plt.close(fig + 10 * ii + 2)
                    plt.close(fig + 10 * ii + 3)
                ods.append(od)

            if fig:
                tilefigs([fig + 1, fig + 11], [2, 2])

            # Define base values

            tmp = copy.copy(basevalues)
            # print(tmp)
            #print('createDoubleDotJobs: call getTwoDotValues: ')
            basevaluesTD, tddata = getTwoDotValues( td, ods, basevaluestd=tmp, verbose=1)
            # print('### createDoubleDotJobs: debug here: ')
            td['basevalues'] = basevaluesTD
            td['tddata'] = tddata

            # Create scan job

            scanjob = dict({'mode': '2d'})
            p1 = ods[0]['gates'][1]
            p2 = ods[1]['gates'][1]

            e1 = ods[0]['pinchvalues'][1]
            e2 = ods[1]['pinchvalues'][1]
            e1 = float(np.maximum(basevaluesTD[p1] - 120, e1))
            e2 = float(np.maximum(basevaluesTD[p2] - 120, e2))
            s1 = basevaluesTD[p1] + 120
            s2 = basevaluesTD[p2] + 120
            #s1=np.minimum(basevalues[p1], e1+240)
            #s2=np.minimum(basevalues[p2], e2+240)
            scanjob['stepdata'] = dict( {'gates': [p1], 'start': s1, 'end': e1, 'step': -2})
            scanjob['sweepdata'] = dict({'gates': [p2], 'start': s2, 'end': e2, 'step': -2})

            scanjob['keithleyidx'] = [1, 2]
            scanjob['basename'] = 'doubledot-2d'
            scanjob['basevalues'] = basevaluesTD
            scanjob['td'] = td
            jobs.append(scanjob)

            print('createDoubleDotJobs: succesfully created job: %s' % str(basevaluesTD))
        except Exception as e:
            print(e)
            print('createDoubleDotJobs: failed to create job file %s' % td['name'])
            #pdb.set_trace()
            continue
            pass

    return jobs

if __name__=='__main__':
    jobs=createDoubleDotJobs(two_dots, one_dots, basevalues=basevalues0, resultsdir=outputdir, fig=None)


#%%
def get_two_dots(full=1):
    """ return all posible simple two-dots """
    two_dots = []
    two_dots += [dict({'gates': ['L', 'P1', 'D1', 'D1', 'P2', 'D2']})]
    if full:
        two_dots += [dict({'gates': ['D1', 'P2', 'D2', 'D2', 'P3', 'D3']})]
        two_dots += [dict({'gates': ['D2', 'P3', 'D3', 'D3', 'P4', 'R']})]
        #two_dots += [dict({'gates': [ 'L', 'P1', 'D1', 'SD4a', 'SD4b', 'SD4c']})]

    for td in two_dots:
        td['name'] = '-'.join(td['gates'])
    return two_dots
    
def stopbias(gates):
    """ Stop the bias currents in the sample """
    gates.set_bias_1(0)
    gates.set_bias_2(0)
    gates.set_bias_3(0)
    
def stop_AWG(awg1):
    """ Stop the AWG """
    print('FIXME: add this function to the awg driver')
    if not awg1 is None:
        awg1.stop()
        awg1.set_ch1_status('off')
        awg1.set_ch2_status('off')
        awg1.set_ch3_status('off')
        awg1.set_ch4_status('off')
    print('stopped AWG...')
    

def printGateValues(gv, verbose=1):
    s = ', '.join(['%s: %.1f' % (x, gv[x]) for x in sorted(gv.keys())])
    return s

def getODbalancepoint(od):
    bp = od['balancepoint']
    if 'balancepointfine' in od:
        bp = od['balancepointfine']
    return bp

import pickle

def loadpickle(pkl_file):
    """ Load objects from file """
    try:    
        output = open(pkl_file, 'rb')
        data2 = pickle.load(output)
        output.close()
    except:
        if sys.version_info.major>=3:
            # if pickle file was saved in python2 we might fix issues with a different encoding
            output = open(pkl_file, 'rb')
            data2 = pickle.load(output, encoding='latin')
            #pickle.load(pkl_file, fix_imports=True, encoding="ASCII", errors="strict")
            output.close()
        else:
            data2=None
    return data2
    
def load_qt(fname):
    """ Load qtlab style file """
    alldata = loadpickle(fname)
    if isinstance(alldata, tuple):
        alldata = alldata[0]
    return alldata
