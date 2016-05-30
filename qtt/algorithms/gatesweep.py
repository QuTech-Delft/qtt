import scipy
import scipy.ndimage
from qtt import cfigure, plot2Dline
import qcodes
import numpy as np
import matplotlib.pyplot as plt

#import qtt.scans # FIXME: circular

def analyseGateSweep(dd, fig=None, minthr=None, maxthr=None, verbose=1, drawsmoothed=True, drawmidpoints=True):
    """ Analyse sweep of a gate for pinch value, low value and high value

    Arguments
    ---------
        dd : DataSet     
            structure containing the scan data
        minthr, maxthr : float
            parameters for the algorithm (default: None)
    
    """

    goodgate = True
   
    if isinstance(dd, qcodes.DataSet):
        data=dd
        XX=None
        
        # should be made generic
        g=[x for x in list(data.arrays.keys()) if x!='amplitude'][0]
        value='amplitude'
        
        x=data.arrays[g]
        value=data.arrays[value]
        
        # detect direction of scan
        scandirection=np.sign(x[-1]-x[0])
        if scandirection<0 and 1:
            pass
            scandirection=1
            x=x[::-1]
            value=value[::-1]
    else:
        # legacy code
        XX = dd['data_array']
        datashape = XX.shape
        goodgate = True
        sweepdata = dd['sweepdata']
        g = sweepdata['gates'][0]

        sr = np.arange(sweepdata['start'], sweepdata['end'], sweepdata['step'])
        if datashape[0] == 2 * sr.size:
                    # double sweep, reduce
            if verbose:
                print('analyseGateSweep: scan with sweepback: reducing data')
            XX = XX[0:sr.size, :]

        x = XX[:, 0]
        value = XX[:, 2]
        XX=None
        
    
    lowvalue = np.percentile(value, 1)
    highvalue = np.percentile(value, 90)
    # sometimes a channel is almost completely closed, then the percentile
    # approach does not function well
    ww = value[value >= (lowvalue + highvalue) / 2]
    #[np.percentile(ww, 1), np.percentile(ww, 50), np.percentile(ww, 91) ]
    highvalue = np.percentile(ww, 90)

    if verbose >= 2:
        print('analyseGateSweep: lowvalue %.1f highvalue %.1f' %
              (lowvalue, highvalue))
    d = highvalue - lowvalue

    vv1 = value > (lowvalue + .2 * d)
    vv2 = value < (lowvalue + .8 * d)
    midpoint1 = vv1 * vv2
    ww = midpoint1.nonzero()[0]
    midpoint1 = np.mean(x[ww])

    # smooth signal
    kk = np.ones(3) / 3.
    ww = value
    for ii in range(4):
        ww = scipy.ndimage.filters.correlate1d(ww, kk, mode='nearest')
        #ww=scipy.signal.convolve(ww, kk, mode='same')
        #ww=scipy.signal.convolve2d(ww, kk, mode='same', boundary='symm')
    midvalue = .7 * lowvalue + .3 * highvalue
    if scandirection>=0:
        mp = (ww >= (.7 * lowvalue + .3 * highvalue)).nonzero()[0][0]
    else:
        mp = (ww >= (.7 * lowvalue + .3 * highvalue)).nonzero()[0][-1]
    mp=max(mp, 2) # fix for case with zero data signal
    midpoint2 = x[mp]
    
    if verbose >= 2:
        print('analyseGateSweep: midpoint2 %.1f midpoint1 %.1f' %
              (midpoint2, midpoint1))

    if minthr is not None and np.abs(lowvalue) > minthr:
        if verbose:
            print('analyseGateSweep: gate not good: gate is not closed')
            print(' minthr %s' % minthr)
        midpoint1 = np.percentile(x, .5)
        midpoint2 = np.percentile(x, .5)

        goodgate = False

    # check for gates that are fully open or closed
    if scandirection>0:    
        xleft = x[0:mp]
        leftval = ww[0:mp]
        rightval = ww[mp]
    else:
        xleft = x[mp:]
        leftval = ww[mp:]
        rightval = ww[0:mp]
    st = ww.std()
    if verbose:
        print('analyseGateSweep: leftval %.1f, rightval %.1f' %
              (leftval.mean(), rightval.mean()))
    if goodgate and (rightval.mean() - leftval.mean() < .3 * st):
        if verbose:
            print(
                'analyseGateSweep: gate not good: gate is not closed (or fully closed)')
        midpoint1 = np.percentile(x, .5)
        midpoint2 = np.percentile(x, .5)
        goodgate = False

    # fit a polynomial to the left side
    if goodgate and leftval.size > 5:
        # TODO: make this a robust fit
        fit = np.polyfit(xleft, leftval, 1)
        pp = np.polyval(fit, xleft)

        #pmid = np.polyval(fit, midpoint2)
        p0 = np.polyval(fit, xleft[-1])
        pmid = np.polyval(fit, xleft[0])
        if verbose:
	        print('p0 %.1f, pmid %.1f, leftval[0] %.1f' % (p0, pmid, leftval[0]))

        if pmid + (pmid - p0) * .25 > leftval[0]:
            midpoint1 = np.percentile(x, .5)
            midpoint2 = np.percentile(x, .5)
            goodgate = False
            if verbose:
                print(
                    'analyseGateSweep: gate not good: gate is not closed (or fully closed) (line fit check)')

    # another check on closed gates
    if scandirection>0:
        leftidx = range(0, mp)
        fitleft = np.polyfit(
            [x[leftidx[0]], x[leftidx[-1]]], [lowvalue, value[leftidx[-1]]], 1)
    else:
        leftidx = range(mp, value.shape[0])
        fitleft = np.polyfit(
            [x[leftidx[-1]], x[leftidx[0]]], [lowvalue, value[leftidx[0]]], 1)
    leftval = value[leftidx]

    #fitleft=np.polyfit([x[leftidx[-1]],x[leftidx[0]]], [lowvalue, midvalue], 1)

    leftpred = np.polyval(fitleft, x[leftidx])

    if np.abs( scandirection*(xleft[1]-xleft[0]) ) >150 and xleft.size>15:
        xleft0 = x[mp+6:]
        leftval0 = ww[mp+6:]
        fitL = np.polyfit(xleft0, leftval0, 1)
        pp = np.polyval(fitL, xleft0)
        nd=fitL[0]/(highvalue-lowvalue)
        if goodgate and (nd*750>1):
            midpoint1 = np.percentile(x, .5)
            midpoint2 = np.percentile(x, .5)
            goodgate = False
            if verbose:
                print('analyseGateSweep: gate not good: gate is not closed (or fully closed) (slope check)')
            pass        

    if np.sum(leftval - leftpred) > 0:
        midpoint1 = np.percentile(x, .5)
        midpoint2 = np.percentile(x, .5)
        goodgate = False
        if verbose:
            print(
                'analyseGateSweep: gate not good: gate is not closed (or fully closed) (left region check)')
        pass
    # gate not closed

    if fig is not None:
        cfigure(fig)
        plt.clf()
        plt.plot(x, value, '.-b', linewidth=2)
        plt.xlabel('Sweep %s [mV]' % g, fontsize=14)
        plt.ylabel('keithley [pA]', fontsize=14)

        if drawsmoothed:
            plt.plot(x, ww, '-g', linewidth=1)

        plot2Dline([0, -1, lowvalue], '--m', label='low value')
        plot2Dline([0, -1, highvalue], '--m', label='high value')

        if drawmidpoints:
            if verbose>=2:
                plot2Dline([-1, 0, midpoint1], '--g', linewidth=1)
            plot2Dline([-1, 0, midpoint2], '--m', linewidth=2)

        if verbose>=2: 
            #plt.plot(x[leftidx], leftval, '.r', markersize=15)
            plt.plot(x[leftidx], leftpred, '--r', markersize=15, linewidth=1)

        if verbose >= 2:
            1
           # plt.plot(XX[ww,0].astype(int), XX[ww,2], '.g')

    adata = dict({'pinchvalue': midpoint2 - 50,
                  'pinchvalueX': midpoint1 - 50, 'goodgate': goodgate})
    adata['lowvalue'] = lowvalue
    adata['highvalue'] = highvalue
    adata['xlabel'] = 'Sweep %s [mV]' % g
    adata['mp'] = mp
    adata['midpoint'] = midpoint2
    adata['midvalue'] = midvalue

    if verbose >= 2:
        print('analyseGateSweep: gate status %d: pinchvalue %.1f' %
              (goodgate, adata['pinchvalue']))
        adata['Xsmooth'] = ww
        adata['XX'] = XX
        adata['X'] = value
        adata['x'] = x
        adata['ww'] = ww

    return adata

#%%


    
   
#%%    
import scipy
import pmatlab
import cv2

def costscoreOD(a, b, pt, ww, verbose=0, output=False):
    """ Cost function for simple fit of one-dot open area """
    pts = np.array(
        [[a, 0], pt, [ww.shape[1] - 1, b], [ww.shape[1] - 1, 0], [a, 0]])
    pts = pts.reshape((5, 1, 2)).astype(int)
    imx = 0 * ww.copy().astype(np.uint8)
    cv2.fillConvexPoly(imx, pts, color=[1])
    #tmp=fillPoly(imx, pts)

    cost = -(imx == ww).sum()

    # add penalty for moving out of range
    cost += (.025 * ww.size) * np.maximum(b - ww.shape[0], 0) / ww.shape[0]
    cost += (.025 * ww.size) * np.maximum(a, 0) / ww.shape[1]

    cost += (.025 * ww.size) * 2 * (pts[2, 0, 1] < 0)

    if verbose:
        print('costscore %.2f' % cost)
    if output:
        return cost, pts, imx
    else:
        return cost
        
def onedotGetBalance(od, dd, verbose=1, fig=None, drawpoly=False, polylinewidth=2, linecolor='c'):
    """ Determine tuning point from a 2D scan of a 1-dot """
    #XX = dd['data_array']
    extent, g0,g1,vstep, vsweep, arrayname=qtt.scans.dataset2Dmetadata(dd, array=None)
    
    scanjob=dd.metadata['scanjob']
    #vstep = np.unique(XX[:, 0])
    #vsweep = np.unique(XX[:, 1])
    stepdata = scanjob['stepdata']
    #g0 = stepdata['gates'][0]
    sweepdata = scanjob['sweepdata']
    #g2 = sweepdata['gates'][0]

    nx = vstep.size
    ny = vsweep.size

    im= np.array(dd.arrays[arrayname])
    #im = im[::, ::-1]

    extentImage = [vstep.min(), vstep.max(), vsweep.min(), vsweep.max()]

    ims = im.copy()
    kk = np.ones((3, 3)) / 9.
    for ii in range(2):
        ims = scipy.ndimage.convolve(ims, kk, mode='nearest', cval=0.0)

    r = np.percentile(ims, 99) - np.percentile(ims, 1)
    lv = np.percentile(ims, 2) + r / 100
    x = ims.flatten()
    lvstd = np.std(x[x < lv])
    lv = lv + lvstd / 2  # works for very smooth images

    lv = (.45 * pmatlab.otsu(ims) + .55 * lv)  # more robust
    if verbose:
        print('onedotGetBalance: threshold for low value %.1f' % lv)

    # balance point: method 1
    try:
        ww = np.nonzero(ims > lv)
        # ww[0]+ww[1]
        zz = -ww[0] + ww[1]
        idx = zz.argmin()
        pt = np.array([[ww[1][idx]], [ww[0][idx]]])
        ptv = np.array([[vstep[pt[0, 0]]], [vsweep[-pt[1, 0]]]])
    except:
        print('qutechtnotools: error in onedotGetBalance: please debug')
        idx = 0
        pt = np.array([[int(vstep.size / 2)], [int(vsweep.size / 2)]])
        ptv = np.array([[vstep[pt[0, 0]]], [vsweep[-pt[1, 0]]]])
        pass
    od['balancepoint0'] = ptv

    # balance point: method 2
    wwarea = ims > lv

    #x0=np.array( [pt[0],im.shape[0]+.1,pt[0], pt[1] ] )
    x0 = np.array([pt[0] - .1 * im.shape[1], pt[1] + .1 *
                   im.shape[0], pt[0], pt[1]]).reshape(4,)  # initial square
    ff = lambda x: costscoreOD(x[0], x[1], x[2:4], wwarea)
    # ff(x0)

    # scipy.optimize.show_options(method='Nelder-Mead')

    opts = dict({'disp': verbose >= 2, 'ftol': 1e-6, 'xtol': 1e-5})
    xx = scipy.optimize.minimize(ff, x0, method='Nelder-Mead', options=opts)
    #print('  optimize: %f->%f' % (ff(x0), ff(xx.x)) )
    opts['disp'] = verbose >= 2
    xx = scipy.optimize.minimize(ff, xx.x, method='Powell', options=opts)
    x = xx.x
    #print('  optimize: %f->%f' % (ff(x0), ff(xx.x)) )
    cost, pts, imx = costscoreOD(x0[0], x0[1], x0[2:4], wwarea, output=True)
    balancefitpixel0 = pts.reshape((-1, 2)).T.copy()
    cost, pts, imx = costscoreOD(x[0], x[1], x[2:4], wwarea, output=True)
    pt = pts[1, :, :].transpose()

    od['balancepointpixel'] = pt
    od['balancepointpolygon'] = pix2scan(pt, dd)
    od['balancepoint'] = pix2scan(pt, dd)
    od['balancefitpixel'] = pts.reshape((-1, 2)).T
    od['balancefit'] = pix2scan(od['balancefitpixel'], dd)
    od['balancefit1'] = pix2scan(balancefitpixel0, dd)
    od['setpoint'] = od['balancepoint'] + 8
    od['x0'] = x0
    # od['xx']=dict(xx)
    ptv = od['balancepoint']

    # print(balancefitpixel0)
    # print(od['balancefitpixel'])

    if verbose:
        print('balance point 0 at: %.1f %.1f [mV]' % (ptv[0, 0], ptv[1, 0]))
        print('balance point at: %.1f %.1f [mV]' % (
            od['balancepoint'][0, 0], od['balancepoint'][1, 0]))

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plt.imshow(im, extent=extentImage, interpolation='nearest')
        plt.axis('image')
        if verbose >= 2 or drawpoly:
            pmatlab.plotPoints(od['balancefit'], '--', color=linecolor, linewidth=polylinewidth)
            if verbose >= 2:
                pmatlab.plotPoints(od['balancefit0'], '--r')
        if verbose>=2:
            pmatlab.plotPoints(od['balancepoint0'], '.r', markersize=13)
        pmatlab.plotPoints(od['balancepoint'], '.m', markersize=17)

        plt.title('image')
        plt.xlabel('%s (mV)' % g2)
        plt.ylabel('%s (mV)' % g0)

        plt.figure(fig + 1)
        plt.clf()
        plt.imshow(ims, extent=None, interpolation='nearest')
        plt.axis('image')
        plt.title('Smoothed image')
        pmatlab.plotPoints(pt, '.m', markersize=16)
        plt.xlabel('%s (mV)' % g2)
        plt.ylabel('%s (mV)' % g0)

        plt.figure(fig + 2)
        plt.clf()
        plt.imshow(ims > lv, extent=None, interpolation='nearest')
        pmatlab.plotPoints(balancefitpixel0, ':y', markersize=16)
        pmatlab.plotPoints(od['balancefitpixel'], '--c', markersize=16)
        pmatlab.plotLabels(od['balancefitpixel'])
        plt.axis('image')
        plt.title('thresholded area')
        plt.xlabel('%s (mV)' % g2)
        plt.ylabel('%s (mV)' % g0)
        pmatlab.tilefigs([fig, fig + 1, fig + 2], [2, 2])

        if verbose >= 2:
            qq = ims.flatten()
            plt.figure(123)
            plt.clf()
            plt.hist(qq, 20)
            plot2Dline([-1, 0, np.percentile(ims, 1)], '--m')
            plot2Dline([-1, 0, np.percentile(ims, 2)], '--m')
            plot2Dline([-1, 0, np.percentile(ims, 99)], '--m')
            plot2Dline([-1, 0, lv], '--r', linewidth=2)

    return od, ptv, pt, ims, lv, wwarea

if __name__=='__main__':
    od, ptv, pt,ims,lv, wwarea=onedotGetBalance(od, alldata, verbose=1, fig=10)
    
#%% Testing

if __name__=='__main__':
     adata = analyseGateSweep(alldata, fig=10, minthr=None, maxthr=None)
     
    