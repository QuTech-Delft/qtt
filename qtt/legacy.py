import qtpy
#print(qtpy.API_NAME)

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

# should be removed later
#from pmatlab import tilefigs


#%%

def getPinchvalues(od, xdir):
    """ Get pinch values from recorded data """
    gg = od['gates']
    od['pinchvalues'] = -800 * np.ones(3)
    for jj, g in enumerate(gg):
        #pp='%s-sweep-1d-%s.pickle' % (od['name'], g)
        pp = pinchoffFilename(g, od=None)
        pfile = os.path.join(xdir, pp + '.pickle')
        dd = load_data(pfile)

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
        # get balance point data
        ods = []
        try:
            for ii, od in enumerate([od1, od2]):

                dstr = '%s-sweep-2d' % (od['name'])
                dd2d = loadExperimentData(resultsdir, tag='one_dot', dstr=dstr)

                od = getPinchvalues(od, xdir)

                if fig:
                    fign = 1000 + 100 * jj + 10 * ii
                    figm = fig + 10 * ii
                else:
                    fign = None
                    figm = None

                # fign=None
                od, ptv, pt0, ims, lv, wwarea = onedotGetBalance(od, dd2d, verbose=1, fig=fign)

                dstrhi = '%s-sweep-2d-hires' % (od['name'])
                alldatahi = loadExperimentData(
                    resultsdir, tag='one_dot', dstr=dstrhi)
                if alldatahi is not None and True:
                    ptv, fimg, _ = onedotGetBalanceFine( dd=alldatahi, verbose=1, fig=None)
                    od['balancepointfine'] = ptv
                    od['setpoint'] = ptv + 10

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
