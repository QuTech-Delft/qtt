""" Main script to perform automatic tuning of quantum dots

   Pieter Eendebak <pieter.eendebak@tno.nl>

"""

#%% Import the modules used in this program:

from imp import reload
import sys,os,platform, pdb
import logging
import numpy as np
os.environ['QT_API'] = 'pyqt'
import pdb
import qtpy
import logging
import matplotlib
#matplotlib.use('Qt4Agg')

import matplotlib.pyplot
matplotlib.pyplot.ion()


import multiprocessing
if __name__=='__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except:
        pass
#print(qtpy.QT_API)

import webbrowser, datetime, copy
import matplotlib.pyplot as plt

import qcodes
import qcodes as qc
from qcodes.plots.qcmatplotlib import MatPlot

from functools import partial
import qtt.scans

import qtt; # reload(qtt)
import qtt.scans
from qtt.scans import experimentFile
import qtt.reports
#reload(qtt); reload(qtt.scans); reload(qtt.data); reload(qtt.algorithms); reload(qtt.algorithms.generic); reload(qtt); reload(qtt.reports)
#import qcodes.utils.reload_code
#_=qcodes.utils.reload_code.reload_code()
from qcodes.utils.validators import Numbers

from qtt.data import *
from qtt.scans import *
from qtt.legacy import *
from qtt.reports import *

#%% Load configuration

if __name__=='__main__':

    if platform.node()=='TUD205521':
        import stationV2 as msetup
        from stationV2 import sample
        awg1=None
        virtualAWG=None
        def simulation():
            ''' Funny ... '''
            return False
    else:
        import virtualV2 as msetup
        from virtualV2 import sample
        awg1=None
        def simulation():
            ''' Funny ... '''
            return True
        
    msetup.initialize(reinit=False, server_name='virtualV2-%d' % np.random.randint(100) )
    #msetup.initialize(reinit=False, server_name=None )
    



#%% Make instances available

if __name__=='__main__':
    station = msetup.getStation()
    
    station.gate_settle=sample.gate_settle
    
    keithley1 = station.keithley1
    keithley3 = station.keithley3
    
    gates = station.gates
    
    station.set_measurement(keithley3.amplitude)
    
    if platform.node()=='TUD205521':
        datadir =  r'p:\data\qcodes'   
    else:
        datadir = '/home/eendebakpt/data/qdata'
    
    qcodes.DataSet.default_io = qcodes.DiskIO(datadir)
    mwindows=qtt.setupMeasurementWindows(station)
    mwindows['parameterviewer'].callbacklist.append( mwindows['plotwindow'].update )
    
    
    mwindows['parameterviewer'].callbacklist.append( partial(qtt.updatePlotTitle, mwindows['plotwindow']) )
    
    #liveplotwindow.win.setWindowTitle('sfs')
    
    qtt.scans.mwindows=mwindows
    liveplotwindow=mwindows['plotwindow']
    qtt.live.liveplotwindow=liveplotwindow
    

#%% Define 1-dot combinations
from stationV2.sample import get_one_dots

if __name__=='__main__':

    
    verbose=2   # set output level of the different functions
    
    full=1      # for full=0 the resolution of the scans is reduced, this is usefull for quick testing
    one_dots=get_one_dots(sdidx=[])
    full=0
    
    sdindices=[1,2]
    sdindices=[1,]
    
    
    sddots=get_one_dots(sdidx=sdindices)[-len(sdindices):]
    
    #one_dots=[one_dots[0], one_dots[-1] ];
    full=0
    dohires=1
    
    logging.info('## 1dot_script_batch: scan mode: full %d, scanning %d one-dots' % (full, len(one_dots)) )
    
    timestart=str(datetime.datetime.now())

#%%
Tvalues=[]
if __name__=='__main__':

    sdgates = qtt.flatten([s['gates'] for s in sddots])
    activegates=['T'] + list(qtt.flatten([s['gates'] for s in one_dots])) + sdgates
    
    basevalues=dict()
    for g in activegates:
        basevalues[g]=0
    
    
    basetag='batch-2016-09-06'; Tvalues=np.array([-380])
    
    
    #basetag='batch-16102015'; Tvalues=np.array([-390])
    
    b=False
    
    if b:
        basetag=basetag + 'b'
        # we set the sensing dots to some biased values, otherwise we cannot close them
        basevalues['SD1a']=-150
        basevalues['SD1b']=-150
        basevalues['SD1c']=-150
    
        basevalues['SD2a']=-150
        basevalues['SD2b']=-150
        basevalues['SD2c']=-150
    else:
        basevalues['SD1a']=0; basevalues['SD1b']=0; basevalues['SD1c']=0
        basevalues['SD2a']=0; basevalues['SD2b']=0; basevalues['SD2c']=0
    
    
    if 1:
        Pbias=-80
        basevalues['P1']=Pbias
        basevalues['P2']=Pbias
        basevalues['P3']=Pbias
        basevalues['P4']=Pbias
    
    #one_dots=one_dots[-1:]; Tvalues=np.array([-350])

#%% Do measurements
if __name__=='__main__':

    cache=1
    measureFirst=True    # measure 1-dots
    measureSecond=True   # measure 2-dots
    
    #measureFirst=0;
    #measureSecond=0
    
    # if we are working offline we cannot measure, but only process results
    if simulation() and 0:
        measureFirst=0; measureSecond=0
    
    
    # new januari sample
    def getSDval(T, gg=None):
        if not gg is None:
            if gg[1]=='SD2b':
                sdval=[-324,-450,-290]  # T ??
            else:
                  sdval=[-330,-170,-380]  # ...
                  sdval=[-160,-430,-480]  # T-350, from scan
                  sdval=[-320,-500,-500]  # from T=-375 SD scan
            return sdval
        #sdval=[-330,-500,-370]  # T-350, natasja
        return sdval
    
    # define the index of the sensing dot to use for double-dot scans
    sdid=1
    #sdid=2
    
    ggsd=['SD%d%s' % (sdid,c) for c in ['a','b','c']];
    
    import qtt.structures
    from qtt.structures import sensingdot_t
    
    # define sensing dot
    sdval=getSDval(Tvalues[0], ggsd)
    sd=sensingdot_t(ggsd, sdval, station=station, index=sdid  )
    
    
    sd2=None
    
    
    if full:
        hiresstep=-2
    else:
        hiresstep=-4
    
    
    def stepDelay(gate, minstepdelay=0, maxstepdelay=10):
        return 0
    
#%%

  
def onedotPlungerScan(station, od, verbose=1):
    """ Make a scan with the plunger of a one-dot """
    # do sweep with plunger
    gates=station.gates
    gg=od['gates']
    ptv=od['setpoint']
              
    gates.set(gg[2], ptv[0,0]+20)    # left gate = step gate in 2D plot =  y axis
    gates.set(gg[0], ptv[1,0]+20)
    
    pv=od['pinchvalues'][1]

    scanjob=dict({'keithleyidx': [od['instrument'] ]})
    scanjob['sweepdata']=dict({'gates': [gg[1]], 'start': 50, 'end': pv,'step': -1})
    
    wait_time = qtt.scans.waitTime(gg[1], gate_settle=getattr(station, 'gate_settle', None))

    alldata=scan1D(scanjob, station, delay=wait_time, title_comment='sweep of plunger')            
    alldata.metadata['od']=od
    scandata=dict(dataset=alldata, od=od)
    return scandata

#alldataplunger=onedotPlungerScan(station, od, verbose=1)

def stop_AWG(station):
    print('this is a dummy function!')
    
def closeExperiment(station, eid=None):
    gates=station.gates
    print('set bias to zero to save energy')
    gates.set_bias_1(0)   # bias through O1, keithley 1
    gates.set_bias_2(0)   # bias through O7, keithley 2
    gates.set_bias_3(0)   # bias through O?, keithley 3

    #if not RFsiggen1 is None:
    #    print(' stop Microwave source()')
    #    RFsiggen1.off()

    print(' stop AWG')
    stop_AWG(station)

    print('closed experiment: %s' % getDateString())
    
#%% One-dot measurements

def onedotScan(station, od, basevalues, outputdir, verbose=1):
    if verbose:
        print('onedotScan: one-dot: %s' % od['name'] )
    gg=od['gates']
    keithleyidx=[ od['instrument'] ]

    #if od['keithley']==1:
    #    RFsiggen1.on()

    gates.set( gg[1], float(basevalues[gg[1]]-0 ) )    # plunger

    pv1=od['pinchvalues'][0]+0
    pv2=od['pinchvalues'][2]+0
    stepstart=float(np.minimum( od['pinchvalues'][0]+400, 90))
    sweepstart=float(np.minimum( od['pinchvalues'][2]+300, 90) )
    stepdata=dict({'gates': [gg[0]], 'start': stepstart, 'end': pv1-10, 'step': -3})
    sweepdata=dict({'gates': [gg[2]], 'start': sweepstart, 'end': pv2-10, 'step': -3})

    if full==0:
        stepdata['step']=-12; sweepdata['step']=-12
        #stepdata['step']=-6; sweepdata['step']=-6


    wait_time = qtt.scans.waitTime(gg[2], gate_settle=getattr(station, 'gate_settle', None))
    scanjob=dict({'stepdata':stepdata, 'sweepdata':sweepdata, 'keithleyidx': keithleyidx})
    alldata=qtt.scans.scan2D(station, scanjob, wait_time=wait_time, background=False)

    od, ptv, pt,ims,lv, wwarea=qtt.onedotGetBalance(od, alldata, verbose=1, fig=None)

    alldata.metadata['od']=od
    
    #basename='%s-sweep-2d' % (od['name'])
    #alldata['od']=od
    return alldata, od

#alldata, od = onedotScan(station,od, basevaluesS, outputdir, verbose=1)
#od, ptv, pt,ims,lv, wwarea=qtt.onedotGetBalance(od, alldata, verbose=1, fig=10)

#%%
def onedotScanPinchValues(od, basevalues, outputdir, cache=False, full=0, verbose=1):
    """ Scan the pinch-off values for the 3 main gates of a 1-dot """
    od['pinchvalue'] = np.zeros((3, 1))
    keithleyidx = [od['instrument']]

    for jj, g in enumerate(od['gates']):
                #basename='%s-sweep-1d-%s' % (od['name'], g)
        alldata=scanPinchValue(station, outputdir, gate=g, basevalues=basevalues, keithleyidx=keithleyidx,cache=cache, full=full)

        adata = alldata.metadata['adata']
        od['pinchvalue'][jj] = adata['pinchvalue']

    return od
#%%
import pyqtgraph
from qtt.scans import scanPinchValue
    
from qtt.data import saveExperimentData, loadExperimentData     
app=pyqtgraph.mkQApp()

#%%


for ii, Tvalue in enumerate(Tvalues):
    if not __name__=='__main__':
        break
    tag=basetag+'-T%d'  % Tvalue
    outputdir=qtt.mkdirc(os.path.join(datadir, tag))
    if not measureFirst:
        continue;

    gates.set_T(float(Tvalue))       # set T value
    basevalues['T']=Tvalue

    gates.set_bias_2(-500)   # bias through O7, keithley 3


    #%% Main loop

    print('## start of scan for topgate %.1f [mV]: tag %s' % (Tvalue, tag) )
    tmp=qtt.mkdirc(os.path.join(outputdir, 'one_dot'));


    print('we have %d one-dots' % len(one_dots) )
    for ii, od in enumerate(one_dots):
        print('  gates: %s' % str(od['gates']) )
        print('     channel: %s (instrument %s)' % (str(od['channel']), od['instrument']) )

    print('active gates: %s' % str(activegates))


    #%% Initialize to default values

    print('## 1dot_script: initializing gates')
    qtt.resetgates(gates, activegates, basevalues)
    qtt.resetgates(gates, sdgates, basevalues)

    #%% Perform sanity check on the channels


    for gate in qtt.flatten([o['gates'] for o in one_dots]):
        alldata=qtt.scans.scanPinchValue(station, outputdir, gate, basevalues=basevalues, keithleyidx=[3], stepdelay=stepDelay(gate), cache=cache, full=full)


    for od in sddots:
        ki=od['instrument']
        for gate in od['gates']:
            scanPinchValue(station, outputdir, gate=gate, basevalues=basevalues, keithleyidx=[ki], cache=cache, stepdelay=stepDelay(gate), full=full)
            qtt.resetgates(gates,activegates, basevalues, verbose=0)

    ww=one_dots
    for od in ww:
        print('getting data for 1-dot: %s' % od['name'] )

        od = onedotScanPinchValues(od, basevalues, outputdir, cache=cache, full=full)

            #break

#scanPinchValue(station, outputdir, gate='SD2b', basevalues=basevalues, keithleyidx=[ki], cache=False, full=full, fig=10)

    #%% Re-calculate basevalues
    # todo: place this in function
    basevaluesS=copy.deepcopy(basevalues)
    for g in ['L', 'D1', 'D2', 'D3', 'R']:
            basename = qtt.scans.pinchoffFilename(g, od=None)
            pfile=os.path.join(outputdir, 'one_dot', basename )

            alldata, mdata=qtt.scans.loadDataset(pfile)

            adata=qtt.analyseGateSweep(alldata, fig=None, minthr=None, maxthr=None, verbose=1)

            basevaluesS[g]=float(min(adata['pinchvalue']+500, 0))

    #%% Analyse the one-dots
    dotlist=one_dots
    for od in dotlist:
        od = qtt.scans.loadOneDotPinchvalues(od, outputdir, verbose=1)

    # Make scans of the sensing dots

    for odii, od in enumerate(sddots):

        gates.resetgates(activegates, basevaluesS)

        basename='%s-sweep-2d' % (od['name'])
        basenameplunger='%s-sweep-plunger' % (od['name'])
        efileplunger=qtt.scans.experimentFile(outputdir, tag='one_dot', dstr=basenameplunger)
        if cache and os.path.exists(efileplunger) and 1:
            print('  skipping 2D and plunger scan of %s'  % od['name' ])
            #alldata=loadExperimentData(outputdir, tag='one_dot', dstr=basename)
            #d=alldata['od']
            continue

        od = qtt.scans.loadOneDotPinchvalues(od, outputdir, verbose=1)
        alldata, od = onedotScan(station, od, basevaluesS, outputdir, verbose=1)
        #qtt.QtPlot(alldata.amplitude, remote=False, interval=0)
        plt.figure(10); plt.clf(); MatPlot(alldata.arrays[alldata.default_parameter_name()], interval=0, num=10)
        pmatlab.plotPoints(od['balancepoint'], '.m', markersize=19)
        
        scandata, od=onedotHiresScan(station, od, dv=70, verbose=1)
        
        alldata.data_manager=None
        write_data(experimentFile(outputdir, tag='one_dot', dstr='%s-sweep-2d-hires' % (od['name'])) , scandata)
        #_=loadQttData(path = experimentFile(outputdir, tag='one_dot', dstr='%s-sweep-2d-hires' % (od['name'])) )
        


        saveExperimentData(outputdir, alldata, tag='one_dot', dstr=basename)

        alldataplunger=onedotPlungerScan(station, od, verbose=1)
        
        saveExperimentData(outputdir, alldataplunger, tag='one_dot', dstr=basenameplunger)

        basenamedot='%s-dot' % (od['name'])
        saveExperimentData(outputdir, od, tag='one_dot', dstr=basenamedot)

        gates.resetgates( od['gates'], basevaluesS)

        
    #%% update basevalues with settings for SD

    for odii, od in enumerate(sddots):
        basename='%s-sweep-2d' % (od['name'])
        basenamedot='%s-dot' % (od['name'])
        alldata = loadExperimentData(outputdir, tag='one_dot', dstr=basename)
        od = loadExperimentData(outputdir, tag='one_dot', dstr=basenamedot)

        ww=getODbalancepoint(od)
        basevaluesS[od['gates'][0]]=float(ww[1])
        basevaluesS[od['gates'][2]]=float(ww[0])

 

    #%%
    print('Do main 2D-sweeps for one-dots')

    #ww = [one_dots[0]]
    ww = one_dots
    for odii, od in enumerate(ww):

        gates.resetgates(activegates, basevaluesS)

        basename='%s-sweep-2d' % (od['name'])
        basenameplunger='%s-sweep-plunger' % (od['name'])
        efileplunger=experimentFile(outputdir, tag='one_dot', dstr=basenameplunger)
        if cache and os.path.exists(efileplunger) and 1:
            print('  skipping 2D and plunger scan of %s'  % od['name' ])
            continue

        alldata,od = onedotScan(station, od, basevaluesS, outputdir,verbose=1)
        saveExperimentData(outputdir, alldata, tag='one_dot', dstr=basename)
        if odii>10:
            STOP


        #% Make high-resolution scans
        if dohires:
            alldatahi, od=onedotHiresScan(station, od, dv=70, verbose=1)
            saveExperimentData(outputdir, alldatahi, tag='one_dot', dstr='%s-sweep-2d-hires' % (od['name']))

            #saveExperimentData(outputdir, alldata, tag='one_dot', dstr=basename) # needed?

        alldataplunger=onedotPlungerScan(station, od, verbose=1)
        saveExperimentData(outputdir, alldataplunger, tag='one_dot', dstr=basenameplunger)

        #saveCoulombData(datadir, alldataplunger)

    saveExperimentData(outputdir, basevaluesS, tag='one_dot', dstr='basevaluesS')


#STOP

#%%


#%% Reports for one-dots

#datadir=experimentdata.getDataDir()
#one_dots=get_one_dots(full=2)
if __name__=='__main__':

    #reload(qtt.data); reload(qtt.reports); from qtt.reports import *
    
    plt.close('all')
    for ii, Tvalue in enumerate(Tvalues):
        tag=basetag+'-T%d'  % Tvalue
    
        resultsdir=qtt.tools.mkdirc(os.path.join(datadir, tag) )
        xdir=os.path.join(resultsdir, 'one_dot')
    
        try:
            print('script: make report for %s' % tag )
            fname=generateOneDotReport(one_dots+sddots,xdir, resultsdir)
            webbrowser.open(fname, new=2)
        except Exception as ex:
            print('failed to generate one-dot report')
            print(ex)
            #print(traceback.format_exception(ex))
            fname=generateOneDotReport(one_dots+sddots,xdir, resultsdir)
    
            pass
    
    if len(Tvalues)>1:
        raise Exception(' not supported...' )
    

#%% Measurements for double-dots

if __name__=='__main__':

    basevaluesS=loadExperimentData(outputdir, tag='one_dot', dstr='basevaluesS')
    basevalues0=copy.copy(basevaluesS)
    
    #sdid=1
    ggsd=['SD%d%s' % (sdid,c) for c in ['a','b','c']];
    sdval=[ basevaluesS[g] for g in ggsd]
    sd=sensingdot_t(ggsd, sdval, station=station, index=sdid  )


#%%


for ii, Tvalue in enumerate(Tvalues):
    if not __name__=='__main__':
        break
    if not measureSecond:
        break;
    print('### 1dot_script_batch: double-dot scans for T=%.1f' % Tvalue)

    tag=basetag+'-T%d'  % (Tvalue)
    tag2d=basetag+'-T%d-sd%d'  % (Tvalue, sdid)

    gates.set_T(Tvalue)       # set T value

    outputdir=qtt.mkdirc(os.path.join(datadir, tag))
    outputdir2d=qtt.mkdirc(os.path.join(datadir, tag2d))

    gates.set_T(Tvalue)       # set T value


    #RFsiggen1.set_frequency(RFfreq)   # SD2
    qtt.legacy.stop_AWG(awg1)
    #RFsiggen1.off()
    qtt.legacy.stopbias(gates)
    #RFsiggen1.on()  # for sensing dot

    #instrumentStatus()

    #readfunc=lambda: keithley1.readnext()*(1e12/(Amp*10e6) )

    two_dots=qtt.legacy.get_two_dots(full=1)
    one_dots=sample.get_one_dots(full=1)

    # make two-dots (code from optimize_one)
    jobs = qtt.legacy.createDoubleDotJobs(two_dots, one_dots, basevalues=basevalues0, resultsdir=outputdir, fig=None)
    saveExperimentData(outputdir2d, jobs, tag='doubledot', dstr='jobs')
    #jobs = loadExperimentData(outputdir2d, tag='doubledot', dstr='jobs')

    print('## %d jobs to run' % len(jobs))
    #STOP

    #jobs=jobs[2:]

    #%% Define sensing dot
    #sd=sensingdot_t(ggsd, sdval)

    autotuneSD=True

    for ji, scanjob in enumerate(jobs):

        scanjob=jobs[ji]
        print('running job %d: %s: sweep step %d' % (ji, scanjob['basename'], scanjob['sweepdata']['step']))


        dstr='doubledot-%s' % scanjob['td']['name']
        xfile=experimentFile(outputdir2d, tag='doubledot', dstr=dstr)
        if cache and os.path.exists(xfile):
            pass
            continue

        basevaluesTD=copy.copy(scanjob['basevalues'])
        gates.resetgates(activegates, basevaluesTD)
        for c in ['a','b','c']:
            basevalues.pop('SD%d%s' % (sdid,c), None)

        stepdata=scanjob['stepdata']
        sweepdata=scanjob['sweepdata']

        # First tune sd2
        if sd2 is not None:
            sd2.initialize()
            tmp, alldata = sd2.autoTune(outputdir=None, max_wait_time=.5, step=-4)
            dstr='tunesd-%s-sd%d' % (scanjob['td']['name'], sd2.index)
            saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)

        #%% Autotune
        sd.initialize()
        if autotuneSD:
            print('### autotune SD')
            sd.autoTuneInit(scanjob)
            tmp, alldata = sd.autoTune(outputdir=outputdir, max_wait_time=.5, step=-4)

            dstr='tunesd-%s-sd%d' % (scanjob['td']['name'], sd.index)
            saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)

        #STOP
        #%% Scan of plunger

        g=sd.tunegate()
        cvalstart=sd.sdval[1]

        if scanjob['sweepdata']['gates'][0]=='P2':
            # swap!
            print('swapping step and sweep gate!!!')
            scanjob['sweepdata'],scanjob['stepdata']=scanjob['stepdata'],scanjob['sweepdata']
            stepdata=scanjob['stepdata']
            sweepdata=scanjob['sweepdata']

        compensateGates=[]
        gate_values_corners=[]

        print(scanjob['stepdata'])
        print(scanjob['sweepdata'])


        #STOP
        #plot1D(alldata, fig=100)

        if 0:
            # calculate settings for gate compensation
            g=sd.tunegate()
            (sdstart, sdend, sdmiddle)=sd.autoTuneFine(scanjob=scanjob, fig=300)
            #(sdstart, sdend, tmp) = autoTuneCompensation(sd, scanjob, fig=300)

            compensateGates=[g]
            gate_values_corners=[[sdstart, sdstart, sdend, sdend]]
            sd.sdval[1]=(sdstart+sdend)/2
        else:
            print('WARNING: fine tuning not enabled implemented!')

        scanjob['compensateGates']=compensateGates
        scanjob['gate_values_corners']=gate_values_corners
        gates.set(sweepdata['gates'][0], sweepdata['start'])



        if 1:
            # slow scan
            print('slow scan without compensation!')
            sd.initialize(setPlunger=True)
            defaultactivegates=[]
            alldata = scan2D(station, scanjob, title_comment='scan double-dot', wait_time=None, background=False)
            dstr='doubledot-%s-gc' % scanjob['td']['name']
            alldata.metadata['sd']=str(sd)
            saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)

            #pt, resultsfine = analyse2dot(alldata, fig=300, efig=None, istep=1)
            print('WARNING: skipping analysis')
            print('WARNING: skipping hires scan')
            if 0:
                scanjobc=positionScanjob(scanjob, resultsfine['ptmv'])
                alldatac, data = scan2Dfastjob(scanjobc, TitleComment='scan double-dot', wait_time=wait_time, activegates=defaultactivegates())
                dstr='doubledot-center-%s' % scanjob['td']['name']
                saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)

                ptI, resultsfineI = analyse2dot(alldatac, fig=300, efig=None, istep=1)

        #%% Scan in fast mode...
        if 0:
            scanjob['Naverage']=160
            gstep, gsweep, center, d,voltages_step = fastScan(stepdata, sweepdata)
            wait_time=None
            wait_time=0 # try
    
            if 0:
                print('fast scan with compensation!')
                sd.initialize(setPlunger=True)
                alldata, data = scan2Dfastjob(scanjob, TitleComment='scan double-dot', wait_time=wait_time, activegates=defaultactivegates())
                dstr='doubledot-%s-gc' % scanjob['td']['name']
                alldata['sd']=str(sd)
                saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)
    
                pt, resultsfine = analyse2dot(alldata, fig=300, efig=None, istep=1)
                if 1:
                    scanjobc=positionScanjob(scanjob, resultsfine['ptmv'])
                    alldatac, data = scan2Dfastjob(scanjobc, TitleComment='scan double-dot', wait_time=wait_time, activegates=defaultactivegates())
                    dstr='doubledot-center-%s' % scanjob['td']['name']
                    saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)
    
                    ptI, resultsfineI = analyse2dot(alldatac, fig=300, efig=None, istep=1)
    
                #_=show2D(alldata, fig=200)
    
                dv=20
                dx=resultsfineI['ptmv'].flatten()-np.array( [(scanjob['sweepdata']['start']+scanjob['sweepdata']['end'])/2, (scanjob['stepdata']['start']+scanjob['stepdata']['end'])/2 ] )-np.array([[dv],[dv]]).flatten()
                dx=np.linalg.norm(dx)
                dthr=10
                nitermax=4; niter=0
                if dx>dthr and niter<nitermax:
                    targetpos =  resultsfineI['ptmv']-np.array([[dv],[dv]])
                    # iterate untill good
                    # FIXME: reposition SD?
                    scanjobc=positionScanjob(scanjob,targetpos)
                    alldatac, data = scan2Dfastjob(scanjobc, TitleComment='scan double-dot', wait_time=wait_time, activegates=defaultactivegates())
                    dstr='doubledot-center-%s' % scanjob['td']['name']
                    saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)
    
                    ptI, resultsfineI = analyse2dot(alldatac, fig=300, efig=None, istep=1)
    
                    # re-calc position
                    dx=resultsfineI['ptmv'].flatten()-np.array( [(scanjob['sweepdata']['start']+scanjob['sweepdata']['end'])/2, (scanjob['stepdata']['start']+scanjob['stepdata']['end'])/2 ] )
                    dx=np.linalg.norm(dx)
                    niter=niter+1
    
            if 0:
                print('fast scan without compensation!')
    
                # initialize sensing dot to centre
                #sd.sdval[1]=(sdstart+sdend)/2
                #scanjobx=copy.copy(scanjob)
                sd.initialize(setPlunger=True)
    
                scanjob['compensateGates']=[]
                #STOP
                alldata, data = scan2Dfastjob(scanjob, TitleComment='scan double-dot', wait_time=wait_time, activegates=defaultactivegates())
                alldata['sd']=str(sd)
    
                if 0:
                    alldataslow, data = scan2Djob(scanjob, TitleComment='scan double-dot', wait_time=wait_time, activegates=defaultactivegates())
                    dstr='doubledot-slow-%s' % scanjob['td']['name']
                    saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)
    
                pt, resultsfine = analyse2dot(alldata, fig=300, efig=None, istep=1)
    
                if 1:
                    scanjobc=positionScanjob(scanjob, resultsfine['ptmv'])
                    alldatac, data = scan2Dfastjob(scanjobc, TitleComment='scan double-dot', wait_time=wait_time, activegates=defaultactivegates())
                    dstr='doubledot-center-%s' % scanjob['td']['name']
                    saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)
    
                dstr='doubledot-%s' % scanjob['td']['name']
                saveExperimentData(outputdir2d, alldata, tag='doubledot', dstr=dstr)
                #saveExperimentData(datadir, alldata, tag='doubledotscans')  # save generic double-dot scans


#%%
if __name__=='__main__':

    timecomplete=str(datetime.datetime.now())
    qtt.legacy.writeBatchData(outputdir, tag, timestart, timecomplete)

#%%

#%% Done

if __name__=='__main__':

    print('##### 1dot_script_batch: measurements done...')
    
    closeExperiment(station)
    gates.get_all()

#%% Make reports
import webbrowser

if __name__=='__main__':

    one_dots=get_one_dots(full=2)
    two_dots=get_two_dots(full=1)
    
    Tvalue=Tvalues[0]
    tag=basetag+'-T%d'  % Tvalue
    tag2d=basetag+'-T%d-sd%d'  % (Tvalue, sdid)

    resultsdir=qtt.mkdirc(os.path.join(datadir, tag) )
    resultsdir2d=qtt.mkdirc(os.path.join(datadir, tag2d) )

    try:
        # generate report
        fname,_= generateDoubleDotReport(two_dots, resultsdir2d, tag=tag2d, sdidx=sdid)
        webbrowser.open(fname,new=2)
    except Exception as e:
        print(e)
        pass
    
    
    print('##### 1dot_script_batch: generation of double-dot report complete...')

#%%
if __name__=='__main__':

    end(noerror=True)