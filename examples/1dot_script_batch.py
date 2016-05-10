""" Main script to perform automatic tuning of quantum dots

   Pieter Eendebak <pieter.eendebak@tno.nl>

"""    
#%% Import the modules used in this program:

from imp import reload
import sys
import logging
import matplotlib.pyplot
matplotlib.pyplot.ion()

import webbrowser
import datetime

import qtt

import virtualV2 as setup
setup.initialize(reinit=False, server_name='test2')

def simulation():
    ''' Funny ... '''
    return False
    
#%% Make instances available

#Amp = 10    # some magic factor 5
#datadir=experimentdata.getDataDir()

station = setup.getStation()

keithley1 = station.keithley1
keithley3 = station.keithley3

station.set_measurement(keithley3.amplitude)


        
#%% Define 1-dot combinations

verbose=2   # set output level of the different functions

full=1      # for full=0 the resolution of the scans is reduced, this is usefull for quick testing
one_dots=setup.get_one_dots(sdidx=[])
#one_dots=one_dots[0:1]; full=0
#one_dots=one_dots[1:3]; full=0
full=0

sdindices=[1,2]
sddots=setup.get_one_dots(sdidx=sdindices)[-2:]

#one_dots=[one_dots[0], one_dots[-1] ];
full=0
dohires=1

logging.info('## 1dot_script_batch: scan mode: full %d, scanning %d one-dots' % (full, len(one_dots)) )

timestart=str(datetime.datetime.now())

#%%

sdgates = qtt.flatten([s['gates'] for s in sddots])
activegates=['T'] + list(qtt.flatten([s['gates'] for s in sddots]))

basevalues=dict()
for g in activegates:
    basevalues[g]=0
    

basetag='batch-04032016'; Tvalues=np.array([-380])    


#basetag='batch-16102015'; Tvalues=np.array([-390])    

#b=False
b=True

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
    basevalues['SD1a']=0
    basevalues['SD1b']=0
    basevalues['SD1c']=0
    
    basevalues['SD2a']=0
    basevalues['SD2b']=0
    basevalues['SD2c']=0
    

if 1:
    Pbias=-80
    basevalues['P1']=Pbias
    basevalues['P2']=Pbias
    basevalues['P3']=Pbias
    basevalues['P4']=Pbias

#one_dots=one_dots[-1:]; Tvalues=np.array([-350])

#%% Do measurements

cache=1
measureFirst=True    # measure 1-dots
measureSecond=True   # measure 2-dots

#measureFirst=0;
#measureSecond=0

# if we are working offline we cannot measure, but only process results
if simulation():
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

# define sensing dot
sdval=getSDval(Tvalues[0], ggsd)
sd=sensingdot_t(ggsd, sdval, index=sdid  )


sd2=None
    
#gg=['SD2%s' % c for c in ['a','b','c']]; RFfreq=86.4e6; # SD2

#print('FIXME: refactor code to extract scan of single-dot' )


if full:    
    hiresstep=-2
else:
    hiresstep=-4



#%% One-dot measurements

def onedotScan(od, basevalues, outputdir, verbose=1):
    if verbose:
        print('onedotScan: one-dot: %s' % od['name'] )
    gg=od['gates']
    keithleyidx=[ od['keithley'] ]
    
    #if od['keithley']==1:
    #    RFsiggen1.on()

    set_gate( gg[1], basevalues[gg[1]]-0 )     # plunger        
     
    pv1=od['pinchvalues'][0]+0
    pv2=od['pinchvalues'][2]+0
    stepstart=np.minimum( od['pinchvalues'][0]+400, 90)
    sweepstart=np.minimum( od['pinchvalues'][2]+400, 90)
    stepdata=dict({'gates': [gg[0]], 'start': stepstart, 'end': pv1-10, 'step': -6})
    sweepdata=dict({'gates': [gg[2]], 'start': sweepstart, 'end': pv2-10, 'step': -6})
    
    if full==0:
        stepdata['step']=-12; sweepdata['step']=-12
        #stepdata['step']=-6; sweepdata['step']=-6


    scanjob=dict({'stepdata':stepdata, 'sweepdata':sweepdata, 'keithleyidx': keithleyidx}) 
    alldata,data=scan2Djob(scanjob, TitleComment='2D scan', wait_time=.05)

    od, ptv, pt,ims,lv, wwarea=onedotGetBalance(od, alldata, verbose=1, fig=None)
                
    #basename='%s-sweep-2d' % (od['name'])
    alldata['od']=od
    return alldata, od
    

#%%

    
for ii, Tvalue in enumerate(Tvalues):
    tag=basetag+'-T%d'  % Tvalue    
    outputdir=mkdirc(os.path.join(datadir, tag))
    if not measureFirst:
        continue;

    gates.set_T(Tvalue)       # set T value
    
    experimentdata.initExperimentTwo(topgate=Tvalue)

    #%% Main loop
            
    print('## start of scan for topgate %.1f [mV]: tag %s' % (Tvalue, tag) )
    tmp=mkdirc(os.path.join(outputdir, 'one_dot'));
       
    
    print('we have %d one-dots' % len(one_dots) )
    for ii, od in enumerate(one_dots):
        print('  gates: %s' % str(od['gates']) )
        print('     channel: %s (keithley %d)' % (str(od['channel']), od['keithley']) )
    
    print('active gates: %s' % str(activegates))
    

    #%% Initialize to default values
    
    print('## 1dot_script: initializing gates')
    resetgates(activegates, basevalues)
    resetgates(sdgates, basevalues)
    

    #%% Perform sanity check on the channels

    for gate in ['L', 'D1', 'D2', 'D3', 'R']+['P1','P2','P3','P4']: # ,'SD1a', 'SD1b', ''SD2a','SD]:
        scanPinchValue(outputdir, gate, basevalues=basevalues, keithleyidx=[3], cache=cache, full=full)
    for od in sddots:
        ki=od['keithley']
        for gate in od['gates']:
            scanPinchValue(outputdir, gate, basevalues=basevalues, keithleyidx=[ki], cache=cache, full=full)
            resetgates(activegates, basevalues, verbose=0)

    ww=one_dots            
    for od in ww:      
        print('getting data for 1-dot: %s' % od['name'] )
        
        od = onedotScanPinchValues(od, basevalues, outputdir, cache=cache, full=full)
    
            #break
    
    #%% Re-calculate basevalues
    # todo: place this in function 
    basevaluesS=copy.deepcopy(basevalues)
    for g in ['L', 'D1', 'D2', 'D3', 'R']:
            basename = pinchoffFilename(g, od=None)
            pfile=os.path.join(outputdir, 'one_dot', basename +'.pickle')
            alldata,=pmatlab.load(pfile);  # alldata=alldata[0]
            adata=analyseGateSweep(alldata, fig=None, minthr=None, maxthr=None, verbose=1)
            
            basevaluesS[g]=float(min(adata['pinchvalue']+500, 0))
    
    #%% Analyse the one-dots
    dotlist=one_dots
    for od in dotlist:
        od = loadOneDotPinchvalues(od, outputdir, verbose=1)
    
    # Make scans of the sensing dots
    
    for odii, od in enumerate(sddots):
        
        resetgates(activegates, basevaluesS)        

        basename='%s-sweep-2d' % (od['name'])
        basenameplunger='%s-sweep-plunger' % (od['name'])
        efileplunger=experimentFile(outputdir, tag='one_dot', dstr=basenameplunger)
        if cache and os.path.exists(efileplunger) and 1:
            print('  skipping 2D and plunger scan of %s'  % od['name' ])
            #alldata=loadExperimentData(outputdir, tag='one_dot', dstr=basename)
            #d=alldata['od']
            continue

        od = loadOneDotPinchvalues(od, outputdir, verbose=1)
        alldata, od = onedotScan(od, basevaluesS, outputdir, verbose=1)
        
        alldatahi, od=onedotHiresScan(od, dv=70, verbose=1)
        saveExperimentData(outputdir, alldatahi, tag='one_dot', dstr='%s-sweep-2d-hires' % (od['name']))

        saveExperimentData(outputdir, alldata, tag='one_dot', dstr=basename)
        
        alldataplunger=onedotPlungerScan(od, verbose=1)
        saveExperimentData(outputdir, alldataplunger, tag='one_dot', dstr=basenameplunger)

        basenamedot='%s-dot' % (od['name'])
        saveExperimentData(outputdir, od, tag='one_dot', dstr=basenamedot)

        resetgates(od['gates'], basevaluesS)        

    # update basevalues with settings for SD
    
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
        
        resetgates(activegates, basevaluesS)        
        
        basename='%s-sweep-2d' % (od['name'])
        basenameplunger='%s-sweep-plunger' % (od['name'])
        efileplunger=experimentFile(outputdir, tag='one_dot', dstr=basenameplunger)
        if cache and os.path.exists(efileplunger) and 1:
            print('  skipping 2D and plunger scan of %s'  % od['name' ])
            continue

        alldata,od = onedotScan(od, basevaluesS, outputdir, verbose=1)                    
        saveExperimentData(outputdir, alldata, tag='one_dot', dstr=basename)
 

        #%% Make high-resolution scans
        if dohires:
            alldatahi, od=onedotHiresScan(od, dv=70, verbose=1)            
            saveExperimentData(outputdir, alldatahi, tag='one_dot', dstr='%s-sweep-2d-hires' % (od['name']))

            #saveExperimentData(outputdir, alldata, tag='one_dot', dstr=basename) # needed?
    
        alldataplunger=onedotPlungerScan(od, verbose=1)
        saveExperimentData(outputdir, alldataplunger, tag='one_dot', dstr=basenameplunger)
            
        #saveCoulombData(datadir, alldataplunger)
    
    saveExperimentData(outputdir, basevaluesS, tag='one_dot', dstr='basevaluesS')
    

#STOP


#%% Reports for one-dots

datadir=experimentdata.getDataDir()
#one_dots=get_one_dots(full=2)

plt.close('all')
for ii, Tvalue in enumerate(Tvalues):
    tag=basetag+'-T%d'  % Tvalue  
    
    resultsdir=mkdirc(os.path.join(datadir, tag) )
    xdir=os.path.join(resultsdir, 'one_dot')
    
    try:
        print('script: make report for %s' % tag )
        fname=generateOneDotReport(one_dots+sddots,xdir, resultsdir)
        webbrowser.open(fname, new=2)
    except Exception as e:
        print('failed to generate one-dot report')
        print(e)
        pass
    
if len(Tvalues)>1:
    raise Exception(' not supported...' )
    
#%% Measurements for double-dots

basevaluesS=loadExperimentData(outputdir, tag='one_dot', dstr='basevaluesS')
basevalues0=copy.copy(basevaluesS)

#sdid=1
ggsd=['SD%d%s' % (sdid,c) for c in ['a','b','c']];
sdval=[ basevaluesS[g] for g in ggsd]
sd=sensingdot_t(ggsd, sdval, index=sdid  )


#cache=0

#end()
#measureSecond=False

    
for ii, Tvalue in enumerate(Tvalues):
    if not measureSecond:      
        break;
    print('### 1dot_script_batch: double-dot scans for T=%.1f' % Tvalue)
    
    tag=basetag+'-T%d'  % (Tvalue)
    tag2d=basetag+'-T%d-sd%d'  % (Tvalue, sdid)
    
    gates.set_T(Tvalue)       # set T value
    
    outputdir=mkdirc(os.path.join(datadir, tag))
    outputdir2d=mkdirc(os.path.join(datadir, tag2d))

    experimentdata.initExperimentTwo(topgate=Tvalue)
    #RFsiggen1.set_frequency(RFfreq)   # SD2
    stop_AWG()
    #RFsiggen1.off()
    measurementfunctions.stopbias()
    #RFsiggen1.on()  # for sensing dot

    instrumentStatus()

    #readfunc=lambda: keithley1.readnext()*(1e12/(Amp*10e6) )
    
    two_dots=get_two_dots(full=1)
    one_dots=get_one_dots(full=1)
    
    # make two-dots (code from optimize_one)
    jobs = createDoubleDotJobs(two_dots, one_dots, basevalues=basevalues0, resultsdir=outputdir, fig=None)
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
        xfile=experimentFile(outputdir2d, alldata, tag='doubledot', dstr=dstr)
        if cache and os.path.exists(xfile):
            pass
            continue
            
        basevaluesTD=copy.copy(scanjob['basevalues'])
        resetgates(activegates, basevaluesTD)
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
         
        if 1:
            # calculate settings for gate compensation
            g=sd.tunegate()
            (sdstart, sdend, sdmiddle)=sd.autoTuneFine(scanjob=scanjob, fig=300)        
            #(sdstart, sdend, tmp) = autoTuneCompensation(sd, scanjob, fig=300)
 
            compensateGates=[g]
            gate_values_corners=[[sdstart, sdstart, sdend, sdend]]
            sd.sdval[1]=(sdstart+sdend)/2
            
        scanjob['compensateGates']=compensateGates
        scanjob['gate_values_corners']=gate_values_corners    
        set_gate(sweepdata['gates'][0], sweepdata['start'])
        

        #%% Scan in fast mode...
        scanjob['Naverage']=160
        gstep, gsweep, center, d,voltages_step = fastScan(stepdata, sweepdata)
        wait_time=None
        wait_time=0 # try
        
    
        if 1:
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
timecomplete=str(datetime.datetime.now())
measurementfunctions.writeBatchData(outputdir, tag, timestart, timecomplete)

#%%
 
#%% Done
print('##### 1dot_script_batch: measurements done...')

experimentdata.closeExperiment()
gates.get_all()
   
#%% Make reports
import experimentdata
import webbrowser

datadir=experimentdata.getDataDir()
one_dots=get_one_dots(full=2)
two_dots=get_two_dots(full=1)

for ii, Tvalue in enumerate(Tvalues):
    tag=basetag+'-T%d'  % Tvalue  
    tag2d=basetag+'-T%d-sd%d'  % (Tvalue, sdid)

    resultsdir=mkdirc(os.path.join(datadir, tag) )
    resultsdir2d=mkdirc(os.path.join(datadir, tag2d) )
        
    try:
        # generate report    
        fname,_= generateDoubleDotReport(two_dots, resultsdir2d, tag=tag2d, sdidx=sdid)
        webbrowser.open(fname,new=2)
    except Exception as e:
        print(e)
        pass
    
    
print('##### 1dot_script_batch: generation of double-dot report complete...')

#%%

end(noerror=True)    