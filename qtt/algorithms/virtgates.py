# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 18:08:57 2017

@author: houckm
"""
import time
start_time=time.clock()

import numpy
import numpy as np
from time import gmtime, strftime
np.set_printoptions(precision=2,suppress=True, threshold=100)


import matplotlib.pyplot as plt

import qtt
from users.houckm.detect_peaks import detect_peaks

from qtt.algorithms.generic import smoothImage

import qtt.instrument_drivers.virtual_gates
import qtt.measurements.videomode
import qtt.structures
from qtt.data import makeDataSet1Dplain
from qtt.measurements.scans import create_vectorscan

from qtt.measurements.ttrace import dummy_read,parse_data

#%%
def scan1D_transitions(tlines, CC, station, virt_gates ,physical_gates, cc_basis, scanjob,  abort=False, mode='real', single_scan1D='off', iteration=1, wait_time_sweep1D=1.2, Naverage=1000, rang=40, nsteps=200, dGstep=5):#TODO:Question: What does the wait_time_sweep1D do? Shouldn't it be a sleep time between the two steps?
#    How to bring a board qtt.abort_measurements()
    """ Makes 1D sweep scans over different transition lines     
    Inputs:
        tlines (dict): containing the points on the different transitionlines chosen to sweep over
        CC (dict): containing all the scanned data
        station (station): station containing parameters needed for the scan
        virt_gates (virtual_gates instrument): virtual gates used in the scans
        physical_gates (list): containing all the physical gates in the setup
        cc_basis (list): containing all the virtual gates in the setup
        scanjob (scanjob_t): information needed to do the scans
        abort (Bool): parameter to abort #TODO: does not work yet
        mode (string): if you are using af real or simulated setup
        single_scan1D (string): if 'on' only 2 1D scans are made
        iteration (float): the number keeping track to of the updates used to store in CC
        wait_time_sweep1D (float): wait time before each sweep
        Naverage (float): strength of the averaging filter
        rang (float): range of each sweep to scan
        nsteps (float): number of steps done in the scan
        dGstep (float): stepsize between two sweep scans
    Outputs:
        CC (dict): containing all the data
       """     
    gates=station.gates
    startgv=gates.allvalues()
    step=rang/nsteps
    for h,l in enumerate(cc_basis,0):
        i=2*h+1
        sweepgate=cc_basis[i]
        CC['iteration %d' %iteration]['sweepgate %s'%cc_basis[i]]={}
        transitionline=cc_basis[i]
        gg=list(tlines[transitionline].keys())
        startgv[gg[0]]=tlines[transitionline][gg[0]]
        startgv[gg[1]]=tlines[transitionline][gg[1]]
#        if  iteration>1:#TODO: Fix that this auto centering works right
#            startgv[physical_gates[i]]=CC['iteration %d'%(iteration-1)]['sweepgate %s'%sweepgate]['stepgate tL']['avg_transitionplace_sweepgate']
#    
        gates.resetgates(startgv, startgv,verbose=0)
        startsweep = virt_gates.get(list(virt_gates.allvalues())[i])-rang/2
        if mode=='real':
            scanjob['sweepdata']=create_vectorscan(virt_gates.parameters[sweepgate], g_range=rang)
            default_parameter_array_key='measured1'
        if mode=='sim':
            scanjob['sweepdata']= dict({'param': virt_gates.parameters[sweepgate], 'start': startsweep, 'end': startsweep+rang, 'step': step, 'wait_time': wait_time_sweep1D})
        scanjob['Naverage'] = Naverage #attenuates noise, real?    
        for  j,k in enumerate(cc_basis,0):
            if i==j:
                pass
            else:
                gates.resetgates(startgv, startgv, verbose=0)
                virt_gates.set(k,(virt_gates.get(k)-dGstep))
                if mode=='real':
                    s1a=qtt.measurements.scans.scan1Dfast(station, scanjob)
                if mode=='sim':
                    s1a=qtt.measurements.scans.scan1D(station, scanjob, verbose=0,liveplotwindow=False) 
                s1=makeDataSet1Dplain(xname='sweepparam', x=(np.linspace(startsweep, (startsweep+rang), num=np.size(s1a.arrays[default_parameter_array_key]))),yname=default_parameter_array_key,y=s1a.arrays[default_parameter_array_key][:])

    
                gates.resetgates(startgv, startgv, verbose=0)
                virt_gates.set(k,(virt_gates.get(k)+dGstep))
                if mode=='real':
                    s2a=qtt.measurements.scans.scan1Dfast(station, scanjob)
                if mode=='sim':
                    s2a=qtt.measurements.scans.scan1D(station, scanjob, verbose=0,liveplotwindow=False)
                s2=makeDataSet1Dplain(xname='sweepparam', x=(np.linspace(startsweep, (startsweep+rang), num=np.size(s2a.arrays[default_parameter_array_key]))),yname=default_parameter_array_key,y=s2a.arrays[default_parameter_array_key][:])
                print(sweepgate,k)
                CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]=dict({'sweep1':dict({'scan':s1}), 'sweep2':dict({'scan':s2})})
                if abort==True or single_scan1D=='on':
                    break
                    raise Exception('Measurement aborted')
        if h==(len(cc_basis)-1)/2-1 or single_scan1D=='on' or abort==True: 
            break
    gates.resetgates(startgv, startgv, verbose=0)
    return(CC)
#%%
def ttrace_transitions(CC,station, virt_gates, cc_basis, normal_gates, biasT_gates, t_RC, t_biasT, ttraces,ttrace,amplitudes,abort=False, single_scan1D='off', iteration=2, dGstep=5):#TODO: can the normal gates, biasT gates, and their wait times be extracted from the station?
    """Makes 1D ttraces over transition lines
    Inputs:
        CC (dict): containing all the scanned data
        station (station): station containing parameters needed for the scan
        virt_gates (virtual_gates instrument): virtual gates used in the scans
        cc_basis (list): containing all the virtual gates in the setup
        normal_gates (list): gates which don't have a biasT
        biasT_gates (list): gates which have a biasT
        t_RC: waiting time for normal gates (on the 3dot as a result of the low pass RC filter time)
        t_biasT: waiting time for gates with bias T(on the 3dot as a result of the bias T acting as an RC filter for DAC steps)
        ttraces,ttrace (dict): data on the traces put on the awg
        amplitudes: amplitudes of the traces
        abort (Bool): parameter to abort #TODO: does not work yet
        single_scan1D (string): if 'on' only 2 1D scans are made
        iteration (float): the number keeping track to of the updates used to store in CC
        dGstep (float): stepsize between two sweep scans
    Outputs:
        CC (dict): containing all the data"""
    start_time=time.clock()
    gates=station.gates
    startgv=gates.allvalues()
    default_parameter_array_key='measured1'
    
    CC['iteration %d' %iteration]=dict({})
    for aa,bb in enumerate(cc_basis):
        if aa%2!=0 :
            CC['iteration %d' %iteration]['sweepgate %s'%bb]=dict({})           
    
    for  j,k in enumerate(cc_basis,0):
        gates.resetgates(startgv, startgv, verbose=0)
        
        virt_gates.set(k,(virt_gates.get(k)-dGstep))
        if k in normal_gates:
            time.sleep(t_RC)
        else:
            time.sleep(t_biasT)
        
        data_raw=dummy_read(station)
        tt, datax, s1a = parse_data(data_raw,ttraces,ttrace)
            
        virt_gates.set(k,(virt_gates.get(k)+2*dGstep))
        if k in normal_gates:
            time.sleep(t_RC)
        else:
            time.sleep(t_biasT)
        data_raw=dummy_read(station)
        tt, datax, s2a = parse_data(data_raw,ttraces,ttrace)
        print(k)   
        
        for h,l in enumerate(cc_basis,0):
            i=2*h+1
            sweepgate=cc_basis[i]
            if i==j:
                pass
            else:
                rang=amplitudes[h]
                startsweep=virt_gates.get(sweepgate)-rang/2
                s1=makeDataSet1Dplain(xname='sweepparam', x=(np.linspace(startsweep, (startsweep+rang), num=np.size(s1a[h],1))),yname=default_parameter_array_key,y=s1a[h][0])
                s2=makeDataSet1Dplain(xname='sweepparam', x=(np.linspace(startsweep, (startsweep+rang), num=np.size(s2a[h],1))),yname=default_parameter_array_key,y=s2a[h][0])
                CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]=dict({'sweep1':dict({'scan':s1}), 'sweep2':dict({'scan':s2})})
            if h==(len(cc_basis)-1)/2-1: 
                break 
        if  single_scan1D=='on':
            break  
    gates.resetgates(startgv, startgv, verbose=0)
    ttraceC=round(time.clock() - start_time,2)
    print("ttraceC time%s" % ttraceC)  
    return(CC)
#%%
def cc_from_scan1D(CC, cc_basis, dGstep, datadir_CC, iteration=1,virt_gates_iter=0,mode='real',mph=0.5,mpd=80, single_scan1D='off',valley=True):
    """Computes cross-capacitance matrix from 1D transitions scans
    Inputs:
        CC (dict): containing all the scanned data
        cc_basis (list): containing all the virtual gates in the setup
        dGstep (float): stepsize between two sweep scans
        datadir_CC (string): place to store the output
        iteration (float): the number keeping track to of the updates used to store in CC
        virt_gates_iter (float): iteration on which the used virtual gates were based
        mode (string): if you are using af real or simulated setup
        mph (float): relative minimum peak height in the peak detecton
        mpd (float): amount of data points between allowed peaks in the peak detection
        single_scan1D (string): if 'on' only 2 1D scans are made
        valley (Bool): if True valley peaks are detected, if False peaks are detected
    Outputs:
        CC (dict): array containing the scanned and analysed data as well
        """
    default_parameter_array_key='measured1'
    meas_arr_name=CC['iteration %d'%iteration]['sweepgate mu1']['stepgate tL']['sweep1']['scan'].default_parameter_name(default_parameter_array_key)
    c=np.zeros((len(cc_basis),len(cc_basis)))
    np.fill_diagonal(a=c,val=1)
    for h,l in enumerate(cc_basis,0):
        i=2*h+1 
        sweepgate=cc_basis[i]
        if mode=='real':
            setpoints='sweepparam'
        if mode=='sim':
            setpoints=sweepgate
        for  j,k in enumerate(cc_basis,0):
            if i==j:
                c[i,j]=1
            else:
                s=np.zeros((2,10))
                
                for ss in ['sweep1','sweep2']:
                    
                    scan=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k][ss]['scan'] #load
                    
                    ds=qtt.diffDataset(scan,'y',None, meas_arr_name)
                    ds.arrays['diff_dir_y'].ndarray=smoothImage(ds.arrays['diff_dir_y'].ndarray)
                    if 0:
                        if mode=='real':
                            peaks_i=numpy.argmin(ds.diff_dir_y,axis=0)
                                               
                        if mode=='sim':
                            peaks_i=numpy.argmax(ds.diff_dir_y,axis=0)
                    if 1:
                        if mode=='real':
                            transition=True
                            polarisation=False
                        if mode=='sim':
                            transition=False
                            polarisation=True
                        peaks_i=detect_peaks(ds.diff_dir_y,mph=mph*max(ds.diff_dir_y),mpd=mpd,show=False, valley=valley)
                        peaks_sweep=scan.arrays[setpoints][peaks_i]
                        peaks_a=scan.default_parameter_array(default_parameter_array_key)[peaks_i]
                        peaks_da=ds.diff_dir_y[peaks_i]
                        print(peaks_i,peaks_sweep,peaks_a,peaks_da)
                    CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k][ss]['analysis']=dict({'ds':ds,'peaks_i':peaks_i,'peaks_sweep':peaks_sweep,'peaks_a':peaks_a,'peaks_da':peaks_da}) 
#             
                peaks_i_1=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]['sweep1']['analysis']['peaks_i']
                peaks_i_2=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]['sweep2']['analysis']['peaks_i']
                scan=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]['sweep2']['scan']
                
                transition_exp=len(scan.arrays[setpoints])/2
                peaks_cc_i_1=peaks_i_1.flat[np.abs(peaks_i_1 - transition_exp).argmin()]

                
                if virt_gates_iter==0:#gets for the second point nearest peak
                     peaks_cc_i_2=peaks_i_2[numpy.argmax([n for n in (peaks_i_2 - peaks_cc_i_1) if n<0])]
                else:
                    peaks_cc_i_2=peaks_i_2.flat[np.abs(peaks_i_2 - peaks_cc_i_1).argmin()]

                for ss in ['sweep1','sweep2']: #storage   
                    scan=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k][ss]['scan'] #load
                    ds=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k][ss]['analysis']['ds'] #load
                    peaks_i=(peaks_cc_i_1 if ss=='sweep1' else peaks_cc_i_2)
                    peaks_sweep=scan.arrays[setpoints][peaks_i]
                    peaks_a=scan.default_parameter_array(default_parameter_array_key)[peaks_i]
                    peaks_da=ds.diff_dir_y[peaks_i]
                    print(peaks_i,peaks_sweep,peaks_a,peaks_da)
                    CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k][ss]['analysis']['for_cc']=dict({'peaks_i':peaks_i,'peaks_sweep':peaks_sweep,'peaks_a':peaks_a,'peaks_da':peaks_da}) 
                
                peaks_cc_sweep_1=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]['sweep1']['analysis']['for_cc']['peaks_sweep']
                peaks_cc_sweep_2=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]['sweep2']['analysis']['for_cc']['peaks_sweep']
                
                dGsweep=peaks_cc_sweep_1-peaks_cc_sweep_2
                avg_transitionplace_sweepgate=(peaks_cc_sweep_1+peaks_cc_sweep_2)/2                    
                c[i,j]=dGsweep/(2*dGstep)
                print(sweepgate,k,c[i,j])
                #storage:
                CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]['avg_transitionplace_sweepgate']=avg_transitionplace_sweepgate
                if single_scan1D=='on':
                    break
                    raise Exception('fout...')
        if h==(len(cc_basis)-1)/2-1 or single_scan1D=='on': 
            break
    CC['iteration %d' %iteration]['cc_corr']=c
    CC['iteration %d' %iteration]['cc']=c@CC['iteration %d' %(virt_gates_iter)]['cc']
    print(CC['iteration %d' %iteration]['cc'])
    
    s=strftime('%d_%m_%H%M', gmtime())
    qtt.write_data(datadir_CC%s,CC)
    return(CC)

#%%
def plot_scan1D_transitions(CC, cc_basis, title_add=None, iteration=1,mode='real',single_scan1D='off', analyses=True):
    """Plots the 1D transitions scans and analysis
    Input:
        CC (dict): containing all the scanned data
        cc_basis (list): containing all the virtual gates in the setup
        title_add (string): addition on the plots title to make
        iteration (float): the number keeping track to of the updates used to store in CC
        mode (string): if you are using af real or simulated setup
        single_scan1D (string): if 'on' only 2 1D scans are made
        analyses (Bool): plot the analyses yes or no
        """
    default_parameter_array_key='measured1'
    if single_scan1D=='on':
        fig, axarr = plt.subplots(1,1)
        fig.suptitle('Transition step for %s' %title_add)
    else:
        fig, axarr = plt.subplots(int((len(cc_basis)-1)/2),int(len(cc_basis)))
        fig.suptitle('Differentiated transition step for %s' %title_add)
    if analyses==True:
        if single_scan1D=='on':
            figd, axarrd = plt.subplots(1,1)
            figd.suptitle('Transition step')
        else:
            figd, axarrd =  plt.subplots(int((len(cc_basis)-1)/2),int(len(cc_basis)))
            figd.suptitle('Transition step')
    for h,l in enumerate(cc_basis,0):
        i=2*h+1
        sweepgate=cc_basis[i]
        if mode=='real':
            setpoints='sweepparam'
        if mode=='sim':
            setpoints=sweepgate
        for  j,k in enumerate(cc_basis,0):
            if i==j:
                pass
            else:
                CCpart=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]
                for ss in ['sweep1', 'sweep2']:
                    if single_scan1D=='on':
                        axarr.set_title(k)
                        axarr.plot(CCpart[ss]['scan'].arrays[setpoints],CCpart[ss]['scan'].default_parameter_array(default_parameter_array_key)) 
                    else:
                        axarr[0,j].set_title(k)
                        axarr[h,j].plot(CCpart[ss]['scan'].arrays[setpoints],CCpart[ss]['scan'].default_parameter_array(default_parameter_array_key)) 
                    if analyses==True:                       
                        if single_scan1D=='on':
                            axarr.plot(CCpart[ss]['analysis']['for_cc']['peaks_sweep'],CCpart[ss]['analysis']['for_cc']['peaks_a'],'^m' , markersize=7)
                                                    
                            axarrd.set_title(k)
                            axarrd.plot(CCpart[ss]['analysis']['ds'].arrays[setpoints],CCpart[ss]['analysis']['ds'].diff_dir_y)
                            axarrd.plot(CCpart[ss]['analysis']['ds'].arrays[setpoints],CCpart[ss]['analysis']['ds'].diff_dir_y)
                            axarrd.plot(CCpart[ss]['analysis']['peaks_sweep'],CCpart[ss]['analysis']['peaks_da'],'^m' , markersize=7)
                       
                        else:
                            axarr[h,j].plot(CCpart[ss]['analysis']['for_cc']['peaks_sweep'],CCpart[ss]['analysis']['for_cc']['peaks_a'],'^m' , markersize=7)
                                                    
                            axarrd[0,j].set_title(k)
                            axarrd[h,j].plot(CCpart[ss]['analysis']['ds'].arrays[setpoints],CCpart[ss]['analysis']['ds'].diff_dir_y)
                            axarrd[h,j].plot(CCpart[ss]['analysis']['ds'].arrays[setpoints],CCpart[ss]['analysis']['ds'].diff_dir_y)
                            axarrd[h,j].plot(CCpart[ss]['analysis']['peaks_sweep'],CCpart[ss]['analysis']['peaks_da'],'^m' , markersize=7)
                if single_scan1D=='on':
                    break
                    raise Exception('fout...')
            if single_scan1D=='on': 
                break
        if h==(len(cc_basis)-1)/2-1 or single_scan1D=='on':
            break  
    qtt.tools.addPPTslide(fig=fig.number)
    if 'analysis' in CC['iteration %d'%iteration]['sweepgate mu1']['stepgate tL']['sweep1']:
        qtt.tools.addPPTslide(fig=figd.number)
