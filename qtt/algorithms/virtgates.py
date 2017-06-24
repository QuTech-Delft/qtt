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
from qtt.algorithms.fitting import detect_peaks

from qtt.algorithms.generic import smoothImage

import qtt.instrument_drivers.virtual_gates
import qtt.measurements.videomode
import qtt.structures
from qtt.data import makeDataSet1Dplain
from qtt.measurements.scans import create_vectorscan
from qtt.measurements.ttrace import dummy_read,parse_data


#%%
def scan1D_transitions(tlines, CC,startgv, gg, station, gates, virt_gates ,physical_gates, cc_basis, scanjob,  abort=False, mode='real', single_scan1D='off', iteration=1, wait_time_sweep1D=1.2, Naverage=1000, rang=40, nsteps=200, dGstep=5):
    """ Makes 1D sweep scans over different transition lines     
    Question: What does the wait_time_sweep1D do? Shouldn't it be a sleep time between the two steps?
    How to brign a board qtt.abort_measurements()"""      
    step=rang/nsteps
    for h,l in enumerate(physical_gates,0):
        i=2*h+1
        sweepgate=cc_basis[i]
        CC['iteration %d' %iteration]['sweepgate %s'%cc_basis[i]]={}
        transitionline=cc_basis[i]
        startgv[gg[0]]=tlines[transitionline][gg[0]]
        startgv[gg[1]]=tlines[transitionline][gg[1]]
        if  iteration>1:
            startgv[physical_gates[i]]=CC['iteration %d'%(iteration-1)]['sweepgate %s'%sweepgate]['stepgate tL']['avg_transitionplace_sweepgate']
    
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
def ttrace_transitions(CC,startgv, station, gates, virt_gates, cc_basis, normal_gates, biasT_gates, t_RC, t_biasT, ttraces,ttrace,amplitudes,abort=False, single_scan1D='off', iteration=2, dGstep=5):
    """Makes 1D ttraces over transition lines"""
    start_time=time.clock()
    default_parameter_array_key='measured1'
    CC['iteration %d' %iteration]=dict({'sweepgate mu1':dict({}),'sweepgate mu2':dict({}),'sweepgate mu3':dict({})})
    
    for  j,k in enumerate(cc_basis,0):
        gates.resetgates(startgv, startgv, verbose=0)
#        virt_gates.set('mu1',(virt_gates.get('mu1')+50)) #update somewhere else in matrix
        
        virt_gates.set(k,(virt_gates.get(k)-dGstep))
        if k in normal_gates:
            time.sleep(t_RC)
        else:
            time.sleep(t_biasT)
        
        data_raw=dummy_read(station,ttraces[0:1])
        tt, datax, s1a = parse_data(station,data_raw,ttraces,ttrace)
            
        virt_gates.set(k,(virt_gates.get(k)+2*dGstep))
        if k in normal_gates:
            time.sleep(t_RC)
        else:
            time.sleep(t_biasT)
        data_raw=dummy_read(station,ttraces[0:1])
        tt, datax, s2a = parse_data(station,data_raw,ttraces,ttrace)
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
def cc_from_scan1D(CC,physical_gates, cc_basis, dGstep, datadir_CC, iteration=1,mode='real',mph=0.5,mpd=80, single_scan1D='off',analysis=True):
    """Computes cross-capacitance matrix from 1D transitions scans"""
    default_parameter_array_key='measured1'
    meas_arr_name=CC['iteration %d'%iteration]['sweepgate mu1']['stepgate tL']['sweep1']['scan'].default_parameter_name(default_parameter_array_key)
    c=np.zeros((len(physical_gates),len(physical_gates)))
    np.fill_diagonal(a=c,val=1)
    for h,l in enumerate(physical_gates,0):
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
#                        peaks_i=detect_peaks(ds.diff_dir_y,mph=0.1*max(ds.diff_dir_y),mpd=100,show=False, valley=polarisation)#best if 0.8
                        peaks_i=detect_peaks(ds.diff_dir_y,mph=mph*max(ds.diff_dir_y),mpd=mpd,show=False, valley=transition)#best if 0.8
                        peaks_sweep=scan.arrays[setpoints][peaks_i]
                        peaks_a=scan.default_parameter_array(default_parameter_array_key)[peaks_i]
                        peaks_da=ds.diff_dir_y[peaks_i]
                        print(peaks_i,peaks_sweep,peaks_a,peaks_da)
                    CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k][ss]['analysis']=dict({'ds':ds,'peaks_i':peaks_i,'peaks_sweep':peaks_sweep,'peaks_a':peaks_a,'peaks_da':peaks_da}) 
#                if peaks_i==None:
#                    break
#                
                peaks_i_1=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]['sweep1']['analysis']['peaks_i']#load
                peaks_i_2=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]['sweep2']['analysis']['peaks_i']#load
                scan=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%k]['sweep2']['scan']#load
                
                if j==0:            
                    transition_exp=len(scan.arrays[setpoints])/2
                else:
                    transition_exp=CC['iteration %d' %iteration]['sweepgate %s'%sweepgate]['stepgate %s'%cc_basis[0]][ss]['analysis']['for_cc']['peaks_i']
                peaks_cc_i_1=peaks_i_1.flat[np.abs(peaks_i_1 - transition_exp).argmin()]

                
                if iteration==0 or iteration==1:#gets for the second point nearest peak
#                    peaks_cc_i_2=peaks_i_2.flat[(peaks_i_2 - peaks_cc_i_1).argmin()]
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
    CC['iteration %d' %iteration]['cc']=c@CC['iteration %d' %(iteration-1)]['cc']
    print(CC['iteration %d' %iteration]['cc'])
    
    s=strftime('%d_%m_%H%M', gmtime())
    qtt.write_data(datadir_CC%s,CC)
    return(CC)
#%%
def plot_scan1D_transitions(CC,physical_gates, cc_basis, iteration=1,mode='real',single_scan1D='off'):
    """Plots the 1D transitions scans and analysis"""
    default_parameter_array_key='measured1'
    if single_scan1D=='on':
        fig, axarr = plt.subplots(1,1)
        fig.suptitle('Transition step')
    else:
        fig, axarr = plt.subplots(int((len(cc_basis)-1)/2),int(len(cc_basis)))
        fig.suptitle('Transition step')
    if 'analysis' in CC['iteration %d'%iteration]['sweepgate mu1']['stepgate tL']['sweep1']:
        if single_scan1D=='on':
            figd, axarrd = plt.subplots(1,1)
            figd.suptitle('Transition step')
        else:
            figd, axarrd =  plt.subplots(int((len(cc_basis)-1)/2),int(len(cc_basis)))
            figd.suptitle('Transition step')
    for h,l in enumerate(physical_gates,0):
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
                        axarr[h,j].set_title(k)
                        axarr[h,j].plot(CCpart[ss]['scan'].arrays[setpoints],CCpart[ss]['scan'].default_parameter_array(default_parameter_array_key)) 
                    if 'analysis' in CC['iteration %d'%iteration]['sweepgate mu1']['stepgate tL']['sweep1']:                       
                        if single_scan1D=='on':
                            axarr.plot(CCpart[ss]['analysis']['for_cc']['peaks_sweep'],CCpart[ss]['analysis']['for_cc']['peaks_a'],'^m' , markersize=7)
                                                    
                            axarrd.set_title(k)
                            axarrd.plot(CCpart[ss]['analysis']['ds'].arrays[setpoints],CCpart[ss]['analysis']['ds'].diff_dir_y)
                            axarrd.plot(CCpart[ss]['analysis']['ds'].arrays[setpoints],CCpart[ss]['analysis']['ds'].diff_dir_y)
                            axarrd.plot(CCpart[ss]['analysis']['peaks_sweep'],CCpart[ss]['analysis']['peaks_da'],'^m' , markersize=7)
                       
                        else:
                            axarr[h,j].plot(CCpart[ss]['analysis']['for_cc']['peaks_sweep'],CCpart[ss]['analysis']['for_cc']['peaks_a'],'^m' , markersize=7)
                                                    
                            axarrd[h,j].set_title(k)
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