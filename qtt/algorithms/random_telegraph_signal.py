# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:20:46 2018

@author: riggelenfv
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import qcodes
from qtt.tools import addPPTslide
import warnings

from qtt.algorithms.functions import double_gaussian, fit_double_gaussian, exp_function, fit_exp_decay

#%% calculate durations of states
    
def transitions_durations(data, split):
    """ For data of a two level system (up and down), this funtion determines which datapoints belong to which level and finds the transitions, in order to determines 
    how long the system stays in these levels.
    
    Args:
        data (numpy array): data from the two level system
        split (float): value that separates the up and down level
    
    Returns: 
        duration_dn (numpy array): array of the durations (unit: data points) in the down level
        duration_up (numpy array): array of durations (unit: data points) in the up level
    """
    
    # split the data and find the index of the transitions, transitions from up to down are marked with -1 and from down to up with 1
    b = data > split
    d=np.diff(b.astype(int))
    transitions_dn = (d==-1).nonzero()[0]
    transitions_up = (d==1).nonzero()[0]
    
    # durations are calculated by taking the difference in data points between the transitions, first and last duration are ignored
    if data[0] <= split and data[-1] <= split:
        duration_up = transitions_dn-transitions_up
        duration_dn = transitions_up[1:]-transitions_dn[:-1]
        
    elif data[0] < split and data[-1] > split:
        duration_up = transitions_dn-transitions_up[:-1]
        duration_dn = transitions_up[1:]-transitions_dn
    
    elif data[0] > split and data[-1] < split:
        duration_up = transitions_dn[1:]-transitions_up
        duration_dn = transitions_up-transitions_dn[:-1]
        
    elif data[0] >= split and data[-1] >= split:
        duration_up = transitions_dn[1:]-transitions_up[:-1]
        duration_dn = transitions_up-transitions_dn

    return duration_dn, duration_up

#%% function to analyse the RTS data
class FittingException(Exception):
    pass    
    
def tunnelrates_RTS(data, samplerate = None, min_sep = 2.0, max_sep = 7.0, min_duration = 5, plungers = [], fig = None, ppt = None, verbose = 0):
    """
    This function takes an RTS dataset, fits a double gaussian, finds the split between the two levels, 
    determines the durations in these two levels, fits a decaying exponantial on two arrays of durations, 
    which gives the tunneling frequency for both the levels.
    
    Args:
        data (array): qcodes DataSet (or 1d data array) with the RTS data
        plungers ([str, str]): array of the two plungers used to perform the RTS measurement
        samplerate (int or float): sampling rate of either the fpga or the digitizer, optional if given in the metadata
                of the measured data
        min_sep (float): if the separation found for the fit of the double gaussian is less then this value, the fit probably failed 
            and a FittingException is raised
        max_sep (float): if the separation found for the fit of the double gaussian is more then this value, the fit probably failed
            and a FittingException is raised
        min_duration (int): minimal number of datapoints a duration should last to be taking into account for the analysis
        fig (None or int): shows figures and sends them to the ppt when is not None
        ppt (None or int): determines if the figures are send to a powerpoint presentation
        verbose (int): prints info to the console when > 0
        
    Returns:
        tunnelrate_dn (numpy.float64): tunneling rate of the down level (kHz)
        tunnelrate_up (numpy.float64): tunneling rate of the up level (kHz)
        parameters (dict): dictionary with relevent (fit) parameters
        
    """
    
    if type(data) == qcodes.data.data_set.DataSet:
        try:
            plungers = plungers
            metadata = data.metadata
            gates = metadata['allgatevalues']
            plungervalue = gates[plungers[0]]
        except:
            plungervalue = []
        if samplerate is None:
            metadata = data.metadata
            samplerate = metadata['samplerate']
        
        data = np.array(data.measured)
    else:
        plungervalue = []   
        
    # plotting a 2d histogram of the RTS
    if fig:     
        xdata = np.array(range(0,len(data)))/samplerate*1000
        Z, xedges, yedges = np.histogram2d(xdata, data, bins=[int(np.sqrt(len(xdata)))/2,int(np.sqrt(len(data)))/2])
        title = '2d histogram RTS'
        plt.figure(title)
        plt.pcolormesh(xedges, yedges, Z.T)
        cb = plt.colorbar()
        cb.set_label('Data points per bin')
        plt.xlabel('Time (ms)')
        plt.ylabel('Signal sensing dot (a.u.)')
        plt.title(title)
        if ppt:
            addPPTslide(title=title, fig=plt.figure(title))
        
    # binning the data and determining the bincentres
    num_bins = int(np.sqrt(len(data)))
    counts, bins = np.histogram(data, bins=num_bins)
    bincentres = np.array([(bins[i] + bins[i+1])/2 for i in range(0, len(bins)-1)])
    
    # fitting the double gaussian and finding the split between the up and the down state, separation between the max of the two gaussians measured in the sum of the std
    par_fit, result_dict = fit_double_gaussian(bincentres, counts)
    separation = result_dict['separation']
    split = result_dict['split']
     
    if verbose:
        print('Fit paramaters double gaussian:\n mean down: %.3f counts' % par_fit[4] +', mean up:%.3f counts' % par_fit[5] + ', std down: %.3f counts' % par_fit[2] +', std up:%.3f counts' % par_fit[3])
        print('Separation between peaks gaussians: %.3f std' % separation)
        print('Split between two levels: %.3f' % split)
        
    # plotting the data in a histogram, the fitted two gaussian model and the split
    if fig:
        title = 'Histogram of two levels RTS'
        plt.figure(title)
        counts, bins, _ = plt.hist(data, bins=num_bins)
        plt.plot(bincentres, double_gaussian(bincentres, par_fit), 'r', label = 'Fitted double gaussian')
        plt.plot(split, double_gaussian(split, par_fit), 'ro', markersize=8, label = 'split: %.3f' % split)
        plt.xlabel('Measured value (a.u.)')
        plt.ylabel('Data points per bin')
        plt.legend()
        plt.title(title)
        if ppt:
            addPPTslide(title=title, fig=plt.figure(title), notes='Fit paramaters double gaussian:\n mean down: %.3f counts' % par_fit[4] +', mean up:%.3f counts' % par_fit[5] + ', std down: %.3f counts' % par_fit[2] +', std up:%.3f counts' % par_fit[3] +'.Separation between peaks gaussians: %.3f std' % separation +'. Split between two levels: %.3f' % split )
        
    if separation < min_sep:
        raise FittingException('Separation between the peaks of the gaussian is less then %.1f std, indicating that the fit was not succesfull.' % min_sep)
        
    if separation > max_sep:
        raise FittingException('Separation between the peaks of the gaussian is more then %.1f std, indicating that the fit was not succesfull.' % max_sep)
   
    
    # count the number of transitions and their duration
    durations_dn_idx, durations_up_idx = transitions_durations(data, split)
    
    # throwing away the durations with less data points then min_duration
    durations_up_min_duration = durations_up_idx > min_duration
    durations_up = durations_up_idx[durations_up_min_duration]
    durations_dn_min_duration = durations_dn_idx > min_duration
    durations_dn = durations_dn_idx[durations_dn_min_duration]
    
    # calculating durations in seconds
    durations_dn= durations_dn/samplerate
    durations_up= durations_up/samplerate
    
    # sorting the durations
    durations_dn_srt = np.sort(durations_dn)
    durations_up_srt = np.sort(durations_up)
    
    # calculating the number of bins and counts for down level
    numbins_dn = int(np.sqrt(len(durations_dn_srt)))
    counts_dn, bins_dn = np.histogram(durations_dn_srt, bins=numbins_dn)
    
    if counts_dn[0] < 400:
        warnings.warn('Number of datapoints might not be enough to make an acurate fit of the exponantial decay for level down.')
        
    if counts_dn[0] < 50:
        raise FittingException('Number of datapoints is not be enough to make an acurate fit of the exponantial decay for level down.')
    
    bincentres_dn = np.array([(bins_dn[i] + bins_dn[i+1])/2 for i in range(0, len(bins_dn)-1)])
    
    #fitting exponantial decay for down level
    A_dn_fit, B_dn_fit, gamma_dn_fit = fit_exp_decay(bincentres_dn, counts_dn)
    tunnelrate_dn = gamma_dn_fit/1000
    
    if verbose:
        print('Tunnel rate down: %.1f kHz' % tunnelrate_dn)
    
    if fig:
        title = 'Fitted exponantial decay, level down'
        plt.figure(title)
        plt.plot(bincentres_dn, counts_dn, 'o', label='Counts down') 
        #plt.plot(bincentres_dn, exp_function(bincentres_dn,  B_dn_fit, A_dn_fit, gamma_dn_fit),'r', label='Fitted exponantial decay \n t_dn: %.1f kHz' % tunnelrate_dn)
        plt.plot(bincentres_dn, exp_function(bincentres_dn,  A_dn_fit, B_dn_fit, gamma_dn_fit),'r', label='Fitted exponantial decay \n t_dn: %.1f kHz' % tunnelrate_dn)
        plt.xlabel('Lifetime (s)')
        plt.ylabel('Counts per bin')
        plt.legend()
        plt.title(title)
        if ppt:
            addPPTslide(title=title, fig=plt.figure(title))
        
    # calculating the number of bins and counts for up level
    numbins_up = int(np.sqrt(len(durations_up_srt)))
    counts_up, bins_up = np.histogram(durations_up_srt, bins=numbins_up)
    
    if counts_up[0] < 400:
        warnings.warn('Number of datapoints might not be enough to make an acurate fit of the exponantial decay for level up.')
        
    if counts_dn[0] < 50:
        raise FittingException('Number of datapoints is not be enough to make an acurate fit of the exponantial decay for level down.')
    
    bincentres_up = np.array([(bins_up[i] + bins_up[i+1])/2 for i in range(0, len(bins_up)-1)])
    
    #fitting exponantial decay for up level
    A_up_fit, B_up_fit, gamma_up_fit = fit_exp_decay(bincentres_up, counts_up)
    tunnelrate_up = gamma_up_fit/1000
    
    if verbose:
        print('Tunnel rate up: %.1f kHz' % tunnelrate_up)
    
    if fig:
        title = 'Fitted exponantial decay, level up'
        plt.figure(title)
        plt.plot(bincentres_up, counts_up, 'o', label='Counts up') 
        #plt.plot(bincentres_up, exp_function(bincentres_up,  B_dn_fit, A_up_fit, gamma_up_fit),'r', label='Fitted exponantial decay \n t_up: %.1f kHz' % tunnelrate_up)
        plt.plot(bincentres_up, exp_function(bincentres_up,  A_dn_fit, B_up_fit, gamma_up_fit),'r', label='Fitted exponantial decay \n t_up: %.1f kHz' % tunnelrate_up)
        plt.xlabel('Lifetime (s)')
        plt.ylabel('Data points per bin')
        plt.legend()
        plt.title(title)
        if ppt:
            addPPTslide(title=title, fig=plt.figure(title))
        
    parameters = {'plunger value':plungervalue, 'sampling rate':samplerate, 'fit parameters double gaussian':par_fit, \
                                         'separations between peaks gaussians':separation, \
                                        'split between the two levels':split, 'fit parameters exp. decay down':[A_dn_fit, B_dn_fit, gamma_dn_fit], \
                                        'fit parameters exp. decay up':[A_up_fit, B_up_fit, gamma_up_fit]}
  
    return tunnelrate_dn, tunnelrate_up, parameters


#%%
def test_RTS():
    data = np.zeros( (100000,))
    for i in range(1, data.size):
        if np.random.rand()>.98:
            data[i]=1-data[i-1]
        else:
            data[i]=data[i-1]
    data=data+np.random.rand( 100000, )/5
    noise = np.random.normal(0, 0.1, data.size)
    rtsdata = data + noise
    r=tunnelrates_RTS(rtsdata, plungers=[], samplerate=10e6)
 
if __name__ == '__main__':
    test_RTS()
