# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:31:42 2017

@author: riggelenfv
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy
import qcodes
from qtt.tools import addPPTslide
from projects.tunnel_coupling_tuning.random_telegraph_signal.rts_functions import fit_exp_decay, exp_decay
import warnings
from projects.calibration_manager.calibration_master import StorageMaster

#%% functions to fit the double gaussian, adjusted version of Sjaak's code in rts_functions

def double_gaussian(signal_amp, params):
    """ A model for the sum of two Gaussian distributions. """
    [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up] = params
    gauss_dn = A_dn*np.exp(-(signal_amp-mean_dn)**2/(2*sigma_dn**2))
    gauss_up = A_up*np.exp(-(signal_amp-mean_up)**2/(2*sigma_up**2))
    double_gauss = gauss_dn + gauss_up
    return double_gauss

def cost_double_gaussian(signal_amp, counts, params):
    """ Cost function for fitting of double Gaussian. """
    model = double_gaussian(signal_amp, params)
    cost = np.linalg.norm(counts - model)
    return cost

def fit_double_gaussian(signal_amp, counts, maxiter=None, maxfun=5000, verbose=1, par_guess=None):
    """ Fitting of double gaussian, in this script applied to a distribution of measured data
    
    Args:
        signal_amp (array): x values of the data, when applied on a distrinution this should be the 
            values of the bin centers
        counts (array): y values of the data, or the counts per bin
        par_guess (None or array): optional, initial guess for the fit parameters: 
            [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]
            
    Returns:
        par_fit (array): fit parameters of the double gaussian: [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]
        par_guess (array): initial guess for the fit parameters, either the ones give to the function,
            or generated by the function: [A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up]
        
    """
    
    func = lambda params: cost_double_gaussian(signal_amp, counts, params)
    maxsignal = np.percentile(signal_amp,98)
    minsignal = np.percentile(signal_amp,2)
    if par_guess is None:
        A_dn = np.max(counts[:int((len(counts)/2))])
        A_up = np.max(counts[int((len(counts)/2)):])
        sigma_dn = (maxsignal-minsignal)*1/20
        sigma_up = (maxsignal-minsignal)*1/20
        mean_dn = minsignal + 1/4*(maxsignal - minsignal)
        mean_up = minsignal + 3/4*(maxsignal - minsignal)
        par_guess = np.array([A_dn, A_up, sigma_dn, sigma_up, mean_dn, mean_up])
    par_fit = scipy.optimize.fmin(func, par_guess, maxiter=maxiter, maxfun=maxfun, disp=verbose >= 2)
    
    return par_fit, par_guess

#%% calculate durations of RTS states
    
def transitions_durations_RTS(data, split):
    """ This funtion determines the durations of the RTS transitions. 
    Args:
        data (numpy array): data from the RTS measurement
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
    if data[0] < split and data[-1] < split:
        duration_up = np.array([transitions_dn[i]-transitions_up[i] for i in range(0,len(transitions_up))])
        duration_dn = np.array([transitions_up[i+1]-transitions_dn[i] for i in range(0,len(transitions_up)-1)])
        
    elif data[0] < split and data[-1] > split:
        duration_up = np.array([transitions_dn[i]-transitions_up[i] for i in range(0,len(transitions_dn))])
        duration_dn = np.array([transitions_up[i+1]-transitions_dn[i] for i in range(0,len(transitions_dn))])
    
    elif data[0] > split and data[-1] < split:
        duration_up = np.array([transitions_dn[i+1]-transitions_up[i] for i in range(0,len(transitions_up))])
        duration_dn = np.array([transitions_up[i]-transitions_dn[i] for i in range(0,len(transitions_up))])
       
    elif data[0] > split and data[-1] > split:
        duration_up = np.array([transitions_dn[i+1]-transitions_up[i] for i in range(0,len(transitions_up)-1)])
        duration_dn = np.array([transitions_up[i]-transitions_dn[i] for i in range(0,len(transitions_up))])
        

    return duration_dn, duration_up


#%% function to analyse the RTS measurements


def tunnelrates_RTS(data, plungers, samplerate = None, min_duration = 5, fig = None, verbose = 0):
    """
    This function takes an RTS dataset, fits a double gaussian, finds the split between the two levels, 
    determines the durations in these two levels, fits a decaying exponantial on two arrays of durations, 
    which gives the tunneling frequency for both the levels.
    
    Args:
        data (array): qcodes DataSet (or 1d data array) with the RTS data
        plungers ([str, str]): array of the two plungers used to perform the RTS measurement
        samplerate (int or float): sampling rate of either the fpga or the digitizer, optional if given in the metadata
                of the measured data
        min_duration (int): minimal number of datapoints a duration should last to be taking into account for the analysis
        fig (Nonetype or int): shows figures and sends them to the ppt when is not None
        verbose (int): prints info to the console when > 0
        
    Returns:
        tunnelrate_dn (numpy.float64): tunneling rate of the down level (kHz)
        tunnelrate_up (numpy.float64): tunneling rate of the up level (kHz)
        dictionary with relevent (fit) parameters
        
    """
    
            
    if type(data) == qcodes.data.data_set.DataSet:
        plungers = plungers
        metadata = data.metadata
        gates = metadata['allgatevalues']
        plungervalue = gates[plungers[0]]
    
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
        addPPTslide(title=title, fig=plt.figure(title))
        
    # binning the data and determining the bincentres
    num_bins = int(np.sqrt(len(data)))
    counts, bins = np.histogram(data, bins=num_bins)
    bincentres = np.array([(bins[i] + bins[i+1])/2 for i in range(0, len(bins)-1)])
    
    # fitting the double gaussian
    par_fit, par_guess = fit_double_gaussian(bincentres, counts)
    
    # finding the split between the up and the down state, seperation between the max of the two gaussians measured in the sum of the std
    seperation = (par_fit[5]-par_fit[4])/(abs(par_fit[2])+abs(par_fit[3])) 
    split = par_fit[4]+seperation*abs(par_fit[2])
    
    
    if verbose:
        print('Fit paramaters double gaussian:\n mean down: %.3f counts' % par_fit[4] +', mean up:%.3f counts' % par_fit[5] + ', std down: %.3f counts' % par_fit[2] +', std up:%.3f counts' % par_fit[3])
        print('Seperation between peaks gaussians: %.3f std' % seperation)
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
        addPPTslide(title=title, fig=plt.figure(title),notes='Fit paramaters double gaussian:\n mean down: %.3f counts' % par_fit[4] +', mean up:%.3f counts' % par_fit[5] + ', std down: %.3f counts' % par_fit[2] +', std up:%.3f counts' % par_fit[3] +'.Seperation between peaks gaussians: %.3f std' % seperation +'. Split between two levels: %.3f' % split )
        
    if seperation < 2:
        print('Seperation between the peaks of the gaussian is less then 2 std, indicating that the fit was not succesfull')
        raise Exception
        
    if seperation > 7:
        print('Seperation between the peaks of the gaussian is more then 7 std, indicating that the fit was not succesfull')
        raise Exception        
   
    
    # count the number of transitions and their duration
    durations_dn_idx, durations_up_idx = transitions_durations_RTS(data, split)
    
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
        print('Number of datapoints is not be enough to make an acurate fit of the exponantial decay for level down.')
        raise Exception('Number of datapoints is not be enough to make an acurate fit of the exponantial decay for level down.')
    
    bincentres_dn = np.array([(bins_dn[i] + bins_dn[i+1])/2 for i in range(0, len(bins_dn)-1)])
    
    #fitting exponantial decay for down level
    [A_dn_fit, gamma_dn_fit] = fit_exp_decay(bincentres_dn, counts_dn)
    tunnelrate_dn = gamma_dn_fit/1000
    
    if verbose:
        print('Tunnel rate down: %.1f kHz' % tunnelrate_dn)
    
    if fig:
        title = 'Fitted exponantial decay, level down'
        plt.figure(title)
        plt.plot(bincentres_dn, counts_dn, 'o', label='Counts down') 
        plt.plot(bincentres_dn, exp_decay(bincentres_dn, [A_dn_fit, gamma_dn_fit]),'r', label='Fitted exponantial decay \n t_dn: %.1f kHz' % tunnelrate_dn)
        plt.xlabel('Lifetime (s)')
        plt.ylabel('Counts per bin')
        plt.legend()
        plt.title(title)
        addPPTslide(title=title, fig=plt.figure(title))
        
    # calculating the number of bins and counts for up level
    numbins_up = int(np.sqrt(len(durations_up_srt)))
    counts_up, bins_up = np.histogram(durations_up_srt, bins=numbins_up)
    
    if counts_up[0] < 400:
        warnings.warn('Number of datapoints might not be enough to make an acurate fit of the exponantial decay for level up.')
        
    if counts_dn[0] < 50:
        print('Number of datapoints is not be enough to make an acurate fit of the exponantial decay for level down.')
        raise Exception('Number of datapoints is not be enough to make an acurate fit of the exponantial decay for level down.')
    
    bincentres_up = np.array([(bins_up[i] + bins_up[i+1])/2 for i in range(0, len(bins_up)-1)])
    
    #fitting exponantial decay for up level
    [A_up_fit, gamma_up_fit] = fit_exp_decay(bincentres_up, counts_up)
    tunnelrate_up = gamma_up_fit/1000
    
    if verbose:
        print('Tunnel rate up: %.1f kHz' % tunnelrate_up)
    
    if fig:
        title = 'Fitted exponantial decay, level up'
        plt.figure(title)
        plt.plot(bincentres_up, counts_up, 'o', label='Counts up') 
        plt.plot(bincentres_up, exp_decay(bincentres_up, [A_up_fit, gamma_up_fit]),'r', label='Fitted exponantial decay \n t_up: %.1f kHz' % tunnelrate_up)
        plt.xlabel('Lifetime (s)')
        plt.ylabel('Data points per bin')
        plt.legend()
        plt.title(title)
        addPPTslide(title=title, fig=plt.figure(title))
  
    return tunnelrate_dn, tunnelrate_up, {'plunger value':plungervalue, 'sampling rate':samplerate, 'fit parameters double gaussian':par_fit, \
                                        'guess parameters double gaussian': par_guess, 'seperations between pieks gaussians':seperation, \
                                        'split between the two levels':split, 'fit parameters exp. decay down':[A_dn_fit, gamma_dn_fit], \
                                        'fit parameters exp. decay up':[A_up_fit, gamma_up_fit]}

#%% function to apply tunnelrates_RTS on a series of RTS measurements
    
def tunnelrates_RTS_multiple(datadir, plungers, datatype = 1, fig=1):
    """ This function applies the tunnelrates_RTS function on a series of RTS measurements for which the plunger value
    was scanned over the addition line. It prints the iteration, tunnelrate down, tunnelrate up and the gate value or 
    the appropiate error message.
    
    Args:
        datadir (list): tags of storage object or list of file names of qucode datasets
        plungers ([str, str]): array of the two plungers used to perform the RTS measurement
        datatype (int): 0 or 1, indicates type of data that is read in, 0 for qucodes datafiles, 1 for storage object
        fig (None or int): shows figure and sends them to the ppt if not None
        
    Returns:
        tunnelrate_down (array): all the values for tunnelrate down
        tunnelrate_up (array): all the values for tunnelrate up
        plunger_value (array): all the plunger values
        
        """
        
    tunnelrate_down = []
    tunnelrate_up = []
    plunger_value = []
    
    for i in range(len(datadir)):
        try:
            if datatype == 0:
                data = qcodes.load_data(datadir[i])
            if datatype ==1:
                data=storage.load_result(datadir[i])
            test = tunnelrates_RTS(data,plungers = plungers)
            tunnelrate_down.append(test[0])
            tunnelrate_up.append(test[1])
            plunger_value.append(test[2]['plunger value'])
            print(i, test[0], test[1], test[2]['plunger value'])
            pass
        except:
            continue
        
    if fig:
        plt.figure()
        plt.plot(plunger_value, tunnelrate_down,'o', label = 'tunnelrate down')
        plt.plot(plunger_value,tunnelrate_up ,'o', label = 'tunnelrate up')
        plt.xlabel('Variation in plunger gate (mV)')
        plt.ylabel('Tunneling rate (kHz)')
        plt.legend()
        plt.show()
        
    return tunnelrate_down, tunnelrate_up, plunger_value

