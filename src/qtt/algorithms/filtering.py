# -*- coding: utf-8 -*-

import scipy.signal


def butter_lowpass(cutoff_frequency, fs, order=5):
    """  Create Butter low-pass filter
    
    Args:
        cutoff: Cut-off frequency
        fs: Sample rate of the signal
        order: Order of the filter
    """ 
    nyq = 0.5 * fs
    normal_cutoff = float(cutoff_frequency) / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff_frequency, fs, order=5):
    """  Aply Butter low-pass filter
    
    Args:
        data: 1D input data to be filtered
        cutoff: Cut-off frequency
        fs: Sample rate of the signal
    Returns:
        Filtered signal
    
    https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
    
    """
    b, a = butter_lowpass(cutoff_frequency, fs, order=order)
    y =  scipy.signal.lfilter(b, a, data)
    return y


