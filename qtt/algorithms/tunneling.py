# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:11:56 2016

@author: diepencjv
"""

#%%
import scipy.optimize
import numpy as np


def polmod_all_2slopes(x, par, kT=0.001):
    ''' Polarization line model

    Arguments:
        x (1 x N array): input values (e.g. gate voltages)
        par (1 x 6 array): parameters for the model
            - par[0]: tunnel coupling in ...
            - par[1]: average x value
            - par[2]: background signal
            - par[3]: slope on left side
            - par[4]: slope on right side
            - par[5]: sensitivity
        kt (float): temperature in mV

    Returns:
        E (float): read-out value of the charge sensor (e.g. using the sensing dot)
    '''
    x = x - par[1]

    Om = np.sqrt(x**2 + 4 * par[0]**2)
    E = 1 / 2 * (1 + x / Om * np.tanh(Om / (2 * kT)))
    slopes = par[3] + (par[4] - par[3]) * E
    E = par[2] + x * slopes + E * par[5]

    return E


def polweight_all_2slopes(delta, data, par, kT=0.001):
    ''' Cost function for polarization fitting

    Arguments:
        delta (array):
        data (array):
        par (array):
    Returns:
        total (float):
    '''
    mod = polmod_all_2slopes(delta, par, kT=kT)
    diffs = data - mod
    norms = np.sqrt(np.sum(diffs**2))
    total = np.sum(norms)

    return total


def fit_pol_all(delta, data):
    ''' Calculate initial values for fitting and fit

    Arguments:
        delta (array):
        data (array):
    Returns:
        par_fit (array):
    '''
    slope_guess = np.polyfit(delta[0:99], data[0:99], 1)[0]  # hard-coded 30 points
#    slope_guess = np.mean(np.diff(data[0:99]))/np.mean(np.diff(delta[0:99])) # hard-coded 30 points
    dat_noslope = data - slope_guess * delta
    b = np.round(len(data) / 2)
#    sensor_offset_guess = data[int(b)-1]
    sensor_offset_guess = np.mean(data)
#    delta_offset_guess = np.mean(delta) # average of the delta values
    delta_offset_guess = delta[int(b) - 1]
    sensitivity_guess = np.max(dat_noslope) - np.min(dat_noslope)
    sensor_offset_guess = sensor_offset_guess - .5 * sensitivity_guess

    t_guess = 1 * 4.2 / 80
#    t_guess = 1 # hard-coded value
    par_guess = np.array([t_guess, delta_offset_guess, sensor_offset_guess, slope_guess, slope_guess, sensitivity_guess])  # second slope guess could be done more accurately

    func = lambda par: polweight_all_2slopes(delta, data, par)

    par_fit = scipy.optimize.fmin(func, par_guess)

    return par_fit, par_guess
