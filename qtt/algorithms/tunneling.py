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


def fit_pol_all(delta, data, kT=0.001, maxiter=None, maxfun=5000, verbose=1, par_guess=None):
    ''' Calculate initial values for fitting and fit. The default value for the 
    maxiter argument of scipy.optimize.fmin is N*200 the number of variables, 
    i.e. 1200 in our case.

    Arguments:
        delta (array):
        data (array):
    Returns:
        par_fit (array):
    '''
    if par_guess is None:
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

    func = lambda par: polweight_all_2slopes(delta, data, par, kT=kT)

    par_fit = scipy.optimize.fmin(func, par_guess, maxiter=maxiter, maxfun=maxfun, disp=verbose >= 2)

    return par_fit, par_guess

#%% Example fitting

if __name__ == '__main__':
    """  Testing of fitting code       
    """
    import matplotlib.pyplot as plt
    import time

    # generate data
    par = np.array([.2, 0, .1, -.0031, -.001, .21])
    #xx0 = np.arange(-10, 10, .1)
    xx0=np.linspace(-10, 10, 200)
    #xx0=4*np.linspace(-10, 10, 200)
    xx=xx0
    xx=2*xx0
    #xx=xx0
    yy0 = polmod_all_2slopes(xx0, par, kT=0.001)
    yy=yy0 + .015 * (np.random.rand(yy0.size) - .5)
    delta = xx
    data = yy

    t0 = time.time()
    for ii in range(5):
        parfit, _ = fit_pol_all(xx, yy)
    dt = time.time() - t0
    yyfit = polmod_all_2slopes(xx, parfit, kT=0.001)
    print('dt: %.3f [s]' % dt)

    # show data
    plt.figure(100)
    plt.clf()
    plt.plot(xx, yy0, '.b', label='data')
    plt.plot(xx, yy, '.m', label='noise 0.1')
    plt.xlabel('x')
    plt.ylabel('value')
    plt.plot(xx, yyfit, '-r', label='fitted')
    plt.title('fitted t %.3f, gt %.3f'  % (parfit[0], par[0]))
    plt.legend(numpoints=1)

    #%% Quick estimate
    import qtt
    from qtt import pmatlab as pgeometry
    noise = np.arange(0, .1, .5e-3)
    noise = np.hstack((noise, np.arange(1e-4, 5e-4, 1e-4)))
    noise.sort()

    pp = np.zeros((len(noise), 6))
    for ii, n in enumerate(noise):
        pgeometry.tprint('quick fit %d/%d' % (ii, len(noise)))
        yyx = yy + n * (np.random.rand(yy.size) - .5)
        parfit, _ = fit_pol_all(xx, yyx, par_guess=None)
        pp[ii] = parfit

    plt.figure(200)
    plt.clf()
    plt.plot(noise, pp[:, 0], '.b', label='tunnel coupling')
    plt.xlabel('Noise')
    plt.ylabel('Estimated tunnel frequency')

    pgeometry.plot2Dline([0, -1, par[0]], '--c', label='true value')

    #%% Show effect of proper initialization
    yyx = yy + n * (np.random.rand(yy.size) - .5)
    parfit1, _ = fit_pol_all(xx, yyx, par_guess=par)
    parfit2, _ = fit_pol_all(xx, yyx, par_guess=None, verbose=2)
    parfit2i, _ = fit_pol_all(xx, yyx, par_guess=parfit2)

    yy1 = polmod_all_2slopes(xx, parfit1, kT=0.001)
    yy2 = polmod_all_2slopes(xx, parfit2, kT=0.001)
    yy2i = polmod_all_2slopes(xx, parfit2i, kT=0.001)

    c0 = polweight_all_2slopes(xx, yy, par, kT=0.001)
    c1 = polweight_all_2slopes(xx, yy1, par, kT=0.001)
    c2 = polweight_all_2slopes(xx, yy2, par, kT=0.001)
    c2i = polweight_all_2slopes(xx, yy2i, par, kT=0.001)
    print('cost %f init %f guess %f -> %f' % (c0, c1, c2, c2i))

    plt.figure(300)
    plt.clf()
    plt.plot(xx, yy, '-', label='model')
    plt.plot(xx, yyx, '.b', label='data')
    plt.plot(xx, yy1, '--r', label='fitted with gt')
    plt.plot(xx, yy2, '--g', label='fitted with guess')
    plt.legend()
    _ = plt.ylabel('Frequency')

    #%% Full estimate
    niter = 60
    ppall = np.zeros((len(noise), niter, 6))
    for ii, n in enumerate(noise):
        print('full fit %d/%d' % (ii, len(noise)))
        for j in range(niter):
            yyx = yy + n * (np.random.rand(yy.size) - .5)
            parfit, _ = fit_pol_all(xx, yyx)
            ppall[ii, j] = parfit

    #%% Show uncertainties
    import pandas as pd
    from pandas import Series

    plot_data = False
    plot_bands = True

    mean = np.mean(ppall[:, :, 0], axis=1)
    mstd = np.std(ppall[:, :, 0], axis=1)

    tb = Series(mean, index=noise)
    ma = pd.rolling_mean(tb, 20)

    plt.figure(1)
    plt.clf()
    if plot_data:
        qq = ppall[:, :, 0]
        ii = np.tile(noise, (niter, 1)).T.flatten()
        plt.plot(ii, qq.flatten(), '.r', alpha=.1)
    if plot_bands:
        nstd = 2
        plt.fill_between(tb.index, tb - nstd * mstd, tb + nstd * mstd, color='b', alpha=0.1, label='uncertainty (%d std)' % nstd)
    #plt.plot(tb.index, ma, 'k', label='mean')

    plt.plot(tb.index, tb, 'k', label='mean')
    pgeometry.plot2Dline([0, -1, par[0]], '--c', label='true value')
    plt.xlabel('Noise')
    plt.ylabel('Tunnel barrier frequency')
    plt.title('Estimated tunnel barrier', fontsize=14)
    plt.legend()

    #%% Number of data points
    Noise = .02
    npoints = np.array([20, 30, 50, 70, 100, 120, 160, 180, 200, 300, 500, 1000])
    niter = 60
    ppall = np.zeros((len(npoints), niter, 6))
    for ii, n in enumerate(npoints):
        print('full fit %d/%d' % (ii, len(npoints)))
        xx = np.linspace(-10, 10, n)
        yy = polmod_all_2slopes(xx, par, kT=0.001)

        for j in range(niter):
            yyx = yy + Noise * (np.random.rand(yy.size) - .5)
            parfit, _ = fit_pol_all(xx, yyx)
            ppall[ii, j] = parfit
#%%
    mean = np.mean(ppall[:, :, 0], axis=1)
    mstd = np.std(ppall[:, :, 0], axis=1)

    plot_data = True

    tb = Series(mean, index=npoints)
    ma = pd.rolling_mean(tb, 20)

    plt.figure(2)
    plt.clf()
    if plot_data:
        qq = ppall[:, :, 0]
        ii = np.tile(tb.index, (niter, 1)).T.flatten()
        plt.plot(ii, qq.flatten(), '.r', alpha=.1)
    if plot_bands:
        nstd = 2
        plt.fill_between(tb.index, tb - nstd * mstd, tb + nstd * mstd, color='b', alpha=0.1, label='uncertainty (%d std)' % nstd)
    #plt.plot(tb.index, ma, 'k', label='mean')

    plt.plot(tb.index, tb, 'k', label='mean')
    pgeometry.plot2Dline([0, -1, par[0]], '--c', label='true value')
    plt.xlabel('Number of data points')
    plt.ylabel('Tunnel barrier frequency')
    plt.title('Estimated tunnel barrier', fontsize=14)
    plt.legend()
