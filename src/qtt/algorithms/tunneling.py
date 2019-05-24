""" Functionality for analysing inter-dot tunnel frequencies

@author: diepencjv
"""

# %%
import scipy.optimize
import numpy as np
import scipy.ndimage


def polmod_all_2slopes(x_data, par, kT, model=None):
    """ Polarization line model.

    This model is based on [DiCarlo2004, Hensgens2017]. For an example see
    https://github.com/VandersypenQutech/qtt/blob/master/examples/example_polFitting.ipynb

    Args:
        x_data (1 x N array): chemical potential difference in ueV
        par (1 x 6 array): parameters for the model
            - par[0]: tunnel coupling in ueV
            - par[1]: offset in x_data for center of transition
            - par[2]: offset in background signal
            - par[3]: slope of sensor signal on left side
            - par[4]: slope of sensor signal on right side
            - par[5]: height of transition, i.e. sensitivity for electron transition
        kT (float): temperature in ueV
        model (): Not used

    Returns:
        y_data (array): sensor data, e.g. from a sensing dot or QPC
    """
    x_data_center = x_data - par[1]
    Om = np.sqrt(x_data_center**2 + 4 * par[0]**2)
    Q = 1 / 2 * (1 + x_data_center / Om * np.tanh(Om / (2 * kT)))
    slopes = par[3] + (par[4] - par[3]) * Q
    y_data = par[2] + x_data_center * slopes + Q * par[5]

    return y_data


def polweight_all_2slopes(x_data, y_data, par, kT, model='one_ele'):
    """ Cost function for polarization fitting.
    Args:
        x_data (1 x N array): chemical potential difference in ueV
        y_data (1 x N array): sensor data, e.g. from a sensing dot or QPC
        par (1 x 6 array): see polmod_all_2slopes
        kT (float): temperature in ueV
    Returns:
        total (float): sum of residues
    """
    mod = polmod_all_2slopes(x_data, par, kT, model=model)
    total = np.linalg.norm(y_data - mod)

    return total


def polweight_all_2slopes_2(x_data, y_data, par, kT, model='one_ele'):
    """ Cost function for polarization fitting.
    Args:
        x_data (1 x N array): chemical potential difference in ueV
        y_data (1 x N array): sensor data, e.g. from a sensing dot or QPC
        par (1 x 6 array): see polmod_all_2slopes
        kT (float): temperature in ueV
    Returns:
        total (float): sum of residues
    """
    mod = pol_mod_two_ele_boltz(x_data, par, kT)
    total = np.linalg.norm(y_data - mod)

    return total


def _polarization_fit_initial_guess(x_data, y_data, kT=0, padding_fraction=0.15, fig=None, verbose=0):
    t_guess = (x_data[-1] - x_data[0]) / 30  # hard-coded guess in ueV
    number_points_padding = round(padding_fraction * len(x_data))
    linear_fit = np.polyfit(x_data[-number_points_padding:], y_data[-number_points_padding:], 1)
    slope_guess = linear_fit[0]
    data_noslope = y_data - slope_guess * (x_data - x_data[0])
    data_noslope_1der = scipy.ndimage.filters.gaussian_filter(data_noslope, sigma=20, order=1)

    data_noslope_1der[:number_points_padding] = 0
    data_noslope_1der[number_points_padding:0] = 0
    transition_idx = np.abs(data_noslope_1der).argmax()
    sensitivity_guess = np.sign(x_data[-1] - x_data[0]) * np.sign(data_noslope_1der[transition_idx]) *\
                        (np.percentile(data_noslope, 90) - np.percentile(data_noslope, 10))
    x_offset_guess = x_data[transition_idx]
    y_offset_guess = y_data[transition_idx] - sensitivity_guess / 2
    par_guess = np.array([t_guess, x_offset_guess, y_offset_guess, slope_guess, slope_guess, sensitivity_guess])
    if verbose >= 2:
        print('_polarization_fit_initial_guess: trans_idx %s' % (transition_idx, ))

    if fig:
        plt.figure(fig)
        plt.clf()
        plt.plot(x_data, y_data, '.', label='data')
        plt.plot(x_data, np.polyval(linear_fit, x_data), 'm', label='linear fit tail')
        plt.plot(x_data, polmod_all_2slopes(x_data, par_guess, 0), 'r', label='initial guess')
        vline = plt.axvline(x_offset_guess, label='centre')
        vline.set_alpha(.5)
        vline.set_color('c')
        plt.legend()

        plt.figure(fig + 1)
        plt.clf()
        plt.plot(x_data, data_noslope_1der, 'm', label='derivative')
        plt.legend()
    return par_guess


def fit_pol_all(x_data, y_data, kT, model='one_ele', maxiter=None, maxfun=5000, verbose=1, par_guess=None,
                method='fmin'):
    """ Polarization line fitting. 

    The default value for the maxiter argument of scipy.optimize.fmin is N*200
    the number of variables, i.e. 1200 in our case.
    Args:
        x_data (1 x N array): chemical potential difference in ueV
        y_data (1 x N array): sensor data, e.g. from a sensing dot or QPC
        kT (float): temperature in ueV
    Returns:
        par_fit (1 x 6 array): fitted parameters, see :func:`polmod_all_2slopes`
        par_guess (1 x 6 array): initial guess of parameters for fitting, see :func:`polmod_all_2slopes`
        results (dictionary): dictionary with fitting results
    """
    if par_guess is None:
        par_guess = _polarization_fit_initial_guess(x_data, y_data, kT, fig=None)

    fitdata = {}
    if method is 'fmin':
        def func_fmin(par): return polweight_all_2slopes(x_data, y_data, par, kT, model=model)
        par_fit = scipy.optimize.fmin(func_fmin, par_guess, maxiter=maxiter, maxfun=maxfun, disp=verbose >= 2)
    elif method is 'curve_fit':
        def func_curve_fit(x_data, tc, x0, y0, ml, mr, h): return polmod_all_2slopes(
            x_data, (tc, x0, y0, ml, mr, h), kT, model=model)
        par_fit, par_cov = scipy.optimize.curve_fit(func_curve_fit, x_data, y_data, par_guess)
        fitdata['par_cov'] = par_cov
    else:
        raise Exception('Unrecognized fitting method')

    results = {'fitted_parameters': par_fit, 'initial_parameters': par_guess, 'type': 'polarization fit', 'kT': kT}
    return par_fit, par_guess, results


def plot_polarization_fit(detuning, signal, results, fig, verbose=1):
    """ Plot the results of a polarization line fit

    Args:
        detuning (array): detuning in ueV
        signal (array): measured signal
        results (dict): results of fit_pol_all
        fig (int or None): figure handle
        verbose (int): Verbosity level

    """
    h = scipy.constants.physical_constants['Planck constant in eV s'][0] * \
        1e15  # ueV/GHz; Planck's constant in eV/Hz*1e15 -> ueV/GHz

    par_fit = results['fitted_parameters']
    initial_parameters = results['initial_parameters']
    kT = results['kT']

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plt.plot(detuning, signal, 'bo')
        plt.plot(detuning, polmod_all_2slopes(detuning, par_fit, kT), 'r', label='fitted model')
        if verbose >= 2:
            plt.plot(detuning, polmod_all_2slopes(detuning, initial_parameters, kT), ':c', label='initial guess')
        plt.title('Tunnel coupling: %.2f (ueV) = %.2f (GHz)' % (par_fit[0], par_fit[0] / h))
        plt.xlabel('Difference in chemical potentials (ueV)')
        _ = plt.ylabel('Signal (a.u.)')
        plt.legend()


def fit_pol_all_2(x_data, y_data, kT, model='one_ele', maxiter=None, maxfun=5000, verbose=1, par_guess=None,
                  method='fmin', returnextra=False):
    raise Exception('please use fit_pol_all instead')


def pol_mod_two_ele_boltz(x_data, par, kT):
    """ Model of the inter-dot transition with two electron spin states, also
    taking into account thermal occupation of the triplets. """
    t = par[0]
    x_offset = par[1]
    y_offset = par[2]
    dy_left = par[3]
    dy_right = par[4]
    dy = par[5]
    omega = np.sqrt((x_data - x_offset)**2 + 8 * t**2)
    E_Smin = - omega / 2
    E_T = (x_data - x_offset) / 2
    E_Splus = omega / 2
    part_func = np.exp(- E_Smin / kT) + 3 * np.exp(- E_T / kT) + np.exp(- E_Splus / kT)

    excess_charge = (np.exp(- E_Smin / kT) * 1 / 2 * (1 + (x_data - x_offset) / omega)
                     + np.exp(- E_Splus / kT) * 1 / 2 * (1 - (x_data - x_offset) / omega)) / part_func

    signal = y_offset + dy * excess_charge + (dy_left + (dy_right - dy_left) * excess_charge) * (x_data - x_offset)

    return signal


def data_to_exc_ch(x_data, y_data, pol_fit):
    """ Convert y_data to units of excess charge.

    Note: also re-centers to zero detuning in x-direction.

    Args:
        x_data (1 x N array): chemical potential difference in ueV
        y_data (1 x N array): sensor data, e.g. from a sensing dot or QPC
        pol_fit (1 x 6 array): fit parameters, see :func:`polmod_all_2slopes`
    """
    x_center = x_data - pol_fit[1]
    y_data_exc_ch = (y_data - pol_fit[2] - x_center * pol_fit[3]) / \
        (pol_fit[5] + (pol_fit[4] - pol_fit[3]) * x_center)

    return x_center, y_data_exc_ch


if __name__ == '__main__':
    """  Testing of fitting code       
    """
    import matplotlib.pyplot as plt
    import time
    from qtt import pgeometry
    import pandas as pd
    from pandas import Series

    # generate data
    par = np.array([.2, 0, .1, -.0031, -.001, .21])
    # xx0 = np.arange(-10, 10, .1)
    xx0 = np.linspace(-10, 10, 200)
    # xx0=4*np.linspace(-10, 10, 200)
    # xx = xx0
    xx = 2 * xx0
    # xx=xx0
    yy0 = polmod_all_2slopes(xx0, par, kT=0.001)
    yy = yy0 + .015 * (np.random.rand(yy0.size) - .5)
    x_data = xx
    data = yy

    t0 = time.time()
    for ii in range(5):
        parfit, _, _ = fit_pol_all(xx, yy, kT=0.001)
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
    plt.title('fitted t %.3f, gt %.3f' % (parfit[0], par[0]))
    plt.legend(numpoints=1)

    # %% Quick estimate
    noise = np.arange(0, .1, .5e-3)
    noise = np.hstack((noise, np.arange(1e-4, 5e-4, 1e-4)))
    noise.sort()

    pp = np.zeros((len(noise), 6))
    for ii, n in enumerate(noise):
        pgeometry.tprint('quick fit %d/%d' % (ii, len(noise)))
        yyx = yy + n * (np.random.rand(yy.size) - .5)
        parfit, _, _ = fit_pol_all(xx, yyx, kT=0.001, par_guess=None)
        pp[ii] = parfit

    plt.figure(200)
    plt.clf()
    plt.plot(noise, pp[:, 0], '.b', label='tunnel coupling')
    plt.xlabel('Noise')
    plt.ylabel('Estimated tunnel frequency')

    pgeometry.plot2Dline([0, -1, par[0]], '--c', label='true value')

    # %% Show effect of proper initialization
    yyx = yy + n * (np.random.rand(yy.size) - .5)
    parfit1, _, _ = fit_pol_all(xx, yyx, kT=0.001, par_guess=par)
    parfit2, _, _ = fit_pol_all(xx, yyx, kT=0.001, par_guess=None, verbose=2)
    parfit2i, _, _ = fit_pol_all(xx, yyx, kT=0.001, par_guess=parfit2)

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

    # %% Full estimate
    niter = 60
    ppall = np.zeros((len(noise), niter, 6))
    for ii, n in enumerate(noise):
        print('full fit %d/%d' % (ii, len(noise)))
        for j in range(niter):
            yyx = yy + n * (np.random.rand(yy.size) - .5)
            parfit, _, _ = fit_pol_all(xx, yyx, kT=0.001)
            ppall[ii, j] = parfit

    # %% Show uncertainties
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
        plt.fill_between(tb.index, tb - nstd * mstd, tb + nstd * mstd, color='b',
                         alpha=0.1, label='uncertainty (%d std)' % nstd)
    # plt.plot(tb.index, ma, 'k', label='mean')

    plt.plot(tb.index, tb, 'k', label='mean')
    pgeometry.plot2Dline([0, -1, par[0]], '--c', label='true value')
    plt.xlabel('Noise')
    plt.ylabel('Tunnel barrier frequency')
    plt.title('Estimated tunnel barrier', fontsize=14)
    plt.legend()

    # %% Number of data points
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
            parfit, _, _ = fit_pol_all(xx, yyx, kT=0.001)
            ppall[ii, j] = parfit
    # %%
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
        plt.fill_between(tb.index, tb - nstd * mstd, tb + nstd * mstd, color='b',
                         alpha=0.1, label='uncertainty (%d std)' % nstd)
    # plt.plot(tb.index, ma, 'k', label='mean')

    plt.plot(tb.index, tb, 'k', label='mean')
    pgeometry.plot2Dline([0, -1, par[0]], '--c', label='true value')
    plt.xlabel('Number of data points')
    plt.ylabel('Tunnel barrier frequency')
    plt.title('Estimated tunnel barrier', fontsize=14)
    plt.legend()
