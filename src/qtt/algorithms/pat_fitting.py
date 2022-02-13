""" Functionality to fit PAT models

For more details see: https://arxiv.org/abs/1803.10352

@author: diepencjv / eendebakpt
"""

import matplotlib.pyplot as plt
# %% Load packages
import numpy as np
import scipy.constants
import scipy.ndimage
import scipy.signal

from qtt.pgeometry import robustCost

# %%
ueV2Hz = scipy.constants.e / scipy.constants.h * 1e-6


def one_ele_pat_model(x_data, pp):
    r""" Model for one electron pat

    This is :math:`\phi=\sqrt{ { ( leverarm * (x-x_0) ) }^2 + 4 t^2 } \mathrm{ueV2Hz}`

    Args:
        x_data (array): detuning (mV)
        pp (array): xoffset (mV), leverarm (ueV/mV) and t (ueV)

    For more details see: https://arxiv.org/abs/1803.10352

    """
    if len(pp) == 1:
        pp = pp[0]
    xoffset = pp[0]
    leverarm = pp[1]
    t = pp[2]
    y = np.sqrt(np.power((x_data - xoffset) * leverarm, 2) + 4 * t**2) * ueV2Hz
    return y


def two_ele_pat_model(x_data, pp):
    r""" Model for two electron pat

    This is \phi = \pm \frac{leverarm}{2} (x - x0) +
        \frac{1}{2} \sqrt{( leverarm (x - x0) )^2 + 8 t^2 }

    Args:
        x_data (array): detuning (mV)
        pp (array): xoffset (mV), leverarm (ueV/mV) and t (ueV)

    """
    if len(pp) == 1:
        pp = pp[0]
    xoffset = pp[0]
    leverarm = pp[1]
    t = pp[2]
    yl = (- leverarm * (x_data - xoffset) / 2 + 1 / 2 * np.sqrt((leverarm * (x_data - xoffset))**2 + 8 * t**2)) * ueV2Hz
    ym = np.sqrt((leverarm * (x_data - xoffset))**2 + 8 * t**2) * ueV2Hz
    yr = (leverarm * (x_data - xoffset) / 2 + 1 / 2 * np.sqrt((leverarm * (x_data - xoffset))**2 + 8 * t**2)) * ueV2Hz

    return yl, ym, yr

# %%


class pat_score():

    def __init__(self, even_branches=[True, True, True], branch_reduction=None):
        """ Class to calculate scores for PAT fitting """
        self.even_branches = even_branches
        self.branch_reduction = branch_reduction

    def pat_one_ele_score(self, xd, yd, pp, weights=None, thr=2e9):
        """ Calculate score for pat one electron model

        Args:
            xd (array): x coordinates of peaks in sensor signal
            yd (array): y coordinates of peaks in sensor signal
            pp (array): model parameters
        """
        ydatax = one_ele_pat_model(xd, pp)
        charge_change = np.abs(np.abs(pp[1]) * (xd - pp[0]) / np.sqrt((pp[1] * (xd - pp[0]))**2 + 4 * pp[2]**2))
        sc = np.abs(ydatax - yd) * charge_change
        scalefac = thr
        sc = np.sqrt(robustCost(sc / scalefac, thr / scalefac, 'BZ0')) * scalefac
        if weights is not None:
            sc = sc * weights
        sc = np.linalg.norm(sc, ord=4) / sc.size
        if pp[1] < 10:
            sc *= 10
        if pp[2] > 150:
            sc *= 10
        return sc

    def pat_two_ele_score(self, xd, yd, pp, weights=None, thr=2e9):
        """ Calculate score for pat two electron model

        Args:
            xd (array): x coordinates of peaks in sensor signal
            yd (array): y coordinates of peaks in sensor signal
            pp (array): model parameters
        """
        ymodel = two_ele_pat_model(xd, pp)
        denom = np.sqrt((pp[1] * (xd - pp[0]))**2 + 8 * pp[2]**2)
        charge_changes = []
        charge_changes.append(1 / 2 * (1 + pp[1] * (xd - pp[0]) / denom))
        charge_changes.append(np.abs(pp[1] * (xd - pp[0]) / denom))
        charge_changes.append(1 / 2 * (1 - pp[1] * (xd - pp[0]) / denom))
        linesize = ymodel[0].shape[0]
        if self.branch_reduction is None or self.branch_reduction == 'minimum':
            sc = np.inf * np.ones(linesize)
            for idval, val in enumerate(self.even_branches):
                if val:
                    sc = np.minimum(sc, np.abs(ymodel[idval] - yd))
        elif self.branch_reduction == 'mean':
            sc = []
            for idval, val in enumerate(self.even_branches):
                if val:
                    sc.append(np.abs(ymodel[idval] - yd))
            sc = np.mean(np.array(sc), axis=1)
        else:
            raise NotImplementedError('branch_reduction %s not implemented' % self.branch_reduction)
        scalefac = thr
        sc = np.sqrt(robustCost(sc / scalefac, thr / scalefac, 'BZ0')) * scalefac
        if weights is not None:
            sc *= weights
        sc = np.linalg.norm(sc, ord=4) / sc.size
        if pp[1] < 10:
            sc *= 10000
        return sc

# %%


def pre_process_pat(x_data, y_data, background, z_data, fig=None):
    """ Pre-process a pair of background and sensor signal from a pat scan.

    Args:
        x_data (array): detuning (mV)
        y_data (array): frequency (Hz)
        background (array): e.g. sensor signal of POL scan
        z_data (array): sensor signal of PAT scan
        fig (None or int)
    Returns:
        imx (array)
        imq (array)
        backgr_sm (array)
    """
    backgr_sm = scipy.ndimage.gaussian_filter(background, sigma=5)

    imq = z_data - backgr_sm
    imq = imq - np.mean(imq, axis=1).reshape((-1, 1))

    ks = 5
    w = np.ones((1, ks)) / ks
    imx = scipy.ndimage.convolve(imq, w, mode='nearest')

    qq = np.percentile(imx, [5, 50, 95])
    imx = imx - qq[1]
    qq = np.percentile(imx, [2, 50, 98])
    scale = np.mean([-qq[0], qq[2]])
    imx = imx / scale

    if fig is not None:
        #        y_data = np.arange(imq.shape[0])
        plt.figure(fig)
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.pcolormesh(x_data, y_data, z_data, shading='auto')
        plt.xlabel('Detuning (mV)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Input data')
        plt.subplot(2, 2, 2)
        plt.pcolormesh(x_data, y_data, imq, shading='auto')
        plt.xlabel('Detuning (mV)')
        plt.ylabel('Frequency (Hz)')
        plt.title('imq')
        plt.subplot(2, 2, 3)
        plt.pcolormesh(x_data, y_data, imx, shading='auto')
        plt.xlabel('Detuning (mV)')
        plt.ylabel('Frequency (Hz)')
        plt.title('imx')
        plt.tight_layout()

    return imx, imq, backgr_sm

# %%


def detect_peaks(x_data, y_data, imx, sigmamv=.25, fig=400, period=1e-3, model='one_ele'):
    """ Detect peaks in sensor signal, e.g. from a pat scan.

    Args:
        x_data (array): detuning (mV)
        y_data (array): frequencies (Hz)
        imx (array): sensor signal of PAT scan, background is usually already subtracted

    Returns:
        detected_peaks (array): coordinates of detected peaks
        results (dict): additional fitting data
    """
    thr = .4
    thr2 = .6

    # chop off part of the data, because T1 is relatively long
    mvedge = .1 * (np.max(x_data) - np.min(x_data))
    if model == 'two_ele':
        mvthr = (np.max(x_data) - np.min(x_data)) * .25e-3 / period  # T1 \approx .1 ms [Ref]
        horz_vals = x_data[(x_data > (np.min(x_data) + np.maximum(mvthr, mvedge)))
                           & (x_data < (np.max(x_data) - mvedge))]
        z_data = imx[:, (x_data > (np.min(x_data) + np.maximum(mvthr, mvedge))) & (x_data < (np.max(x_data) - mvedge))]
    elif model == 'one_ele':
        horz_vals = x_data[(x_data > (np.min(x_data) + mvedge)) & (x_data < (np.max(x_data) - mvedge))]
        z_data = imx[:, (x_data > (np.min(x_data) + mvedge)) & (x_data < (np.max(x_data) - mvedge))]
    else:
        raise Exception('no such model')

    scalefac = (np.max(horz_vals) - np.min(horz_vals)) / (z_data.shape[1] - 1)  # mV/pixel

    # smooth input image
    kern = scipy.signal.gaussian(71, std=sigmamv / scalefac)
    kern = kern / kern.sum()
    imx2 = scipy.ndimage.convolve(z_data, kern.reshape((1, -1)), mode='nearest')

    # get maximum value for each row
    mm1 = np.argmax(imx2, axis=1)
    val = imx2[np.arange(0, imx2.shape[0]), mm1]

    idx1 = np.where(np.abs(val) > thr)[0]    # only select indices above scaled threshold

    xx1 = np.vstack((horz_vals[mm1[idx1]], y_data[idx1]))  # position of selected points

    # get minimum value for each row
    mm2 = np.argmin(imx2, axis=1)
    val = imx2[np.arange(0, imx2.shape[0]), mm2]
    # remove points below threshold
    idx2 = np.where(np.abs(val) > thr)[0]

    xx2 = np.vstack((horz_vals[mm2[idx2]], y_data[idx2]))

    # join the two sets
    detected_peaks = np.hstack((xx1, xx2))

    # determine weights for the points
    qq = np.intersect1d(idx1, idx2)
    q1 = np.searchsorted(idx1, qq)
    q2 = np.searchsorted(idx2, qq)
    w1 = .5 * np.ones(len(idx1))
    w1[q1] = 1
    w2 = .5 * np.ones(len(idx2))
    w2[q2] = 1

    wfac = .1
    w1[np.abs(val[idx1]) < thr2] = wfac
    w2[np.abs(val[idx2]) < thr2] = wfac
    weights = np.hstack((w1, w2))

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plt.pcolormesh(x_data, y_data, imx, shading='auto')
        plt.plot(horz_vals[mm1[idx1]], y_data[idx1], '.b', markersize=14, label='idx1')
        plt.plot(horz_vals[mm2[idx2]], y_data[idx2], '.r', markersize=14, label='idx2')
        plt.xlabel('Detuning (mV)')
        plt.ylabel('Frequency (Hz)')

    return detected_peaks, {'weights': weights, 'detected_peaks': detected_peaks}

# %%


def fit_pat_to_peaks(pp, xd, yd, trans='one_ele', even_branches=[True, True, True], weights=None, xoffset=None, verbose=1, branch_reduction=None):
    """ Core fitting function for PAT measurements, based on detected resonance
    peaks (see detect_peaks).

    Args:
        pp (array): initial guess of fit parameters
        xd (array): x coordinates of peaks in sensor signal (mV)
        yd (array): y coordinates of peaks in sensor signal (Hz)
        trans (string): 'one_ele' or 'two_ele'
        xoffset (float): the offset from zero detuning in voltage. If this has been determined before, then fixing this
            parameter reduces the fitting time.
    """
    ppx = pp.copy()
    pat_score_class = pat_score(even_branches=even_branches, branch_reduction=branch_reduction)
    if trans == 'one_ele':
        pat_model_score = pat_score_class.pat_one_ele_score
    elif trans == 'two_ele':
        pat_model_score = pat_score_class.pat_two_ele_score
    else:
        raise Exception('This model %s is not implemented.' % trans)

    if xoffset is None:
        def ff(x): return pat_model_score(xd, yd, [x, pp[1], ppx[2]], weights=weights)
        r = scipy.optimize.brute(ff, ranges=[(pp[0] - 2, pp[0] + 2)], Ns=20, disp=False)
        ppx[0] = r
        sc0 = pat_model_score(xd, yd, pp, weights=weights)
        sc = pat_model_score(xd, yd, ppx, weights=weights)
        if verbose >= 2:
            print('fit_pat_model: %s: %.4f -> %.4f' % (['%.2f' % x for x in ppx], sc0 / 1e6, sc / 1e6))

    if xoffset is None:
        def ff(x): return pat_model_score(xd, yd, x, weights=weights)
        r = scipy.optimize.minimize(ff, ppx, method='Powell', options=dict({'disp': verbose >= 1}))
        ppx = r['x']
    else:
        def ff(x): return pat_model_score(xd, yd, np.array([xoffset, x[0], x[1]]), weights=weights)
        r = scipy.optimize.minimize(ff, np.array([ppx[1], ppx[2]]),
                                    method='Powell', options=dict({'disp': verbose >= 1}))
        ppx = np.insert(r['x'], 0, xoffset)

    if xoffset is None:
        def ff(x): return pat_model_score(xd, yd, x, weights=weights, thr=.5e9)
        r = scipy.optimize.minimize(ff, ppx, method='Powell', options=dict({'disp': verbose >= 1}))
        ppx = r['x']
    else:
        def ff(x): return pat_model_score(xd, yd, np.array([xoffset, x[0], x[1]]), weights=weights)
        r = scipy.optimize.minimize(ff, np.array([ppx[1], ppx[2]]),
                                    method='Powell', options=dict({'disp': verbose >= 1}))
        ppx = np.insert(r['x'], 0, xoffset)

    sc0 = pat_model_score(xd, yd, pp, weights=weights)
    sc = pat_model_score(xd, yd, ppx, weights=weights)
    if verbose:
        print('fit_pat_model: %.4f -> %.4f' % (sc0 / 1e6, sc / 1e6))

    return ppx

# %%


def fit_pat(x_data, y_data, z_data, background, trans='one_ele', period=1e-3,
            even_branches=[True, True, True], par_guess=None, xoffset=None, verbose=1):
    """ Wrapper for fitting the energy transitions in a PAT scan.

    For more details see: https://arxiv.org/abs/1803.10352

    Args:
        x_data (array): detuning (mV)
        y_data (array): frequencies (Hz)
        z_data (array): sensor signal of PAT scan
        background (array): sensor signal of POL scan
        trans (str): can be 'one_ele' or 'two_ele'
        even_branches (list of booleans): indicated which branches of the model to use for fitting

    Returns:
        pp (array): fitted xoffset (mV), leverarm (ueV/mV) and t (ueV)
        results (dict): contains keys par_guess (array), imq (array) re-scaled and re-centered sensor signal, imextent (array), xd, yd, ydf
    """
    imx, imq, _ = pre_process_pat(x_data, y_data, background, z_data)

    xx, dpresults = detect_peaks(x_data, y_data, imx, model=trans, period=period, sigmamv=.05, fig=None)
    xd = xx[0, :]
    yd = xx[1, :]

    if par_guess is None:
        par_guess = np.array([np.nanmean(x_data), 65, 10])

    pp = fit_pat_to_peaks(par_guess, xd, yd, trans=trans, even_branches=even_branches,
                          xoffset=xoffset, verbose=0)
    if trans == 'one_ele':
        model = one_ele_pat_model
    elif trans == 'two_ele':
        model = two_ele_pat_model
    ydf = model(xd, pp)

    return pp, {'imq': imq, 'xd': xd, 'yd': yd, 'ydf': ydf, 'par_guess': par_guess}

# %%


def plot_pat_fit(x_data, y_data, z_data, pp, trans='one_ele', fig=400, title='Fitted model', label='model'):
    """ Plot the fitted model of the PAT transition(s)

    Args:
        x_data (array): detuning in millivolts
        y_data (array): frequencies
        z_data (array): sensor signal of PAT scan
        pp (array): xoffset (mV), leverarm (ueV/mV) and t (ueV)
        model (function): model describing the PAT transitions
    """
    if z_data is not None:
        plt.figure(fig)
        plt.clf()
        plt.pcolormesh(x_data, y_data, z_data, shading='auto')
        plt.title(title)
        plt.xlabel('Detuning (mV)')
        plt.ylabel('Frequency (Hz)')

    if trans == 'one_ele':
        model = one_ele_pat_model
        yfit = model(x_data, pp)
        plt.plot(x_data, yfit, '-g', label=label)
        yfit_t0 = model(x_data, np.array([pp[0], pp[1], 0]))
        plt.plot(x_data, yfit_t0, '--g')
    elif trans == 'two_ele':
        model = two_ele_pat_model
        ylfit, ymfit, yrfit = model(x_data, pp)
        plt.plot(x_data, ylfit, '-g', label='S-T')
        plt.plot(x_data, ymfit, '-r', label='S-S')
        plt.plot(x_data, yrfit, '-b', label='T-S')

    plt.ylim([np.min(y_data), np.max(y_data)])

# %%


def show_traces(x_data, z_data, fig=100, direction='h', title=None):
    """ Show traces of an image

    Args:
        x_data (array): detuning in millivolts
        z_data (array): input image. rows are taken as the traces
        fig (int): number for figure window to use
        direction (str): can be 'h' or 'v'
    """
    plt.figure(fig)
    plt.clf()
    if direction == 'v' or direction == 'vertical':
        for ii, l in enumerate(z_data.T):
            c = []
            c = plt.cm.jet(float(ii) / z_data.shape[1])
            plt.plot(x_data, l, '', color=c)
        if title is None:
            title = 'Blue: left vertical lines, red: right lines'
        plt.title(title)
    else:
        for ii, l in enumerate(z_data):
            c = []
            c = plt.cm.jet(float(ii) / z_data.shape[0])
            plt.plot(x_data, l, '', color=c)
        if title is None:
            title = 'Blue: top lines, red: bottom lines'
        plt.title(title)
    plt.xlabel('Detuning (mV)')
    plt.ylabel('Signal (a.u.)')
