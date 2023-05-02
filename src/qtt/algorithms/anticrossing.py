""" Functions to analyse anti-crossings in charge stability diagrams.

Created on Wed Jul 12 08:08:57 2017

@author: diepencjv / eendebakpt
"""

# %%
import copy
import numpy as np
import qtt
import qtt.algorithms.images
from qtt.utilities.imagetools import cleanSensingImage
from qtt.utilities.imagetools import evaluateCross, fitModel
import matplotlib.pyplot as plt

from qcodes_loop.plots.qcmatplotlib import MatPlot

# %%


def fit_anticrossing(dataset, width_guess=None, angles_guess=None, psi=None, w=2.5,
                     diff_dir='dx', plot=False, verbose=1, param={}):
    """ Fits an anti-crossing model to a 2D scan.

    The model fitted is without tunnel coupling.

    Args:
        dataset (qcodes dataset): the 2D scan measurement dataset
        width_guess (None or float): Initial estimate for width of anti-crossing in the same unit as the dataset axis.
        angles_guess (None or float): Initial estimate for angles of anti-crossing in radians
        psi (None or float): angle for cross section of anti-crossing
        w (float): Width of lines in anti-crossing model
        diff_dir (str): differentiation direction
        plot (int): If non-zero, then plot into specified matplotlib window
        verbose (int): verbosity level
        param (dict): additional parameters

    Returns:
        anticross_fit: (dict) the parameters describing the fitted anti-cross
                       optionally the cost for fitting and the figure number

    The main fields in the result structure are:

        * centre (2x1 array): position of the fitted anti-crossing
        * fit_params (array): fitting parameters of the model. See :func:`fitModel`
        * fitpoints (dict): a dictionary with fitted points. centre is the centre point; left_point and right_point are the corners
                    of the anti-crossing. the four lines outwards are define by inner_points and outer_points

    """
    abs_val = True

    set_arrays = dataset.default_parameter_array().set_arrays
    labels = [x.name for x in set_arrays]

    im, tr = qtt.data.dataset2image(dataset)
    imextent = tr.scan_image_extent()

    # get parameters of the algorithm
    istep = param.get('istep', 0.25)  # resolution to convert the data into
    istepmodel = param.get('istepmodel', 0.5)  # resolution in which to perform model fitting
    ksizemv = param.get('ksizemv', 31)  # kernel size in mV

    diffvals = {'dx': 0, 'dy': 1, 'dxy': 2, 'xmy': 'xmy', 'g': 'g'}
    imc = cleanSensingImage(im, sigma=0.93, dy=diffvals[diff_dir])
    imx, (fw, fh, mvx, mvy, Hstraight) = qtt.algorithms.images.straightenImage(
        imc, imextent, mvx=istep, verbose=verbose)

    imx = imx.astype(np.float64) * \
        (100. / np.percentile(imx, 99))  # scale image

    # initialization of anti-crossing model
    param0 = [(imx.shape[0] / 2 + .5) * istep, (imx.shape[0] / 2 + .5) * istep,
              3.5, 1.17809725, 3.5, 4.3196899, 0.39269908]
    if width_guess is not None:
        param0[2] = width_guess

    if angles_guess is not None:
        param0[3:] = angles_guess

    if psi is not None:
        param0e = np.hstack((param0, [psi]))
    else:
        psi = np.pi / 4
        param0e = copy.copy(param0)

    # fit anti-crossing (twice)
    res = fitModel(param0e, imx, verbose=verbose, cfig=10, istep=istep,
                   istepmodel=istepmodel, ksizemv=ksizemv, use_abs=abs_val, model_line_width=w)
    fitparam = res.x
    res = fitModel(fitparam, imx, verbose=verbose, cfig=10, istep=istep,
                   istepmodel=istepmodel, ksizemv=ksizemv, use_abs=abs_val, model_line_width=w)
    fitparam = res.x

    cost, patch, cdata, _ = evaluateCross(
        fitparam, imx, verbose=0, fig=None, istep=istep, istepmodel=istepmodel)
    ccpixel_straight = cdata[0]
    ccpixel = qtt.pgeometry.projectiveTransformation(np.linalg.inv(Hstraight), (istepmodel / istep) * ccpixel_straight)

    if verbose:
        print('fit_anticrossing: patch size %s' % (patch.shape, ))

    def convert_coordinates(xpix):
        ccpixel = qtt.pgeometry.projectiveTransformation(np.linalg.inv(Hstraight), (istepmodel / istep) * xpix)
        return tr.pixel2scan(ccpixel)

    (cc, lp, hp, ip, op, _, _, _) = cdata

    centre = convert_coordinates(np.array(cc))
    lp = convert_coordinates(np.array(lp))
    hp = convert_coordinates(np.array(hp))
    ip = convert_coordinates(np.array(ip).T)
    op = convert_coordinates(np.array(op).T)

    anticross_fit = {'labels': labels}
    anticross_fit['centre'] = tr.pixel2scan(ccpixel)
    anticross_fit['fitpoints'] = {'centre': centre, 'left_point': lp,
                                  'right_point': hp, 'inner_points': ip, 'outer_points': op}

    if len(param) == 7:
        param = np.append(fitparam, psi)
    anticross_fit['fit_params'] = fitparam
    anticross_fit['params'] = param

    if plot:
        if isinstance(plot, int):
            fig = plot
        else:
            fig = None
        fig_anticross = plt.figure(fig)
        cost, patch, cdata, _ = evaluateCross(
            fitparam, imx, verbose=verbose, fig=fig_anticross.number, istep=istep, istepmodel=istepmodel, w=w)
        anticross_fit['cost'] = cost
        anticross_fit['patch'] = patch
        anticross_fit['cdata'] = cdata
        anticross_fit['fig_num'] = fig_anticross.number
        anticross_fit['imx'] = imx

    return anticross_fit


def plot_anticrossing(ds, afit, fig=100, linewidth=2):
    """ Plot fitted anti-crossing on dataset.

    Args:
        afit (dict): fit data from fit_anticrossing
        ds (None or DataSet): dataset to show
        fig (int): index of matplotlib window
        linewidth (integer): plot linewidth, default = 2

    Returns:
        -

    """
    fitpoints = afit['fitpoints']
    plt.figure(fig)
    plt.clf()

    if ds is not None:
        MatPlot(ds.default_parameter_array('diff_dir_g'), num=fig)
    cc = fitpoints['centre']
    plt.plot(cc[0], cc[1], '.m', markersize=12, label='fit centre')

    lp = fitpoints['left_point']
    hp = fitpoints['right_point']
    op = fitpoints['outer_points'].T
    ip = fitpoints['inner_points'].T
    plt.plot([float(lp[0]), float(hp[0])], [float(lp[1]), float(hp[1])], '.--m',
             linewidth=linewidth, markersize=10, label='transition line')

    for ii in range(4):
        if ii == 0:
            lbl = 'electron line'
        else:
            lbl = None
        plt.plot([op[ii, 0], ip[ii, 0]], [op[ii, 1], ip[ii, 1]], '.-',
                 linewidth=linewidth, color=[0, .7, 0], label=lbl)
        qtt.pgeometry.plotLabels(np.array((op[ii, :] + ip[ii, :]) / 2).reshape((2, -1)), '%d' % ii)
