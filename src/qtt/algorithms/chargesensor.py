""" Analyse effect of sensing dot on fitting of tunnel barrier

@author: eendebakpt
"""

# %% Load packages
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np

import qtt
import qtt.pgeometry
import qtt.algorithms.coulomb
from qtt import pgeometry

# %% Find range of sensing dot used


class DataLinearizer:

    def __init__(self, xsr, ysr):
        """ Class to linearize data

        The to data a smooth curve is fitted and a linear function is
        fitted at the centre point.

        Args:
            xsr, ysr (array): data containing the signal


        Internal variables:
            tck: fit the charge sensor slope
            tck_inv: inverse fit the charge sensor slope
            linfit: linear fit to charge sensor slope

        """
        self.xsr = xsr
        self.ysr = ysr

        tck = scipy.interpolate.splrep(xsr, ysr, s=0)
        tck_inv = scipy.interpolate.splrep(ysr, xsr, s=0)

        # linear fit
        self.linfit = np.poly1d(np.polyfit(xsr, ysr, 1))
        self.tck = tck
        self.tck_inv = tck_inv

    def show(self, fig=100):
        """ Plots forward_curve vs. xsr

        Args:
            fig(int): index of matplotlib window

        """
        plt.figure(fig)
        plt.clf()
        ysnew = self.forward_curve(self.xsr)
        plt.plot(self.xsr, ysnew, '-r', linewidth=4, label='spline')
        plt.plot(self.xsr, self.forward(self.xsr), ':k', label='linear fit')
        plt.legend(numpoints=1)

    def forward(self, x):
        """ Apply fitted linear transformation to data

        Args:
            x (array): data to be fit
        Returns:
            y (array): fitted data

        """
        y = self.linfit(x)
        return y

    def forward_curve(self, x):
        """ Evaluate a B-spline, see documentation scipy.interpolate.splev

        Args:
            x (array): to be fitted curve
        Returns:
            y (array): array representing the evaluated spline function

        """
        y = scipy.interpolate.splev(x, self.tck, der=0)
        return y

    def backward_curve(self, y):
        """ Evaluate a B-spline, see documentation scipy.interpolate.splev

        Args:
            y (array): to be fitted curve
        Returns:
            x (array): backward array representing the evaluated spline function

        """
        x = scipy.interpolate.splev(y, self.tck_inv, der=0)
        return x

    def rectify(self, y):
        """ Rectify data

        Args:
            y (array)
        Returns:
            yc (array): corrected data

        """
        x = self.backward_curve(y)
        yc = self.forward(x)
        return yc


def correctChargeSensor(xscan, yscan, xs, ys, fig=None):
    """ Calculate correction for a non-linear charge sensor

    Args:
        xscan, yscan: data of scan to be corrected
        xs, ys: scan of charge sensor response

    Returns:
        results (dict): dictionary with intermediate results

    """

    peaks = qtt.algorithms.coulomb.coulombPeaks(xs, ys, fig=fig, sampling_rate=1)

    try:
        # range for fit: select from the first detected peak
        xsr = xs[peaks[0]['pbottoml']:peaks[0]['p']]
        ysr = ys[peaks[0]['pbottoml']:peaks[0]['p']]
    except Exception as ex:
        # no good peaks, just take the entire range
        xsr = xs
        ysr = ys

    # smooth fit to SD curve
    dl = DataLinearizer(xsr, ysr)

    if fig is not None:
        dl.show(fig)
        minv, maxv = np.min(yscan), np.max(yscan)
        pgeometry.plot2Dline([0, -1, minv], '--c', label='range of scan')
        pgeometry.plot2Dline([0, -1, maxv], '--c', label=None)
#        plt.plot(xsr, linfit(xsr), ':k', label='linear fit')

    return dl, {'peaks': peaks}

