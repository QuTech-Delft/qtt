# -*- coding: utf-8 -*-
""" Functionality to test fit PAT models.

For more details see: https://arxiv.org/abs/1803.10352

"""

# %% Load packages
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from qtt.pgeometry import pcolormesh_centre
from qtt.algorithms.pat_fitting import one_ele_pat_model, fit_pat, pre_process_pat, plot_pat_fit, detect_peaks
# %%


class TestPatFitting(TestCase):

    @staticmethod
    def test_pat_fitting(fig=100):
        pp0 = [0, 50, 5]
        x_data = np.arange(-3, 3, 0.025)
        y_data = np.arange(.5e9, 10e9, .5e9)
        z_data = 1.2 * np.random.rand(y_data.size, x_data.size)
        background = np.zeros(x_data.size)
        for ii, x in enumerate(x_data):
            y = one_ele_pat_model(x, pp0)
            jj = np.argmin(np.abs(y_data - y))
            z_data[jj, ii] += 2  # ((x>0)*2-1)
        pp, _ = fit_pat(x_data, y_data, z_data, background, verbose=0)
        imx, imq, _ = pre_process_pat(x_data, y_data, background, z_data, fig=fig)

        pat_fit_fig = plt.figure(fig)
        plt.clf()
        plot_pat_fit(x_data, y_data, imq, pp, fig=pat_fit_fig.number, label='fitted model')
        plot_pat_fit(x_data, y_data, None, pp0, fig=pat_fit_fig.number, label='initial model')

        trans = 'one_ele'
        period = 1e-3
        xx, _ = detect_peaks(x_data, y_data, imx, model=trans, period=period, sigmamv=.05, fig=fig+100)

        plt.figure(fig + 3)
        plt.clf()
        pcolormesh_centre(x_data, y_data, imq)
        plt.close('all')