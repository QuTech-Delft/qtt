""" Functionality to determine the awg_to_plunger ratio.

pieter.eendebak@tno.nl

"""

import copy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from qcodes_loop.plots.qcmatplotlib import MatPlot

import qtt.algorithms.images
import qtt.pgeometry
import qtt.utilities.imagetools
from qtt.algorithms.images import straightenImage
from qtt.data import get_dataset, plot_dataset
from qtt.measurements.scans import scan2Dfast, scanjob_t
from qtt.utilities.imagetools import cleanSensingImage

# %% Helper functions


def click_line(fig: Optional[int], show_points: bool = False) -> Tuple[float, float]:
    """ Define a line through two points. The points are choosen by clicking on a position in a plot, two times.
    Args:
        fig: number of figure to plot the points in, when None it will plot a figure.
        showpoints: If True, then plot the selected points in the figure
    Returns:
        offset (float): offset of the line.
        slope (float): slope of the line.
    """
    if fig is not None:
        plt.figure(fig)
    pts0, pts1 = plt.ginput(2)
    if (pts1[0] - pts0[0]) == 0:
        raise Exception('vertical line not implemented')
    slope = (pts1[1] - pts0[1]) / (pts1[0] - pts0[0])
    offset = (pts0[1] - slope * pts0[0])

    if show_points:
        pts = np.array([pts0, pts1]).T
        plt.plot(pts[0], pts[1], '.-g')
    return offset, slope

# %% Main functions


def measure_awg_to_plunger(station, gate, minstrument, scanrange=30, step=0.5):
    """ Performing a scan2Dfast measurement, same gate on both axis, where the one axis is sweeped with the awg
    and one axis is stepped with the DAC's. Measurement should be centred around an addition line. From the slope
    of the addition line the awg to plunger conversion factor can be checked with the function analyse_awg_to_plunger.

    Args:
        station (QCoDeS station): measurement setup.
        gate (str): gate for which the awg to plunger conversion.
        minstrument (str, int): list with the name of the measurement instrument (str), and the channel number (int).
        scanrange (float): sweep- and steprange (mV), making a square 2d measurement.
        step (float): stepsize (mV).

    Returns:
        result (dict): resultresult (dic): result dictionary of the function measure_awg_to_plunger,
            shape: result = {'type': 'awg_to_plunger', 'awg_to_plunger': None, 'dataset': ds.location}.

    """
    gates = station.gates
    value0 = gates.get(gate)
    scanjob = scanjob_t({'sweepdata': {'param': gate, 'range': scanrange}, 'stepdata': {
                        'param': gate, 'range': scanrange, 'step': step, 'start': value0 - scanrange / 2}})
    scanjob['minstrument'] = minstrument
    scanjob['minstrumenthandle'] = minstrument
    scanjob['Naverage'] = 500
    scanjob.setWaitTimes(station)
    scanjob['wait_time_startscan'] += .5

    ds = scan2Dfast(station, scanjob)
    result = {'type': 'awg_to_plunger', 'awg_to_plunger': None, 'dataset': ds.location}
    return result


def analyse_awg_to_plunger(result: dict, method: str = 'hough', fig: Optional[int] = None) -> dict:
    """ Determine the awg_to_plunger conversion factor from a 2D scan, two possible methods: 'hough' it fits the slope
        of the addition line and calculates the correction to the awg_to_plunger conversion factor from there. if this
        doesn't work for some reason, method 'click' can be used to find the addition lines by hand/eye.

    Args:
        result: result dictionary of the function measure_awg_to_plunger,
            shape: result = {'type': 'awg_to_plunger', 'awg_to_plunger': None, 'dataset': ds.location}.
        method: either 'hough' or 'click'.
        fig : determines of the analysis staps and the result is plotted.

    Returns:
        result: including to following entries:
            angle (float): angle in radians.
            angle_degrees (float): angle in degrees.
            correction of awg_to_plunger (float): correction factor.
            dataset (str): location where the dataset is stored.
            type(str): type of calibration, 'awg_to_plunger'.

    """
    # getting the dataset from the result from the measure_awg_to_plunger function
    if result.get('type') != 'awg_to_plunger':
        raise AssertionError('not of type awg_to_plunger!')

    ds = get_dataset(result)

    # choosing a method;
    # the method 'hough' fits the addition line
    if method == 'hough':
        import cv2
        im, tr = qtt.data.dataset2image(ds)
        imextent = tr.scan_image_extent()
        istep = tr.scan_resolution()
        _, r = qtt.algorithms.images.straightenImage(
            im, imextent, mvx=istep, mvy=None)
        H = r[4]

        imc = cleanSensingImage(im, sigma=0.93, dy=0)

        imx, _ = straightenImage(imc, imextent, mvx=istep, verbose=0)

        imx = imx.astype(np.float64) * \
            (100. / np.percentile(imx, 99))  # scale image

        gray = qtt.pgeometry.scaleImage(imx)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # discard any edge effects
        edges[0,:]=edges[-1,:]=edges[:,-1]=edges[:,0]=0

        lines = cv2.HoughLines(edges, 1, np.pi / 180, int(gray.shape[0] * .5))
        if lines is None:
            angle_pixel = None
            angle = None
            xscan = None
            angle_deg = None
            correction = None
        else:
            angles = lines[:, 0, 1]
            angle_pixel = angles[0]  # take most voted line

            fac = 2
            xpix = np.array([[0, 0], [-fac * np.sin(angle_pixel), fac * np.cos(angle_pixel)]]).T
            tmp = qtt.pgeometry.projectiveTransformation(np.linalg.inv(H), xpix)
            xscan = tr.pixel2scan(tmp)

            def vec2angle(v):
                return np.arctan2(v[0], v[1])
            angle = vec2angle(xscan[:, 1] - xscan[:, 0])
            correction = -1 / np.tan(angle)
            angle_deg = np.rad2deg(angle)

        # plotting the analysis steps of the data
        if fig is not None:
            plt.figure(fig + 1)
            plt.clf()
            plt.imshow(gray)
            plt.axis('image')

            if angle_pixel is not None:
                for offset_in_pixels in [-40, -20, 0, 20, 40]:
                    label = None
                    if offset_in_pixels == 0:
                        label = 'detected angle'
                    qtt.pgeometry.plot2Dline(
                        [np.cos(angle_pixel), np.sin(angle_pixel), offset_in_pixels], 'm', label=label)
            if angle is not None:
                plt.title(f'Detected line direction: angle {angle_deg:.2f} [deg]')

            plt.figure(fig + 2)
            plt.clf()
            plt.imshow(edges)
            plt.axis('image')
            plt.title('Detected edge points')

    # the method click relies on the user clicking two points to indicate the addition line
    elif method == 'click':
        if fig is not None:
            plt.figure(fig)
            plt.clf()
            MatPlot(ds.default_parameter_array(), num=fig)
            plt.draw()
            plt.pause(1e-3)

        print("Please click two different points on the addition line")
        offset, slope = click_line(fig=fig)

        angle_pixel = None
        angle = -np.pi / 2 - np.tanh(slope)
        correction = -1 / np.tan(angle)
        angle_deg = angle / (2 * np.pi) * 360

    else:
        raise Exception('method %s not implemented' % (method,))

    # filling the result dictionary
    result = copy.copy(result)
    result['_angle_pixel'] = angle_pixel
    result['angle'] = angle
    result['angle_degrees'] = angle_deg
    result['correction of awg_to_plunger'] = correction
    if method == 'click':
        result['slope'] = slope
        result['offset'] = offset

    # optional, plotting figures showing the analysis
    if fig is not None:
        plot_awg_to_plunger(result=result, fig=fig)

    return result


def plot_awg_to_plunger(result, fig=10):
    """ This function tests the analyse_awg_to_plunger function. Plotting is optional and for debugging purposes.

    Args:
        result (dict): result dictionary from the analyse_awg_to_plunger function.
        fig (int): index of matplotlib window.

    """
    if not result.get('type', None) == 'awg_to_plunger':
        raise Exception('calibration result not of correct type ')

    angle = result['angle']

    ds = get_dataset(result)
    v = ds.default_parameter_array()
    dy = v.set_arrays[0][-1]-v.set_arrays[0][0]
    centre = [np.mean(v.set_arrays[1]), np.mean(v.set_arrays[0])]

    plot_dataset(ds, fig=fig)
    if angle is not None:
        rho = -(centre[0] * np.cos(angle) - np.sin(angle) * centre[1])

        for offset in dy*np.linspace(-1, 1, 7):
            label = None
            if offset == 0:
                label = 'detected angle'
            qtt.pgeometry.plot2Dline(
                [np.cos(angle), -np.sin(angle), rho + offset], '--m', alpha=.6, label=label)
        angle_deg = np.rad2deg(angle)
        plt.title(f'Detected line direction: angle {angle_deg:.2f} [deg]')
