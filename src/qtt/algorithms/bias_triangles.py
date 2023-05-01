""" Functionality to analyse bias triangles

@author: amjzwerver
"""

# %%

import numpy as np
import qtt
import qtt.pgeometry
import matplotlib.pyplot as plt

from qcodes_loop.plots.qcmatplotlib import MatPlot
from qtt.data import diffDataset


def plotAnalysedLines(clicked_pts, linePoints1_2, linePt3_vert, linePt3_horz, linePt3_ints, intersect_point):
    """ Plots lines based on three points clicked.

    Args:
        clicked_pts (array): lines between the three points (1-2), (2-3).
        linePoints1_2 (array): line fitted through points 1 and 2.
        linePt3_vert (array): vertical line through point 3.
        linePt3_horz (array): horizontal line through point 3.
        linePt3_ints (array): line through point 3 and its vert/horz intersection 
                              with the line through point 1,2.
        intersect_point (array): intersection point point 3, line1_2.
    """
    qtt.pgeometry.plot2Dline(linePoints1_2, ':c', alpha=.5)
    qtt.pgeometry.plot2Dline(linePt3_vert, ':b', alpha=.4)
    qtt.pgeometry.plot2Dline(linePt3_horz, ':b', alpha=.4)
    qtt.pgeometry.plot2Dline(linePt3_ints, ':b', alpha=.4)

    qtt.pgeometry.plotPoints(intersect_point, '.b')
    qtt.pgeometry.plotPoints(clicked_pts[:, 2:3], '.b')

    linePt3_ints_short = np.column_stack((intersect_point, clicked_pts[:, 2:3]))
    qtt.pgeometry.plotPoints(linePt3_ints_short, 'b')


def perpLineIntersect(ds, description, vertical=True, points=None, fig=588, diff_dir='xy'):
    """ Takes three points in a graph and calculates the length of a linepiece 
        between a line through points 1,2 and a vertical/horizontal line
        through the third point. Uses the currently active figure.

        Args:
            ds (dataset): dataset with charge stability diagram and gate voltage in mV.
            description: TODO
            vertical (bool): find intersection of point with line vertically (True) 
                or horizontally (False).
            points (None or array): if None, then let the user select points.
            fig (int): figure number.
            diff_dir (None or 'xy'): specification of differentiation direction.

        Returns:
            (dict): 'intersection_point' = intersection point
                    'distance' = length of line from 3rd clicked point to line
                    through clicked points 1 and 2
                    'clicked_points' = coordinates of the three clicked points
    """

    if diff_dir is not None:
        diffDataset(ds, diff_dir='xy')
        array_name = 'diff_dir_xy'
    else:
        array_name = ds.default_parameter_name()
    plt.figure(fig)
    plt.clf()
    MatPlot(ds.arrays[array_name], num=fig)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    if description == 'lever_arm' and vertical:
        print('''Please click three points;
            Point 1: on the addition line for the dot represented on the vertical axis
            Point 2: further on the addition line for the dot represented on the vertical axis
            Point 3: on the triple point at the addition line for the dot represented on the horizontal axis
            where both dot levels are aligned''')
    elif description == 'lever_arm' and not vertical:
        print('''Please click three points;
            Point 1: on the addition line for the dot represented on the horizontal axis
            Point 2: further on the addition line for the dot represented on the horizontal axis
            Point 3: on the triple point at the addition line for the dot represented on the horizontal axis
            where both dot levels are aligned''')
    elif description == 'E_charging':
        print('''Please click three points;
            Point 1: on the (0, 1) - (0,2) addition line
            Point 2: further on the (0, 1) - (0,2) addition line
            Point 3: on the (0, 0) - (0, 1) addition line ''')
    else:
        # Do something here such that no three points need to be clicked
        raise Exception('''Please make sure that the description argument of this function
              is either 'lever_arm' or 'E_charging' ''')

    if points is not None:
        clicked_pts = points
    else:
        plt.title('Select three points')
        plt.draw()
        plt.pause(1e-3)
        clicked_pts = qtt.pgeometry.ginput(3, '.c')

    qtt.pgeometry.plotPoints(clicked_pts, ':c')
    qtt.pgeometry.plotLabels(clicked_pts)

    linePoints1_2 = qtt.pgeometry.fitPlane(clicked_pts[:, 0:2].T)

    yy = clicked_pts[:, [2, 2]]
    yy[1, -1] += 1
    line_vertical = qtt.pgeometry.fitPlane(yy.T)

    xx = clicked_pts[:, [2, 2]]
    xx[0, -1] += 1
    line_horizontal = qtt.pgeometry.fitPlane(xx.T)

    if vertical:
        i = qtt.pgeometry.intersect2lines(linePoints1_2, line_vertical)
        intersectPoint = qtt.pgeometry.dehom(i)
        line = intersectPoint[:, [0, 0]]
        line[0, -1] += 1
    else:
        i = qtt.pgeometry.intersect2lines(linePoints1_2, line_horizontal)
        intersectPoint = qtt.pgeometry.dehom(i)
        line = intersectPoint[:, [0, 0]]
        line[1, -1] += 1

    linePt3_ints = qtt.pgeometry.fitPlane(line.T)
    line_length = np.linalg.norm(intersectPoint - clicked_pts[:, 2:3])

    # visualize
    plotAnalysedLines(clicked_pts, linePoints1_2, line_vertical, line_horizontal, linePt3_ints, intersectPoint)

    return {'intersection_point': intersectPoint, 'distance': line_length, 'clicked_points': clicked_pts,
            'array_names': [array.name for array in ds.default_parameter_array().set_arrays]}


def lever_arm(bias, results, fig=None):
    """ Calculates the lever arm of a dot by using bias triangles in charge sensing. Uses currently active figure.

    Args:
        bias (float): bias in uV between source and drain while taking the bias triangles.
        results (dict): dictionary returned from the function perpLineIntersect
                        containing three points, the intersection point
                        between a line through 1,2 and the third point and the
                        length from points 3 to the intersection (horz/vert).
        fig (bool): adds lever arm to title of already existing figure with points.

    Returns:
        lev_arm (float): the lever arm of the assigned dot in uV/mV.
    """
    line_length = results['distance']

    # in uV/mV
    lev_arm = abs(bias / line_length)

    if fig and len(plt.get_fignums()) != 0:
        ax = plt.gca()
        ax.set_autoscale_on(False)
        if np.round(results['clicked_points'][0, 2], 2) == np.round(results['intersection_point'][0], 2):
            gate = ax.get_ylabel()
        else:
            gate = ax.get_xlabel()
        title = r'Lever arm %s:   %.2f $\mu$eV/mV' % (gate, lev_arm)
        plt.annotate('Length %s: %.2f mV' % (gate, line_length), xy=(0.05, 0.1), xycoords='axes fraction', color='k')
        plt.annotate(title, xy=(0.05, 0.05), xycoords='axes fraction', color='k')
        ax.set_title(title)

    return lev_arm


def E_charging(lev_arm, results, fig=None):
    """
    Calculates the charging energy of a dot by using charge stability diagrams.
    Uses currently active figure.

    Args:
        lev_arm (float): lever arm for the gate to the dot.
        results (dict): dictionary returned from the function perpLineIntersect
                        containing three points, the intersection point
                        between a line through 1,2 and the third point and the
                        length from points 3 to the intersection (horz/vert).
       fig (bool): adds charging energy to title of already existing figure with points.

    Returns:
        E_charging (float): the charging energy for the dot.
    """

    line_length = results['distance']
    E_c = line_length * lev_arm

    if fig and len(plt.get_fignums()) != 0:
        ax = plt.gca()
        ax.set_autoscale_on(False)
        if np.round(results['clicked_points'][0, 2], 2) == np.round(results['intersection_point'][0], 2):
            gate = ax.get_ylabel()[:2]
        else:
            gate = ax.get_xlabel()[:2]
        title = 'E_charging %s: %.2f meV' % (gate, E_c / 1000)
        plt.annotate('Length %s:  %.2f mV' % (gate, line_length), xy=(0.05, 0.1), xycoords='axes fraction', color='k')
        plt.annotate(title, xy=(0.05, 0.05), xycoords='axes fraction', color='k')
        ax.set_title(title)

    return E_c
