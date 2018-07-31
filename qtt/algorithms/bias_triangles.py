# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:33:36 2018

@author: amjzwerver
"""

#%%

import numpy as np
import qcodes
import qtt
import qtt.pgeometry
import matplotlib.pyplot as plt

from qcodes.plots.qcmatplotlib import MatPlot
from qtt.data import diffDataset
from qtt.instrument_drivers.parameter_scaler import ParameterScaler as vt


def plotAnalysedLines(clicked_pts, linePoints1_2, linePt3_vert, linePt3_horz, linePt3_ints, intersect_point):
    """ Plots lines based on three points clicked
    
    Args:
        clicked_pts (array): lines between the three points (1-2), (2-3)
        linePoints1_2 (array): line fitted through points 1 and 2
        linePt3_vert (array): vertical line through point 3
        linePt3_horz (array): horizontal line through point 3
        linePt3_ints (array): line through point 3 and its vert/horz intersection 
                                with the line through point 1,2
        intersect_point (array): intersection point point 3, line1_2
    """
    qtt.pgeometry.plot2Dline(linePoints1_2, ':c', alpha = .5)
    qtt.pgeometry.plot2Dline(linePt3_vert, ':b', alpha=.4)
    qtt.pgeometry.plot2Dline(linePt3_horz, ':b', alpha=.4)
    qtt.pgeometry.plot2Dline(linePt3_ints, ':b', alpha=.4)
    
    qtt.pgeometry.plotPoints(intersect_point, '.b')
    qtt.pgeometry.plotPoints(clicked_pts[:,2:3], '.b')
    
    linePt3_ints_short = np.column_stack((intersect_point, clicked_pts[:,2:3]))
    qtt.pgeometry.plotPoints(linePt3_ints_short, 'b')
    
     
def perpLineIntersect(ds, description, vertical = True):
    """ Takes three points in a graph and calculates the length of a linepiece 
        between a line through points 1,2 and a vertical/horizontal line
        through the third point. Uses the currently active figure.
        
        Args:
            ds (dataset): dataset
            vertical (bool): find intersection of point with line vertically (True) 
            or horizontally (False)
            description: 
        
        Returns:
            (dict): 'intersection_point' = intersetcion point
                    'distance' = length of line from 3rd clicked point to line
                    through clicked points 1 and 2
                    'clicked_points' = coordinates of the three clicked points
    """
    diffDataset(ds, diff_dir='xy')
    plt.figure(588); plt.clf()
    MatPlot(ds.diff_dir_xy, num = 588)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    ax.set_xlabel(ax.get_xlabel()[:2])
    ax.set_ylabel(ax.get_ylabel()[:2])
    
#    ax = plt.gca()
#    ax.set_autoscale_on(False)
    if description == 'lever_arm' and vertical == True:
        print('''Please click three points;
            Point 1: on the addition line for the dot represented on the vertical axis
            Point 2: further on the addition line for the dot represented on the vertical axis
            Point 3: on the triple point at the addition line for the dot represented on the horizontal axis
            where both dot levels are aligned''')
    elif description == 'lever_arm' and vertical == False:
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
        print('''Please make sure that the descirption argument of this function
              is either 'lever_arm' or 'E_charging' ''')
    
    clicked_pts=qtt.pgeometry.ginput(3, '.c')
    
    qtt.pgeometry.plotPoints(clicked_pts, ':c')
    qtt.pgeometry.plotLabels(clicked_pts)
    
    linePoints1_2 = qtt.pgeometry.fitPlane( clicked_pts[:, 0:2].T )
    
    yy = clicked_pts[:,[2, 2]]; yy[1, -1] += 1
    line_vertical = qtt.pgeometry.fitPlane( yy.T )

    xx = clicked_pts[:,[2, 2]]; xx[0, -1] += 1
    line_horizontal = qtt.pgeometry.fitPlane( xx.T )
        
    if vertical == True:
        i = qtt.pgeometry.intersect2lines(linePoints1_2, line_vertical)
        intersectPoint = qtt.pgeometry.dehom(i[1])
        line = intersectPoint[:,[0,0]]; line[0,-1]+=1
    else:
        i = qtt.pgeometry.intersect2lines(linePoints1_2, line_horizontal)
        intersectPoint = qtt.pgeometry.dehom(i[1])
        line = intersectPoint[:,[0,0]]; line[1,-1]+=1
    
    linePt3_ints = qtt.pgeometry.fitPlane(line.T)
    line_length = np.linalg.norm(intersectPoint - clicked_pts[:,2:3])
    
    # visualize
    plotAnalysedLines(clicked_pts, linePoints1_2, line_vertical, line_horizontal, linePt3_ints, intersectPoint)
    
    return {'intersection_point': intersectPoint, 'distance': line_length, 'clicked_points': clicked_pts}

#def intersect2lines(l1, l2):
#    """ Caculate intersection between 2 lines """
#    r = qtt.pgeometry.null(np.vstack( (l1, l2)) )
#    a = qtt.pgeometry.dehom(r[1])
#    return a 

def lever_arm(bias, results, fig = None):
    """ Calculates the lever arm of a dot by using bias triangles in charge sensing. Uses currently active figure.
    
    Args:
        bias (int/float): bias between source and drain while taking the bias triangles
        results (dict): dictionary returned from the function perpLineIntersect
                        containing three points, the intersection point
                        between a line through 1,2 and the third point and the
                        length from points 3 to the intersection (horz/vert)
        fig (bool): adds lever arm to title of already existing figure with points
        
    Returns:
        lev_arm (float): the lever arm of the assigned dot in uV/mV
    """
    line_length = results['distance']
    
    #in uV/mV
    lev_arm = abs(bias/line_length)
    
    if fig and len(plt.get_fignums()) != 0:
        ax = plt.gca()
        ax.set_autoscale_on(False)
        title = 'Lever arm %s:   %.2f $\mu$eV/mV'%(ax.get_xlabel()[:2], lev_arm)
        plt.annotate('Length %s: %.2f mV'%(ax.get_xlabel()[:2], line_length), xy = (0.05, 0.1), xycoords='axes fraction', color = 'k')
        plt.annotate(title, xy = (0.05, 0.05), xycoords='axes fraction', color = 'k')
        ax.set_title(title)
        
    return lev_arm


def E_charging(lev_arm, results, fig = None):
    """
    Calculates the charging energy of a dot by using charge stability diagrams.
    Uses currently active figure.
    
    Args:
        lev_arm (int/float): lever arm for the gate to the dot
        results (dict): dictionary returned from the function perpLineIntersect
                        containing three points, the intersection point
                        between a line through 1,2 and the third point and the
                        length from points 3 to the intersection (horz/vert)
       fig (bool): adds charging energy to title of already existing figure with points
        
    Returns:
        E_charging (float): the charging energy for the dot
    """

    line_length = results['distance']
    E_c = line_length * lev_arm
    
    if fig and len(plt.get_fignums()) != 0:
        ax = plt.gca()
        ax.set_autoscale_on(False)
        title = 'E_charging %s: %.2f meV'%(ax.get_xlabel()[:2], E_c/1000)
        plt.annotate('Length %s:  %.2f mV'%(ax.get_xlabel()[:2], line_length), xy = (0.05, 0.1), xycoords='axes fraction', color = 'k')
        plt.annotate(title, xy = (0.05, 0.05), xycoords='axes fraction', color = 'k')
        ax.set_title(title)
    
    return E_c    

   
#%%
    
dataset = r'D:\data\amjzwerver\Qubyte2_cooldown4\E_add\2018-07-28\14-33-26_qtt_scan2Dfast'
ds = qcodes.load_data(dataset)

diffDataset(ds, diff_dir='xy')

plt.figure(3); plt.clf()
MatPlot(ds.measured, num = 3)

  
#%%
ll = perpLineIntersect(ds, vertical = True, description = 'lever_arm')
#%%
lev_arm = lever_arm(200, vertical = True)
#%%
E_c = E_charging(lev_arm, results = ll, fig = True)
