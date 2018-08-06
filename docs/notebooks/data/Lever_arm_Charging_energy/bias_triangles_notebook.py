# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:16:32 2018

@author: amjzwerver
"""

import numpy as np
import qcodes
import qtt
import qtt.pgeometry
import matplotlib.pyplot as plt

from qcodes.plots.qcmatplotlib import MatPlot
from qtt.data import diffDataset
from qtt.instrument_drivers.parameter_scaler import ParameterScaler as vt
from qtt.algorithms.bias_triangles import plotAnalysedLines



def perpLineIntersect_ipynb(ds, description, vertical = True, points = None):
    """ Takes three points in a graph and calculates the length of a linepiece 
        between a line through points 1,2 and a vertical/horizontal line
        through the third point. Uses the currently active figure.
        
        Args:
            ds (dataset): dataset
            vertical (bool): find intersection of point with line vertically (True) 
            or horizontally (False)
            description: type of measurement, either 'lever_arm' or 'E_charging'
        
        Returns:
            (dict): 'intersection_point' = intersetcion point
                    'distance' = length of line from 3rd clicked point to line
                    through clicked points 1 and 2
                    'clicked_points' = coordinates of the three clicked points
    """
    diffDataset(ds, diff_dir='xy')
    plt.figure(588); plt.clf()
    MatPlot(ds.diff_dir_xy, num = 588)
    plt.suptitle('Clicked points')
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
    
    if points is not None:
        clicked_pts = points
    else:
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
        intersectPoint = qtt.pgeometry.dehom(i)
        line = intersectPoint[:,[0,0]]; line[0,-1]+=1
    else:
        i = qtt.pgeometry.intersect2lines(linePoints1_2, line_horizontal)
        intersectPoint = qtt.pgeometry.dehom(i)
        line = intersectPoint[:,[0,0]]; line[0,-1]+=1

    linePt3_ints = qtt.pgeometry.fitPlane(line.T)
    line_length = np.linalg.norm(intersectPoint - clicked_pts[:,2:3])
    
    # visualize
    plotAnalysedLines(clicked_pts, linePoints1_2, line_vertical, line_horizontal, linePt3_ints, intersectPoint)
    
    return {'intersection_point': intersectPoint, 'distance': line_length, 'clicked_points': clicked_pts}
