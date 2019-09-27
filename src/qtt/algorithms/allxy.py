# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:04:41 2019

@author: eendebakpt
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

import qtt
from qtt.utilities.visualization import plot_vertical_line
from sqt.measurements.pulse_generator import generate_allxy_combinations

def allxy_model (x, offset0, slope0, offset1, slope1, offset2, slope2):
    """ Model for AllXY experiment
    
    The model consists of three linear segments
    
    """
    x=np.array(x)
    x0=x<4.5
    x1 = np.logical_and(x>=4.5,  x<16.5)
    x2  = (x>=16.5)
    
    v1=x0 *(offset0+x*slope0)
    
    v2=x1 *(offset1+x*slope1)
    v3=x2 *(offset2+x*slope2)
    
    return v1+v2+v3


#xy_pairs = generate_allxy_combinations()


def _estimate_allxy_parameters(allxy_data : np.ndarray):
    """ Return estimate of allxy model parameters """
    p=[np.mean(allxy_data[0:5]),0,np.mean(allxy_data[5:17]), 0, np.mean(allxy_data[17:]),0]
    return p    


from typing import Dict, Any

def _default_measurement_array(dataset):
       mm = [ name for (name, a) in dataset.arrays.items() if not a.is_setpoint]       
       return dataset.arrays[mm[0]]

def fit_allxy(dataset, initial_parameters : np.ndarray = None) -> Dict[str, Any]:
    allxy_data = _default_measurement_array(dataset)

    x_data=np.arange(21)
    lmfit_model = Model(allxy_model)
    
    if initial_parameters is None:
        initial_parameters = _estimate_allxy_parameters(allxy_data)
    param_names = lmfit_model.param_names
    result = lmfit_model.fit(allxy_data, x=x_data, **dict(zip(param_names, initial_parameters)), verbose=0)
    fitted_parameters = np.array([result.best_values[p] for p in param_names])
    return {'fitted_parameters': fitted_parameters, 'description': 'allxy fit', 'initial_parameters': initial_parameters}


def plot_allxy(dataset, result, fig : int =1, verbose : int =0):
    allxy_data = _default_measurement_array(dataset)
    xy_pairs = generate_allxy_combinations()
    x_data=np.arange(21)
    
    plt.figure(fig);
    plt.clf()
    fitted_parameters = result['fitted_parameters']
    xfine=np.arange(0, 21, 1e-3)
    plt.plot(xfine, allxy_model(xfine, *fitted_parameters), 'm', label='fitted allxy', alpha=.5)

    plt.plot(x_data, allxy_data, '.b', label='allxy data')

    if verbose:
        p=[0,0,.5, 0, 1,0]
        plt.plot(xfine, allxy_model(xfine, *p), 'c', label='baseline', alpha=.5)
        
        initial_params=_estimate_allxy_parameters(allxy_data)
        plt.plot(xfine, allxy_model(xfine, *initial_params), 'g', label='initial estimate', alpha=.35)

        initial_parameters = result['initial_parameters']
        plt.plot(xfine, allxy_model(xfine, *initial_parameters), ':g', label='initial estimate', alpha=.35)

    
    plt.xticks(x_data, [v[0]+v[1] for v in xy_pairs], rotation='vertical' )    
    vl=plot_vertical_line(4.5)
    vl.set_linestyle(':')
    vl=plot_vertical_line(16.5); vl.set_linestyle(':')
    plt.title('AllXY')
    
    plt.legend()


 
         
