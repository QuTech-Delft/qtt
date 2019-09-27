# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:04:41 2019

@author: eendebakpt
"""

from typing import Dict, Any, List, Union
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

import qcodes
from qtt.utilities.visualization import plot_vertical_line


def generate_allxy_combinations() -> List[Any]:
    """ Generate all combinations of the AllXY sequence from Reed 2013 """
    xymapping = {'I': 'I', 'x': 'X90', 'y': 'Y90', 'X': 'X180', 'Y': 'Y180'}
    allxy_combinations_input = ['II', 'XX', 'YY', 'XY', 'YX'] + ['xI', 'yI', 'xy', 'yx',
                                                           'xY', 'yX', 'Xy', 'Yx', 'xX', 'Xx', 'yY', 'Yy'] + ['XI', 'YI', 'xx', 'yy']
    allxy_combinations = [(xymapping[x[0]], xymapping[x[1]]) for x in allxy_combinations_input]

    return allxy_combinations


def allxy_model(x: Union[float, np.ndarray], offset0:float, slope0:float, offset1:float, slope1:float, offset2:float, slope2:float) -> Union[float, np.ndarray]:
    """ Model for AllXY experiment

    The model consists of three linear segments

    """
    x = np.array(x)
    x0 = x < 4.5
    x1 = np.logical_and(x >= 4.5, x < 16.5)
    x2 = (x >= 16.5)

    v1 = x0 * (offset0 + x * slope0)

    v2 = x1 * (offset1 + x * slope1)
    v3 = x2 * (offset2 + x * slope2)

    return v1 + v2 + v3


def _estimate_allxy_parameters(allxy_data: np.ndarray):
    """ Return estimate of allxy model parameters """
    p = [np.mean(allxy_data[0:5]), 0, np.mean(allxy_data[5:17]), 0, np.mean(allxy_data[17:]), 0]
    return p


def _default_measurement_array(dataset):
    mm = [name for (name, a) in dataset.arrays.items() if not a.is_setpoint]
    return dataset.arrays[mm[0]]


def fit_allxy(dataset: qcodes.DataSet, initial_parameters: np.ndarray = None) -> Dict[str, Any]:
    """ Fit AllXY measurement to piecewise linear model 
    
    Args:
        dataset: Dataset containing the AllXY measurement
        initial_parameters: Optional set of initialization parameters
    Returns:
        Dictionary with the fitting results
    """
    allxy_data = _default_measurement_array(dataset)

    x_data = np.arange(21)
    lmfit_model = Model(allxy_model)

    if initial_parameters is None:
        initial_parameters = _estimate_allxy_parameters(allxy_data)
    param_names = lmfit_model.param_names
    result = lmfit_model.fit(allxy_data, x=x_data, **dict(zip(param_names, initial_parameters)), verbose=0)
    fitted_parameters = np.array([result.best_values[p] for p in param_names])
    return {'fitted_parameters': fitted_parameters, 'description': 'allxy fit', 'initial_parameters': initial_parameters}


def plot_allxy(dataset: qcodes.DataSet, result: dict, fig: int = 1, verbose: int = 0):
    """ Plot the results of an AllXY fit
    
    """
    allxy_data = _default_measurement_array(dataset)
    xy_pairs = generate_allxy_combinations()
    x_data = np.arange(21)

    plt.figure(fig)
    plt.clf()
    fitted_parameters = result['fitted_parameters']
    xfine = np.arange(0, 21, 1e-3)
    plt.plot(xfine, allxy_model(xfine, *fitted_parameters), 'm', label='fitted allxy', alpha=.5)

    plt.plot(x_data, allxy_data, '.b', label='allxy data')

    if verbose:
        p = [0, 0, .5, 0, 1, 0]
        plt.plot(xfine, allxy_model(xfine, *p), 'c', label='baseline', alpha=.5)

        initial_params = _estimate_allxy_parameters(allxy_data)
        plt.plot(xfine, allxy_model(xfine, *initial_params), 'g', label='initial estimate', alpha=.35)

        initial_parameters = result['initial_parameters']
        plt.plot(xfine, allxy_model(xfine, *initial_parameters), ':g', label='initial estimate', alpha=.35)

    plt.xticks(x_data, [v[0] + "," + v[1] for v in xy_pairs], rotation='vertical')
    vl = plot_vertical_line(4.5)
    vl.set_linestyle(':')
    vl = plot_vertical_line(16.5)
    vl.set_linestyle(':')
    plt.title('AllXY')

    plt.legend()
