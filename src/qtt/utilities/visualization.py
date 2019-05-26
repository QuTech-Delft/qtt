from typing import Optional, Any
import matplotlib.pyplot as plt
import numpy as np

import qtt.algorithms.functions


def plot_horizontal_line(x: float, color : str ='c', alpha : float =.5, label : Optional[str] =None) -> Any:
    """ Plot vertical alignment line
    
    Args:
        x: Position on horizontal axis to plot the line
        color: Color specification of the line
        alpha: Value to use for the transparency of the line
        label: Label for the line
    Return:
        Handle to the plotted line
    """
    vline = plt.axhline(x, label=label)
    vline.set_alpha(alpha)
    vline.set_color(color)
    vline.set_linestyle('--')
    return vline

def plot_vertical_line(x: float, color : str ='c', alpha : float =.5, label : Optional[str] =None) -> Any:
    """ Plot vertical alignment line

    Args:
        x: Position on horizontal axis to plot the line
        color: Color specification of the line
        alpha: Value to use for the transparency of the line
        label: Label for the line
    Return:
        Handle to the plotted line

    """
    vline = plt.axvline(x, label=label)
    vline.set_alpha(alpha)
    vline.set_color(color)
    vline.set_linestyle('--')
    return vline

def plot_double_gaussian_fit(result_dict : dict, xdata : np.array) -> None:
    """ Plot a two Gaussians from a double Gaussian fit
    
    Args:
        result_dict: Result of the double Gaussian fitting
        xdata: Independent data
    """
    plt.plot(xdata, qtt.algorithms.functions.gaussian(xdata, *result_dict['left']), 'g', label='left' )
    _=plt.plot(xdata, qtt.algorithms.functions.gaussian(xdata, *result_dict['right']), 'r', label='right' )
    