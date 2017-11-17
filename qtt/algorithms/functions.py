""" Mathematical functions and models """

import numpy as np
import scipy.constants


def gaussian(x, mean, s, amplitude=1, offset=0):
    """ Model for Gaussuan function

       $$y = offset + amplitude * np.exp(-(1/2)*(x-mean)^2/s^2)$$

    Args:
        x (array): data points
        mean, std, amplitude, offset: parameters
    Returns:
        y (array)

    """
    y = offset + amplitude * np.exp(- (x - mean) * (x - mean) / (2 * s * s))
    return y


def exp_function(x, a, b, c):
    """ Model for exponential function

       $$y = a + b * np.exp(-c * x)$$

    Args:
        x (array): data points
        a, b, c: parameters
    Returns:
        y (array)

    """
    y = a + b * np.exp(-c * x)
    return y


def linear_function(x, a, b):
    """ Linear function with offset"""
    return a * x + b


def Fermi(x, cc, A, T, kb=1):
    """ Fermi distribution 

    Arguments:
        x (numpy array): independent variable
        cc (float): center of Fermi distribution
        A (float): amplitude of Fermi distribution
        T (float): temperature Fermi distribution
        kb (float, default: 1): temperature scaling factor

    Returns:
        y (numpy array): value of the function

    .. math::

        y =  A*(1/ (1+\exp( (x-cc)/(kb*T) ) ) )
    """
    y = A * 1. / (1 + np.exp((x - cc) / (kb * T)))
    return y


def FermiLinear(x, a, b, cc, A, T, l=1.16):
    """ Fermi distribution with linear function added 

    Arguments:
        x (numpy array): independent variable 
        a, b (float): coefficients of linear part
        cc (float): center of Fermi distribution
        A (float): amplitude of Fermi distribution
        T (float): temperature Fermi distribution in mili Kelvin
        l (float): leverarm divided by kb

    The default value of the leverarm is
        (100 ueV/mV)/kb = (100*1e-6*scipy.constants.eV )/kb = 1.16.

    For this value the input variable x should be in mV, the 
    temperature T in K. We input the leverarm divided by kb for numerical stability.

    Returns:
        y (numpy array): value of the function

    .. math::

        y = a*x + b + A*(1/ (1+\exp( l* (x-cc)/(T) ) ) )

    """
    y = a * x + b + A * 1. / (1 + np.exp(l * (x - cc) / (T)))
    return y


def logistic(x, x0=0, alpha=1):
    """ Logistic function

    Arguments:
    x : array
        values
    x0, alpha : float
        parameters of function

    Example
    -------

    >>> y=logistic(0, 1, alpha=1)
    """
    f = 1 / (1 + np.exp(-2 * alpha * (x - x0)))
    return f
