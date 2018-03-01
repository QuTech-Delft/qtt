# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:12:04 2018

@author: riggelenfv
"""

import numpy as np
import qcodes
from qtt.algorithms.fitting import FermiLinear, fitFermiLinear

#%%

def fit_addition_line(dataset, trimborder = True):
    """Fits a FermiLinear function to the addition line and finds the middle of the step.
    
    Args:
        dataset (qcodes dataset): 1d measured data of additionline
        trimborder (bool): determines if the edges of the data are taken into account for the fit
    
    Returns:
        m_addition_line (float): x value of the middle of the addition line
        pfit (array): fit parameters of the Fermi Linear function
        pguess (array): parameters of initial guess
        dataset_fit (qcodes dataset): dataset with fitted Fermi Linear function
        dataset_guess (qcodes dataset):dataset with guessed Fermi Linear function
        
    """
    y_array = dataset.default_parameter_array()
    setarray = y_array.set_arrays[0]
    xdata = np.array(setarray)
    ydata = np.array(y_array)

    if trimborder:
        ncut = max(min(int(xdata.size/40), 100),1)
        xdata, ydata, setarray =xdata[ncut:-ncut], ydata[ncut:-ncut], setarray[ncut:-ncut]

    # fitting of the FermiLinear function
    pp = fitFermiLinear(xdata, ydata, verbose=1, fig=None)
    pfit = pp[0] #fit parameters
    pguess = pp[1]['p0'] #initial guess parameters
    
    y0 = FermiLinear(xdata, *list(pguess) )
    dataset_guess = qcodes.DataArray(name='fit', label='fit', preset_data=y0,  set_arrays=(setarray,) )
    y = FermiLinear(xdata, *list(pfit) )
    dataset_fit = qcodes.DataArray(name='fit', label='fit', preset_data=y,  set_arrays=(setarray,) )
    
    m_addition_line = pfit[2]
    return  m_addition_line, pfit, pguess, dataset_fit, dataset_guess