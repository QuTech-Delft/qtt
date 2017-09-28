# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:30:36 2017

@author: cjvandiepen
"""

#%%
import numpy as np
from collections import OrderedDict

#%%
def acfit_to_ccmap(ac_fits, plungers, barrier):
    """ Convert angles in anti-cross model to cross-capacitance matrix. 
    
    Note: this function only supports 2 plungers and one or no barrier gates.
    
    Args:
        ac_fits (array of tuples): scanned gates and array with the angles
        plungers (list): names of the plunger gates 
        barrier (list): name of the barrier gate
        
    Returns:
        cc_map (dict): describes the cross-capacitance matrix
    """
    gate_combs = [ac_fit[0] for ac_fit in ac_fits]
    gate_combs_tmp = gate_combs[:2]
    cc_map = OrderedDict()
    # fill in the first row
    cc_map['V'+plungers[0]] = {plungers[0]: 1} # re-scale the matrix
    plg = plungers[0]
    for idc, gate_comb in enumerate(gate_combs_tmp):
        if gate_comb[0] == plg:
            cc_map['V'+plg][gate_comb[1]] = 1 / np.tan((ac_fits[idc][1][0] + ac_fits[idc][1][2] - np.pi)/2)
    # fill in the rows for the other plungers
    for plg in plungers[1:]:
        cc_map['V'+plg] = {}
        # first the diagonal entries
        for idc, gate_comb in enumerate(gate_combs_tmp):
            if gate_comb[0] is plungers[0] and gate_comb[1] is plg:
                cc_map['V'+plg][plg] = (1 + 1/(np.tan((ac_fits[idc][1][0] + ac_fits[idc][1][2] - np.pi)/2)*np.tan(ac_fits[idc][1][4]))) / \
                      (1 / np.tan(ac_fits[idc][1][4]) + np.tan((ac_fits[idc][1][1] + ac_fits[idc][1][3] + np.pi)/2))
        # then the first column
        for idc, gate_comb in enumerate(gate_combs_tmp):
            if gate_comb == (plungers[0], plg):
                cc_map['V'+plg][plungers[0]] = cc_map['V'+plg][plg]*np.tan((ac_fits[idc][1][1] + ac_fits[idc][1][3] - np.pi)/2)
        # and then the third column
        for idc, gate_comb in enumerate(gate_combs_tmp):
            if gate_comb[0] == plungers[0] and gate_comb[1] not in plungers:
                cc_map['V'+plg][gate_comb[1]] = cc_map['V'+plg][plungers[0]]/np.tan((ac_fits[idc][1][1] + ac_fits[idc][1][3] - np.pi)/2)
    # fill in the rows for the barriers
    for bar_id in barrier:
        cc_map['V'+bar_id] = {bar_id: 1}
        for plg in plungers:
            cc_map['V'+bar_id][plg] = 0
    
    return cc_map