# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:48:56 2016

@author: eendebakpt
"""

#%% Make update widget


from qtt.gui.parameterviewer import ParameterViewer, createParameterWidget
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI, VirtualMeter

#%% Create a (virtual) instrument

gates = VirtualIVVI(name='ivvi', model=None)
print(gates)


#%% Create a parameter widget for the instrument

w = createParameterWidget([gates])
w.setGeometry(1600, 110, 300, 540)  # p.show()
