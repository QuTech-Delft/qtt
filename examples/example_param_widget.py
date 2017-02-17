# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:48:56 2016

@author: eendebakpt
"""

#%% Make update widget


from qtt.parameterviewer import ParameterViewer, createParameterWidget
from qtt.qtt_toymodel import VirtualIVVI, VirtualMeter

#%% Create a (virtual) instrument

if __name__=='__main__':

    gates = VirtualIVVI(name='ivvi', model=None)
    print(gates)


#%% Create a parameter widget for the instrument

if __name__=='__main__':
    w=createParameterWidget([gates])
    w.setGeometry(1600,110,300,540); # p.show()
    
