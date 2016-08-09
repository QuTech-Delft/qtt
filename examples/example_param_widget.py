# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:48:56 2016

@author: eendebakpt
"""

#%% Make update widget

import qtpy.QtCore as QtCore
import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets

if __name__=='__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except:
        pass

from qtt.parameterviewer import *


#name='L'
if 0:
    p=ParameterViewer(instruments=[gates], instrumentnames=['ivvi'])

    p.setGeometry(1700,110,200,40); p.show()
    self=p


#%% Local test

if __name__=='__main__':
    server_name='s2'
    #server_name=None
    from qtt.qtt_toymodel import FourdotModel, VirtualIVVI, VirtualMeter
    gates = VirtualIVVI(name='ivvi', model=None, server_name=server_name)
    #gates=None



#%% Local option

if __name__=='__main__' and 0:
    w=createUpdateWidget([gates], doexec=False)
    

#%% Remote option

#%%

if __name__=='__main__':
    pass
  
    #p=mp.Process(target=createUpdateWidget)
    #p.start()

    p=mp.Process(target=createUpdateWidget, args=([gates],))
    p.start()

#%% 
#  FIXME: show2D
#  FIXME: 1dot_script
  
  