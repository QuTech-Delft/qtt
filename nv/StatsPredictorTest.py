# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:50:35 2017

@author: Laurens
"""

from StatsPredictor import StatsPredictor
import numpy as np
import pandas as pd
import qcodes
import os

#%%
print('Generating Data')
data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T
data=data[:,0:6]
df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump'])
jumps = df[['gate jump', 'yellow jump']]

labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))
labels[labels==-1] = 5 #this makes it a bit nicer to handle

