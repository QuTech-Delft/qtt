#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:30:27 2017

@author: eendebakpt
"""

#%% Load packages
from imp import reload
import sys
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import pgeometry

import qcodes
import qtt
reload(qtt)
from qtt import pmatlab as pgeometry
import nvtools
from nvtools.nvtools import plotSection
from nvtools.nvtools import avg_steps, fmt
import sklearn

interpolated = False

#%% Load data
jumpdata = np.load(os.path.join(qcodes.config['user']['nvDataDir'], 'jdata.npy')).T
allData = np.load(os.path.join(qcodes.config['user']['nvDataDir'], 'jdata-alldata.npy')).T
df = pd.DataFrame(jumpdata[:, 0:6], columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump'])
labels = np.load(os.path.join(qcodes.config['user']['nvDataDir'], 'labels.npy'))

time = allData[:, 0]
yellow = allData[:, 1]
gate = allData[:, 2]
dt = np.median(np.diff(time))
jumpSelect = (np.diff(time) > 3 * dt) & (np.diff(time) < 30 * 60)
jumpSelect = np.append(jumpSelect, False)

#%% Plot data, add callback for easy viewing

import pgeometry
from pgeometry import plotCallback

plt.close(60)
ScatterFig = plt.figure(60)
plt.clf()
plt.jet()
df.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), c=labels, cmap=matplotlib.cm.jet, linewidths=0, colorbar=False)
plt.title('Click on figure to view data')
Fig = plt.figure(10)
plt.clf()
pgeometry.tilefigs([ScatterFig, Fig])


def f(plotidx, *args, **kwargs):
    verbose = kwargs.get('verbose', 1)
    if verbose:
        print('plotidx = %s' % plotidx)
    plt.figure(Fig.number)
    plt.clf()
    dataidx = int(jumpdata[plotidx, 6])
    plotSection(allData, list(range(dataidx - 60, dataidx + 100)), jumpSelect, si=dataidx)
    plt.pause(1e-4)

pc = plotCallback(func=f, xdata=df['gate jump'], ydata=df['yellow jump'], scale=[1, 100], verbose=0)
cid = ScatterFig.canvas.mpl_connect('button_press_event', pc)

#%% Example of very simple approach to classification: just probabilities

encoder = sklearn.preprocessing.LabelEncoder()
encoder.fit(labels)
lx = encoder.transform(labels)

char_indices = encoder.transform([-1, 0, 1, 2, 3, 4])
char_map = dict((c, i) for i, c in enumerate(chars))

bc = np.bincount(lx)
prob = bc / bc.sum()
print('probabilities %s' % fmt(prob))

y_pred = np.tile(prob, (len(labels), 1))
print('  avg number of steps: %.3f' % avg_steps(lx, y_pred))


#%% Only select 1 and 2 labels

idx = np.logical_or(labels == 1, labels == 2)
rlabels = labels[labels != 0]
rlabels = labels[idx]

plt.figure(10)
plt.clf()
plt.plot(rlabels, '.-b')

# note: the 1 and 2 clusters are pretty alternating, let's use this in our prediction

#%% Also simple: after a 1 first a 2 before a new 1, etc.

onetwo = np.zeros(len(labels))
threefour = np.zeros(len(labels))
for ii, l in enumerate(labels):
    if ii == 0:
        continue
    if l == 1:
        onetwo[ii] = 1
    elif l == 2:
        onetwo[ii] = 2
    else:
        onetwo[ii] = onetwo[ii - 1]
    if l == 3:
        threefour[ii] = 3
    elif l == 4:
        threefour[ii] = 4
    else:
        threefour[ii] = threefour[ii - 1]

y_pred = np.tile(prob, (len(labels), 1))
y_pred[(onetwo == 1).nonzero()[0] - 1, char_map[2]] = 0
y_pred[(onetwo == 2).nonzero()[0] - 1, char_map[1]] = 0
y_pred[(threefour == 3).nonzero()[0] - 1, char_map[4]] = 0
y_pred[(threefour == 4).nonzero()[0] - 1, char_map[3]] = 0
print('  avg number of steps: %.3f' % avg_steps(lx, y_pred))
