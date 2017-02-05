# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:55:39 2017

@author: Laurens
"""

from scipy.fftpack import rfft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
import os
import qcodes
import pandas as pd
from pynufft import pynufft # pip3 install pynufft --user
from scipy.interpolate import interp1d
import matplotlib
import nufftpy

#%%
print('Generating Data')
data = np.load(os.path.join(qcodes.config['user']['nvDataDir'],'jdata.npy')).T

df=pd.DataFrame(data, columns=['time', 'gate', 'yellow', 'new', 'gate jump', 'yellow jump','jump index'])
#plt.figure(300); plt.clf()
#df.plot(kind='scatter', x='gate jump', y='yellow jump', ax=plt.gca(), linewidths=0)

labels=np.load(os.path.join(qcodes.config['user']['nvDataDir'],'labels.npy'))

#%% Check the time steps
time = df[['time']].values.ravel()
dTime = [time[i+1]-time[i] for i in range(time.size-1)]

#%% interpolation test
interp = interp1d(df[['time']].values.ravel(),df[['gate']].values.ravel(), kind='nearest')
x=range(8000,10000,10)

ind = np.where((time>=8000) & (time<=10000))

plt.figure()
plt.scatter(time[ind],df[['gate']].values.ravel()[ind])
plt.plot(x,interp(x))

#%% fft as is
gvf=rfft(df[['gate']].values.ravel())
plt.figure()
plt.title('Fourier transformation of gate voltage \"Uniform\"')
plt.plot(gvf[1:])


#%% fft on the intrepolated data
# reconstruction parameters
Nd =(time.size,) # image space size
Kd =(time.size,) # k-space size
Jd =(1,) # interpolation size
# initiation of the object
NufftObj = pynufft.pynufft()
NufftObj.plan(np.reshape(time,(np.size(time),1)),Nd,Kd,Jd)
gateData=df[['gate']].values.ravel()

gdf=NufftObj.forward(gateData)

#xTicks = range(gdf.size)


plt.figure()
plt.subplot(211)
plt.title('Fourier transformation of gate voltage - irregular')
plt.plot(gdf)

# initiation of the object
NufftObj = pynufft.pynufft()
NufftObj.plan(np.reshape(time,(np.size(time),1)),Nd,Kd,Jd)
gateData=df[['yellow']].values.ravel()

gdf=NufftObj.forward(gateData)

plt.subplot(212)
plt.title('Fourier transformation of yellow frequency - irregular')
plt.plot(gdf)

#%% Trying out a hopefully more usefull nufft
freqBins = 600

test = nufftpy.nufft1(time,df[['yellow']].values.ravel(),freqBins)
xt = nufftpy.nufftfreqs(freqBins)[1:freqBins:25]

plt.figure()
plt.plot(test)
plt.xticks(range(freqBins)[1:freqBins:25], list(xt))
plt.title('non-Uniform Fourier transform yellow frequency')
# So that peak probably was just the centre fold, it is weird that the peak isn't centered for the other
# plot, and it is also strange that the other plot is not symmetric