# -*- coding: utf-8 -*-
""" Example to solve bias-T compensation 

@author: eendebakpt
"""

#%% Load packages

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%% Define AWG pulse

samplerate=1e6
T=1e-3 # 1 ms pulse

# time points
t=np.arange(0, T, 1./samplerate)

q=int(samplerate*1e-4)
pulse=t.copy()
pulse[0:q]=5
pulse[q:2*q]=-2

if 1:
    # mirror
    pulse=np.hstack((pulse, -pulse))
    t=np.arange(0, pulse.size/samplerate, 1./samplerate)
    
#%% Model
#
# First order diff equation: V is voltage over capacitor, W is input voltage (e.g. voltage on AWG)
#
# dV/dt = - alpha (W-V)

# function that returns dy/dt
def model(y,t, alpha=600.):
    tt=int(samplerate*t)
    tt=max(tt,0)
    tt=min(tt, pulse.size-1)
    W = pulse[tt]
    dydt = -alpha * (y-W)
    print('t %.1f [micros], W %.3f, dydy %.1f' % (1e6*t, W, dydt) )
    return dydt

# initial condition
y0 = 0

# solve ODE
y = odeint(model,y0,t)

print('accumulation of charge at end of pulse: %.2f' % (y[-1],))

#%% Plot results
plt.figure(100); plt.clf()
plt.plot(1e3*t,pulse, ':c', label='Pulse')
plt.plot(1e3*t,y, '--g', label='Condensator charge')
plt.plot(1e3*t,pulse-y.flatten(), 'r', label='Voltage on sample')
plt.xlabel('time [ms]')
plt.ylabel('Voltages [V]')
plt.show()
plt.legend()
