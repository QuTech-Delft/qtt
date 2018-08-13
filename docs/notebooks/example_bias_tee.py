# -*- coding: utf-8 -*-
""" Example to solve bias-T compensation 

For a bias-T the voltage set on the AWG does not completely match the voltage that arrives on the device gate. Besides the attenuation (which we assume to be linear), 
the capacitors in the bias-T will charge over time. In this example we show how to solve this exactly.


@author: eendebakpt
"""

#%% Load packages

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%% Define AWG pulse

samplerate = 1e6
T = 1e-3  # 1 ms pulse

# time points
t = np.arange(0, T, 1. / samplerate)

q = int(samplerate * 1e-4)
pulse = 0*t.copy()
pulse[0:q] = 50e-3 # typical pulse: about 50 mV
pulse[q:2 * q] = -20e-3

if 1:
    # mirror
    pulse = np.hstack((pulse, -pulse))
    t = np.arange(0, pulse.size / samplerate, 1. / samplerate)

#%% Model
#
# First order differential equation: V is voltage over capacitor, W is input voltage (e.g. voltage on AWG)
# The constant alpha depends on the RC time of the bias T.
#
# dV/dt = - alpha (W-V)

# function that returns dy/dt


def model(y, t, rc_time=1, verbose=1):
    alpha=1./rc_time
    tt = int(samplerate * t)
    tt = max(tt, 0)
    tt = min(tt, pulse.size - 1)
    W = pulse[tt]
    dydt = -alpha * (y - W)
    if verbose:
        print('t %.1f [micros], W %.3f, dydy %.3f' % (1e6 * t, W, dydt))
    return dydt


# initial condition
y0 = 0
y0 = .1e-3
# solve ODE
y = odeint(model, y0, t)

print('accumulation of charge at end of pulse: %.3f [mV]' % (1e3*y[-1],))
print('  diff %.4f [mV]' % (1e3*(y[-1]-y0), ) )

#%% Plot results
plt.figure(100)
plt.clf()
plt.plot(1e3 * t, pulse, ':c', label='Pulse')
plt.plot(1e3 * t, y, '--g', label='Condensator charge')
plt.plot(1e3 * t, pulse - y.flatten(), 'r', label='Voltage on sample')
plt.xlabel('time [ms]')
plt.ylabel('Voltages [V]')
plt.axhline(0, alpha=.5, color='m', linestyle=':')
plt.show()
plt.legend()


#%% Solve to steady state [work in progress]
for ii in range(40):
    y = odeint(lambda *x: model(*x, verbose=0), y0, t)
    print('y0: %.3f [mV] -> %.3f [mV]' % (1e3*y0, 1e3*y[-1]))
    y0=y[-1]


