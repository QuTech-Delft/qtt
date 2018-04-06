# -*- coding: utf-8 -*-
"""
Contains code to do live plotting 

"""
#%%
import numpy as np
import logging
from colorama import Fore
import qcodes
import qtt

#%%


class SimulationDigitizer(qcodes.Instrument):

    def __init__(self, name, model=None, **kwargs):
        """ Class to provide virtual digitizer for simulation model """
        super().__init__(name, **kwargs)
        self.current_sweep = None
        self.model = model
        self.add_parameter('sample_rate', set_cmd=None, initial_value=1e6)
        self.debug = {}
        self.verbose = 0

    def measuresegment(self, waveform, channels=[0]):
        import time
        if self.verbose:
            print('%s: measuresegment: channels %s' % (self.name, channels))
            print(waveform)
        self._waveform = waveform

        sd1, sd2 = self.myhoneycomb()
        time.sleep(.1)
        return [sd1.T, sd2.T][0:len(channels)]

    def myhoneycomb(self, multiprocess=False, verbose=0):
        """
        Args:
            model (object):
            Vmatrix (array): transformation from ordered scan gates to the sourcenames 
            erange, drange (float):
            nx, ny (integer):
        """
        test_dot = self.model.ds
        waveform = self._waveform
        model = self.model
        sweepgates = waveform['sweepgates']
        if isinstance(sweepgates, dict):
            sweepgates=[sweepgates]
            
        ndim = len(sweepgates)

        nn = waveform['resolution']
        if isinstance(nn, float):
            nn = [nn] * ndim
        nnr = nn[::-1]  # funny reverse ordering

        if verbose >= 2:
            print('myhoneycomb: start resolution %s' % (nn,))

        if ndim != len(nn):
            raise Exception(
                'number of sweep gates %d does not match resolution' % ndim)

        ng = len(model.gate_transform.sourcenames)
        test2Dparams = np.zeros((test_dot.ngates, *nnr))
        gate2Dparams = np.zeros((ng, *nnr))
        logging.info('honeycomb: %s' % (nn,))

        rr = waveform['sweepranges']
        wtype = waveform.get('type')

        v = model.gate_transform.sourcenames
        Vmatrix = np.eye(len(v))

        if wtype == 'sweep_2D_virt':
            iih = [v.index(s) for s in sweepgates[0]]
            iiv = [v.index(s) for s in sweepgates[1]]

            vh = list(sweepgates[0].values())
            vv = list(sweepgates[1].values())
            Vmatrix[0, :] = 0
            Vmatrix[1, :] = 0
            for idx, j in enumerate(iih):
                Vmatrix[0, j] = vh[idx]
            for idx, j in enumerate(iiv):
                Vmatrix[1, j] = vv[idx]


            inverseV = Vmatrix.T
            Vmatrix = None
        else:
            ii = [v.index(s) for s in sweepgates[0]]

            idx = np.array((range(len(v))))
            for i, j in enumerate(ii):
                idx[i], idx[j] = idx[j], idx[i]
            Vmatrix = Vmatrix[:, idx].copy()
            inverseV = np.linalg.inv(Vmatrix)
        sweeps = []
        for ii in range(ndim):
            sweeps.append(np.linspace(-rr[ii]/2, rr[ii]/2, nn[ii]))
        meshgrid = np.meshgrid(*sweeps)
        mm = tuple([xv.flatten() for xv in meshgrid])
        w = np.vstack((*mm, np.zeros((ng - ndim, mm[0].size))))
        ww = inverseV.dot(w)

        for ii, p in enumerate(model.gate_transform.sourcenames):
            val = model.get_gate(p)
            gate2Dparams[ii] = val

        for ii, p in enumerate(model.gate_transform.sourcenames):
            gate2Dparams[ii] += ww[ii].reshape(nnr)

        qq = model.gate_transform.transformGateScan(
            gate2Dparams.reshape((gate2Dparams.shape[0], -1)))
        # for debugging
        self.debug['gate2Dparams'] = gate2Dparams
        self.debug['qq'] = qq
        self.debug['inverseV'] = inverseV

        for ii in range(test_dot.ndots):
            test2Dparams[ii] = qq['det%d' % (ii + 1)].reshape(nnr)

        if ndim == 1:
            test2Dparams = test2Dparams.reshape(
                (test2Dparams.shape[0], test2Dparams.shape[1], 1))
        # run the honeycomb simulation
        test_dot.simulate_honeycomb(
            test2Dparams, multiprocess=multiprocess, verbose=0)

        sd1 = ((test_dot.hcgs) * (model.sddist1.reshape((1, 1, -1)))).sum(axis=-1)
        sd2 = ((test_dot.hcgs) * (model.sddist2.reshape((1, 1, -1)))).sum(axis=-1)
        sd1 *= (1 / np.sum(model.sddist1))
        sd2 *= (1 / np.sum(model.sddist2))

        if model.sdnoise > 0:
            sd1 += model.sdnoise * \
                (np.random.rand(*test_dot.honeycomb.shape) - .5)
            sd2 += model.sdnoise * \
                (np.random.rand(*test_dot.honeycomb.shape) - .5)
        if ndim == 1:
            sd1 = sd1.reshape((-1,))
            sd2 = sd2.reshape((-1,))
        #plt.figure(1000); plt.clf(); plt.plot(sd1, '.b'); plt.plot(sd2,'.r')
        self.debug['sd']=sd1, sd2
        return sd1, sd2


class SimulationAWG(qcodes.Instrument):

    def __init__(self, name, **kwargs):
        """ Class to provide an AWG object when using the simulation """
        super().__init__(name, **kwargs)
        self.add_parameter('sampling_frequency',
                           set_cmd=None, initial_value=1e6)

    def awg_gate(self, name):
        return False

    def sweep_gate(self, gate, sweeprange, period, width=.95, wave_name=None, delete=True):
        self.current_sweep = {'waveform': 'simulation_awg', 'gate': gate, 'sweeprange': sweeprange,
                              'type': 'sweep_gate', 'period': period, 'width': width, 'wave_name': wave_name}

        waveform = self.current_sweep
        waveform['resolution'] = [int(period * self.sampling_frequency())]
        waveform['sweepgates'] = [waveform['gate']]
        waveform['sweepranges'] = [waveform['sweeprange']]

        sweep_info = None
        self._waveform  = waveform
        return waveform, sweep_info

    def sweep_gate_virt(self, fast_sweep_gates, sweeprange, period, delete=None):
        self.current_sweep = {'waveform': 'simulation_awg', 'sweepgates':  [fast_sweep_gates], 'sweeprange': sweeprange,
                              'type': 'sweep_1D_virt', 'period': period, }
        waveform = self.current_sweep

        waveform['resolution'] = [int(period * self.sampling_frequency())]
        waveform['sweepranges'] = [waveform['sweeprange']]

        self._waveform  = waveform
        return waveform, None

    def sweep_2D_virt(self, samp_freq, gates_horz, gates_vert, sweepranges, resolution):
        self.current_sweep = {'waveform': 'simulation_awg', 'sweepgates':  [gates_horz, gates_vert], 'sweepranges': sweepranges,
                              'type': 'sweep_2D_virt', 'samp_freq': samp_freq, 'resolution': resolution}
        waveform = self.current_sweep
        self._waveform  = waveform
        return waveform, None

    def sweep_2D(self, samp_freq, sweepgates, sweepranges, resolution):
        self.current_sweep = {'waveform': 'simulation_awg', 'sweepgates': sweepgates, 'sweepranges': sweepranges,
                              'type': 'sweep_2D', 'samp_freq': samp_freq, 'resolution': resolution}
        waveform = self.current_sweep
        self._waveform  = waveform
        return waveform, None


    def stop(self):
        pass


