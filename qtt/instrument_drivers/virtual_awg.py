# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:04:09 2016

@author: diepencjv
"""
#%%
''' Desired functionality
- Sweep a gate with a triangular signal
- Veryfast honeycombs
- Compensate gate sweep
'''

#%%
import numpy as np
from qcodes import Instrument
import scipy.signal
from qcodes import QtPlot, DataArray

#%%
class virtual_awg(Instrument):
    def __init__(self, name, instruments=[], awg_map=None, hardware=None, verbose=1, **kwargs):
#        shared_kwargs = ['instruments']
        super().__init__(name, **kwargs)
        self._awgs=instruments
        self.awg_map = awg_map
        self.hardware = hardware
        self.verbose = verbose
        
        if len(self._awgs)==0 and verbose:
            print('no physical AWGs connected')
        elif len(self._awgs) == 1:
            self.awg_cont = self._awgs[0]
            self.awg_cont.set('run_mode','CONT')
        elif len(self._awgs) == 2 and 'awg_mk' in self.awg_map.keys():
            self.awg_cont = self._awgs[self.awg_map['awg_mk'][0]]
            self.awg_cont.set('run_mode','CONT')
            self.awg_seq = self._awgs[(self.awg_map['awg_mk'][0]+1) % 2]
            self.awg_seq.set('run_mode','SEQ')
            self.awg_seq.set_sq_length(1)
        else:
            raise Exception('Configuration of AWGs not supported by virtual_awg instrument')

        self.AWG_clock = 1e7
        self.ch_amp = 4.0
        for awg in self._awgs:
            awg.set('clock_freq',self.AWG_clock)
            awg.delete_all_waveforms_from_list()
            for i in range(1,5):
                awg.set('ch%s_amp'% i, self.ch_amp)
        
    def get_idn(self):
        ''' Overrule because the default VISA command does not work '''
        IDN = {'vendor': 'QuTech', 'model': 'virtual_awg',
                    'serial': None, 'firmware': None}
        return IDN
        
    def stop(self,verbose=0):
        ''' Stops all AWGs and turns of all channels '''
        for awg in self._awgs:
            awg.stop()
            awg.set('ch1_state',0)
            awg.set('ch2_state',0)
            awg.set('ch3_state',0)
            awg.set('ch4_state',0)
        if verbose:
            print('Stopped AWGs')
        
    def sweep_init(self, waveforms):
        ''' Send waveform(s) to gate(s)
        
        Arguments:
            waveforms (dict): the waveforms with the gates as keys
        
        Returns:
            sweep_info (dict): the keys are tuples of the awgs and channels to activate
        
        Example:
        -------
        >>> sweep_info = sweep_init(waveforms) 
        '''
        sweepgates = [g for g in waveforms]
        
        for awg in self._awgs:
            awg.delete_all_waveforms_from_list()
            
        awgs = [self._awgs[self.awg_map[g][0]] for g in sweepgates]
        fpga_info = self.awg_map['fpga_mk']
        awgs.append(self._awgs[fpga_info[0]])
        
        sweep_info = dict()
        wave_zero = np.zeros(len(waveforms[sweepgates[0]]))
        for g in sweepgates:
            sweep_info[self.awg_map[g]] = dict()
            sweep_info[self.awg_map[g]]['waveform'] = waveforms[g]
            sweep_info[self.awg_map[g]]['marker1'] = wave_zero
            sweep_info[self.awg_map[g]]['marker2'] = wave_zero
            sweep_info[self.awg_map[g]]['name'] = 'waveform_%s' % g
        
        # fpga marker
        delay_FPGA = 25.0e-6 # should depend on filterboxes
        fpga_marker = np.zeros(len(wave_zero))
        fpga_marker[int(delay_FPGA*self.AWG_clock):(int(delay_FPGA*self.AWG_clock)+len(wave_zero)//20)]=1.0
        
        if fpga_info[:2] not in sweep_info:
            sweep_info[fpga_info[:2]] = dict()
            sweep_info[fpga_info[:2]]['waveform'] = wave_zero
            sweep_info[fpga_info[:2]]['marker1'] = wave_zero
            sweep_info[fpga_info[:2]]['marker2'] = wave_zero
            sweep_info[fpga_info[:2]]['name'] = 'fpga_mk'
            
        sweep_info[fpga_info[:2]]['marker%d' % fpga_info[2]] = fpga_marker
        self._awgs[fpga_info[0]].set('ch%i_m%i_low' % (fpga_info[1], fpga_info[2]),0)
        self._awgs[fpga_info[0]].set('ch%i_m%i_low' % (fpga_info[1], fpga_info[2]),2.6)
        
        # awg marker
        if self.awg_seq in awgs:
            awg_info = self.awg_map['awg_mk']
            if awg_info[:2] not in sweep_info:
                awgs.append(self._awgs[awg_info[0]])
                sweep_info[awg_info[:2]] = dict()
                sweep_info[awg_info[:2]]['waveform'] = wave_zero
                sweep_info[awg_info[:2]]['marker1'] = wave_zero
                sweep_info[awg_info[:2]]['marker2'] = wave_zero
                sweep_info[awg_info[:2]]['name'] = 'awg_mk'
            
            delay_AWG=22.0e-6
            awg_marker = np.zeros(len(wave_zero))
            awg_marker[0:len(wave_zero)//20]=1
            awg_marker=np.roll(awg_marker,len(wave_zero)-int(delay_AWG*self.AWG_clock))
            sweep_info[awg_info[:2]]['marker%d' % self.awg_map['awg_mk'][2]] = awg_marker
            self._awgs[awg_info[0]].set('ch%i_m%i_low' % (awg_info[1], awg_info[2]),0)
            self._awgs[awg_info[0]].set('ch%i_m%i_low' % (awg_info[1], awg_info[2]),2.6)
        
        # send waveforms
        for sweep in sweep_info:
            self._awgs[sweep[0]].send_waveform_to_list(sweep_info[sweep]['waveform'],sweep_info[sweep]['marker1'],sweep_info[sweep]['marker2'],sweep_info[sweep]['name'])
            if self._awgs[sweep[0]] == self.awg_seq:
                self._awgs[sweep[0]].set_sqel_waveform(sweep_info[sweep]['name'],sweep[1],1)
                self._awgs[sweep[0]].set_sqel_loopcnt_to_inf(1)
                self._awgs[sweep[0]].set_sqel_event_jump_target_index(sweep[1],1)
                self._awgs[sweep[0]].set_sqel_event_jump_type(1,'IND')
            else:
                self._awgs[sweep[0]].set('ch%i_waveform' % sweep[1], sweep_info[sweep]['name'])
        
        return sweep_info
        
    def sweep_run(self,sweep_info):
        ''' Activate AWG(s) and channel(s) for the sweep(s) '''
        for sweep in sweep_info:
            self._awgs[sweep[0]].set('ch%i_state' %sweep[1],1)
            
        awgnrs = set([sweep[0] for sweep in sweep_info])
        for nr in awgnrs:
            self._awgs[nr].run()
            
    def make_sawtooth(self,sweeprange,risetime,width=.95):
        '''Make a sawtooth with a decline width determined by width. Not yet scaled with 
        awg_to_plunger value.
        '''
        samplerate = 1./self.AWG_clock
        tt = np.arange(0,2*risetime+samplerate,samplerate)
        v_wave = float(sweeprange/((self.ch_amp/2.0)))
        wave=(v_wave/2)*scipy.signal.sawtooth(np.pi*tt/risetime, width=width)
        
        return wave

    def sweep_gate(self, gate, sweeprange, risetime, width=.95):
        ''' Send a triangular signal with the AWG to a gate to sweep. Also 
        send a marker to the FPGA.
        
        Arguments:
            gate (string): the name of the gate to sweep
            sweeprange (float): the range of voltages to sweep over
            risetime (float): the risetime of the triangular signal
        
        Returns:
            waveform (dictionary): The waveform being send with the AWG.
        
        Example:
        -------
        >>> wave = sweep_gate('P1',sweeprange=60,risetime=1e-3)
        '''
        waveform = dict()
        wave_raw = self.make_sawtooth(sweeprange,risetime,width)
        awg_to_plunger = self.hardware.parameters['awg_to_%s' % gate].get()
        wave = np.array([x/awg_to_plunger for x in wave_raw])
        waveform[gate]=wave
        sweep_info = self.sweep_init(waveform)
        self.sweep_run(sweep_info)
        
        return waveform
        
    def plot_wave(self,wave):
        ''' Plot the wave '''
        horz_var = np.arange(0,len(wave)/self.AWG_clock,1/self.AWG_clock)
        x = DataArray(name='time(s)', label= 'time (s)', preset_data=horz_var, is_setpoint=True)
        y = DataArray(label='sweep value (mV)',preset_data=wave, set_arrays=(x,))        
        plot = QtPlot(x,y)

        return plot