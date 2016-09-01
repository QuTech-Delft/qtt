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
import scipy

#%%
class virtual_awg(Instrument):
    def __init__(self, name, instruments, awg_map, hardware, verbose=1, **kwargs):
        shared_kwargs = ['instruments']
        super().__init__(name, **kwargs)
        self._awgs=instruments
        self.awg_map = awg_map
        self.hardware = hardware
        self.verbose = verbose
        
        if 'awg_mk' in self.awg_map.keys():
            awg_cont = self._awgs[self.awg_map['awg_mk'][0]]
            awg_cont.set('run_mode','CONT')
            for awg_seq in self._awgs:
                if awg_seq != awg_cont:
                    awg_seq.set('run_mode','SEQ')
                    awg_seq.set_sq_length(1)
        else:
            for awg in self._awgs:
                awg.set('run_mode','CONT')
            if self.verbose:
                print('All AWGs are set in continuous mode')

        self.AWG_clock = 1e7
        ch_amp = 4.0
        for awg in self._instruments:
            awg.set('clock_freq',self.AWG_clock)
            awg.delete_all_waveforms_from_list()
            for i in range(5):
                awg.set('ch%s_amp'% i, ch_amp)
        
    def get_idn(self):
        ''' Overrule because the default VISA command does not work '''
        IDN = {'vendor': 'QuTech', 'model': 'virtual_awg',
                    'serial': None, 'firmware': None}
        return IDN
        
    def stop(self,verbose=0):
        ''' Stops all AWGs and turns of all channels '''
        for awg in self._instruments:
            awg.stop()
            awg.set('ch1_state',0)
            awg.set('ch2_state',0)
            awg.set('ch3_state',0)
            awg.set('ch4_state',0)
        if verbose:
            print('Stopped AWGs')
            
    def sweep_gate(self, gate, sweeprange, risetime):
        '''Send a triangular signal with the AWG to a gate to sweep. Also 
        send a marker to the FPGA.
        
        Arguments:
            gate (string): the name of the gate to sweep
            sweeprange (float): the range of voltages to sweep over
            risetime (float): the risetime of the triangular signal
        
        Returns:
            wave (array): The wave being send with the AWG.
        
        Example:
        -------
        >>> wave = sweep_gate('P1',sweeprange=60,risetime=1e-3)
        '''
        awg=self.awg_map[gate][0]
        wave_ch=self.awg_map[gate][1]
        awg_fpga=self.awg_map['fpga_mk'][0]
        fpga_ch=self.awg_map['fpga_mk'][1]
        fpga_ch_mk=self.awg_map['fpga_mk'][2]
        
        tri_wave='tri_wave'
        awg_to_plunger = self.hardware.parameters['awg_to_%s' % gate].get()
        v_wave=float(sweeprange[0]/((awg.get('ch%d_amp' % wave_ch)/2.0)))
        v_wave=v_wave/awg_to_plunger
        samplerate = 1./self.AWG_clock
        tt = np.arange(0,2*risetime+samplerate,samplerate)
        wave=(v_wave/2)*scipy.signal.sawtooth(np.pi*tt/risetime, width=.5)
        awg.send_waveform_to_list(wave,np.zeros(len(wave)),np.zeros(len(wave)),tri_wave)
        
        delay_FPGA=25e-6
        marker2 = np.zeros(len(wave))
        marker2[int(delay_FPGA/samplerate):int((delay_FPGA+0.40*risetime)/samplerate)]=1.0
        fpga_marker='fpga_marker'
        awg.send_waveform_to_list(np.zeros(len(wave)),np.zeros(len(wave)),marker2,fpga_marker)
        
        awg.set('ch%i_waveform' %wave_ch,tri_wave)
        awg.set('ch%i_state' %wave_ch,1)
        awg_fpga.set('ch%i_waveform' %fpga_ch,fpga_marker)
        awg_fpga.set('ch%i_m%i_low' % (fpga_ch, fpga_ch_mk),0)
        awg_fpga.set('ch%i_m%i_high' % (fpga_ch, fpga_ch_mk),2.6)
        awg_fpga.set('ch%i_state' %fpga_ch,1)
        awg.run()
        awg_fpga.run()