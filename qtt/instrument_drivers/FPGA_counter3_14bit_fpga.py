# Driver for FPGA
#
# Pieter Eendebak <pieter.eendebak@tno.nl>
#
# Based on code written within the QuTech consortium

#%%
#import serial
import types
import logging
import numpy as np
import time
from numpy import arange,ceil
import sys

import pyvisa
import logging
import functools
import time
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.visa import visa

#%%
class FPGApulse(VisaInstrument):
    pass

    '''
    This is the python driver for the FPGA pulse counting module made by Marijn and Raymond

    Note: not all functionality ported
    Note: driver only for a single FPGA
    Usage:
    Initialize with
        <name> = instruments.create('name', 'FPGA_AVE', address='<COM PORT>')
        <COM PORT> = COM5 e.g.
    '''
    def __init__(self, name, address, verbose=1, **kwargs):
        self.verbose=verbose
        logging.debug(__name__ + ' : Initializing instrument')
        super().__init__(name, address, **kwargs)

    
        # TODO: functionality only for 1 
        
        #self.serialconnection1=serial.Serial(address[0], 57600, timeout=1)
        self.visa_handle.baud_rate=57600
        
        v=self.visa_handle
        v.set_visa_attribute(visa.constants.VI_ATTR_TERMCHAR_EN, 0)
        v.set_visa_attribute(visa.constants.VI_ATTR_ASRL_END_IN, 0)

        self.add_parameter('absolute_threshold_ch1',
                           get_cmd=self._partial(self.do_get_absolute_threshold, 1),
                            set_cmd=self._partial_set(self.do_set_absolute_threshold, 1),
                            units='mV ?')
        self.add_parameter('absolute_threshold_ch2',
                           get_cmd=self._partial(self.do_get_absolute_threshold, 2),
                            set_cmd=self._partial_set(self.do_set_absolute_threshold, 2),
                            units='mV ?')

    def _set_value(self, value, address, ch=1):
        pass
        # FIXME: implemente generic set

    def _get_value(self, value, address, ch=1):
        # FIXME: implement generic get
        if ch==1:
            self.ask_from_serial(address[0],value)
        elif ch==2:
            self.ask_from_serial(address[1],value)
        # implemente generic get/set

    def _partial_set(self, function, ch=1):
        def fun(value):
            return function(value, ch=1)
        return fun
        
    def _partial(self, function, ch=1):
        def fun():
            return function(ch=1)
        return fun
        
    def do_set_absolute_threshold(self,value,ch=1):
        '''
        Set hysteresis value for the pulse detection 
        '''
        if value <0:
            # value=value+32768
            value=value+65536
        
        if ch==1:
            self.write_to_serial(132,value)
        elif ch==2:
            self.write_to_serial(163,value)
            
    def do_get_absolute_threshold(self,ch=1):
        '''
        ????
        '''

        if ch==1:
            return self.convert_bit_to_mV(self.ask_from_serial(132)) 
        elif ch==2:
            return self.convert_bit_to_mV(self.ask_from_serial(163)) 
            
    def convert_bit_to_mV(self,bit):
        """ ???? """
        return bit*1
    def convert_mV_to_bit(self,mV):
        """ ???? """
        return mV/1

            
    def write_to_serial(self,address,value):
        register=(address)
        first_byte=(value>>8)
        last_byte=(value&255)
        self._write( bytes([register,first_byte,last_byte] ) )
        
    def _write(self, x):
        n, status = self.visa_handle.write_raw(bytes(x))
        return n
        
    def _read(self, size=None):
        '''
        Returns the values that are in the FPGA buffer. Replacement for 
        read_raw in messagebased.py, also see:
        https://github.com/hgrecco/pyvisa/issues/93
        https://github.com/hgrecco/pyvisa/issues/190
        '''
        size = self.visa_handle.chunk_size if size is None else size

        with self.visa_handle.ignore_warning(visa.constants.VI_SUCCESS_MAX_CNT):
            chunk, status = self.visa_handle.visalib.read(self.visa_handle.session, size)

        if not len(chunk)==size:
            raise Exception ('Visa received incorrect number of bytes')

        return chunk        
            
    def ask_from_serial(self,address):
        for iread in [1,2,3]:
            self._write([255, 0, address])
            readresult=self._read(3) #aantal bites = 3
            self.visa_handle.flush(pyvisa.constants.VI_READ_BUF)
            #self.serialconnection1.flushInput()
        
            if (readresult[0])==address:
                result=((readresult[1])<<8) | (readresult[2])
                return result
            else:
                print('Wrong address returned')
                time.sleep(0.1)
        
        
        print('Wrong address returned for three times') 
        return False
    
    def do_get_number_traces(self,ch=1):
        '''
        Set the number of traces over which pulse events will be counted
        '''
        if ch==1:
            return self.ask_from_serial(129)
        elif ch==2:
            return self.ask_from_serial(160)
            
    def do_get_clock_speed(self,ch=1):
        """
        Gets the sampling rate of the FPGA
        0 = 195 kHz
        1 = 390 kHz
        2= 781 kHz
        3 = 1562 kHz
        4 = 3.124 MHz
        5 = 6.248 MHz
        6 = 12.496 MHz
        7 = 24.992 MHz
        Note: you have to specify the clockspeed of the FPGA in the matlab script for making histograms (see variable timing_scale)
        Note2: during each gating period the FPGA can acquire at most 1024 datapoints, this limits you're maximum gating time!
        """
        if ch==1:
            return self.ask_from_serial(136)
        elif ch==2:
            return self.ask_from_serial(167)
        #elif ch==3:
        #    return self.ask_from_serial(136,fpga=2)
        #elif ch==4:
        #    return self.ask_from_serial(167,fpga=2)

    def do_set_clock_speed(self,value,ch=1):
            """
            Sets the sampling rate of the FPGA
            0 = 195 kHz
            1 = 390 kHz
            2= 781 kHz
            3 = 1.562 MHz
            4 = 3.124 MHz
            5 = 6.248 MHz
            6 = 12.496 MHz
            7 = 24.992 MHz
            Note: you have to specify the clockspeed of the FPGA in the matlab script for making histograms (see variable timing_scale)
            Note2: during each gating period the FPGA can acquire at most 1024 datapoints, this limits you're maximum gating time!
            """     
            if ch==1:
                self.write_to_serial(136,value)
            elif ch==2:
                self.write_to_serial(167,value)
            
if __name__=='__main__':
    # testing
    server_name=None
    fpga = FPGApulse('fpga_pulse', 'ASRL4::INSTR', server_name=server_name)
    self=fpga 
        
    # test
    fpga.do_set_absolute_threshold(60)
    print('absolute_threshold: ch 1: %f'  % fpga.do_get_absolute_threshold(ch=1))
    print('absolute_threshold: ch 1: %f'  % fpga.absolute_threshold_ch1.get())
    fpga.do_set_absolute_threshold(50)
    print('absolute_threshold: ch 1: %f'  % fpga.do_get_absolute_threshold(ch=1))
    print('absolute_threshold: ch 1: %f'  % fpga.absolute_threshold_ch1.get())
    
    #%%
Instrument = object # dummy for old qt style instrument
    
class FPGA_counter3_14bit_fpga(Instrument):
    
    
    
    '''
    This is the python driver for the FPGA pulse counting module made by Marijn and Raymond

    Usage:
    Initialize with
    <name> = instruments.create('name', 'FPGA_counter', address='<COM PORT>')
    <COM PORT> = COM5 e.g.
    '''

    def __init__(self, name, address):
        '''
        Initializes the FPGA counting module

        Input:
            name (string)    : name of the instrument
            address (string) : Virtual COM port

        Output:
            None
        '''
        logging.debug(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])

        self._address = address
        
        self._values = {}


            
        # Add parameters        
        self.add_parameter('hysteresis', type=int,
            flags=Instrument.FLAG_GETSET, units='', format='')
        self.add_parameter('number_traces', type=int,
            flags=Instrument.FLAG_GETSET, units='', format='')
        self.add_parameter('number_events', type=list,
            flags=Instrument.FLAG_GET, units='', format='')
        self.add_parameter('measure_done', type=int,
            flags=Instrument.FLAG_GET, units='', format='')
        self.add_parameter('threshold', type=int,
            flags=Instrument.FLAG_GETSET, units='mV', format='')
        self.add_parameter('absolute_threshold', type=int,
            flags=Instrument.FLAG_GETSET, units='mV', format='')
        self.add_parameter('clock_speed', type=int,
            flags=Instrument.FLAG_GETSET, units='', format='')
        self.add_parameter('PSB_mode', type=int,
            flags=Instrument.FLAG_GETSET, units='', format='')
        # self.add_parameter('peak_value_of_noise', type=types.IntType,
            # flags=Instrument.FLAG_GET, units='mV', format='')           
        self.add_parameter('amount_of_packets', type=int,
            flags=Instrument.FLAG_GET, units='', format='')            
        self.add_parameter('cycle_number', type=int,
            flags=Instrument.FLAG_GET, units='', format='')                
        self.add_parameter('timing_data', type=list,
            flags=Instrument.FLAG_GET, units='', format='')
        self.add_parameter('PSB_results', type=list,
            flags=Instrument.FLAG_GET, units='', format='')
        self.add_parameter('pulse_detection_data', type=list,
            flags=Instrument.FLAG_GET, units='', format='')                         
        self.add_parameter('pulse_detection_data_binary', type=list,
            flags=Instrument.FLAG_GET, units='', format='')                             
        self.add_parameter('pulse_detection_data_decimal', type=list,
            flags=Instrument.FLAG_GET, units='', format='')                         
        self.add_parameter('average', type=list,
            flags=Instrument.FLAG_GET, units='', format='')  
        self.add_parameter('Npoints_in_a_window', type=list,
            flags=Instrument.FLAG_GET, units='', format='')  
        self.add_parameter('average_check_points', type=list,
            flags=Instrument.FLAG_GET, units='', format='')  
        self.add_parameter('min', type=list,
            flags=Instrument.FLAG_GET, units='', format='')  
        self.add_parameter('max', type=list,
            flags=Instrument.FLAG_GET, units='', format='')     
        self.add_parameter('first_point_detection_mode', type=int,
            flags=Instrument.FLAG_GETSET, units='', format='')            
        
        self.add_function('wait_till_measurement_is_done')
        self.add_function('set_number_traces')
        
        self.add_function('close_serial_ports')
        
        self.serialconnection1=serial.Serial(address[0], 57600, timeout=1)
        
        if len(address)==2:
            self.serialconnection2=serial.Serial(address[1], 57600, timeout=1)
            
        
        self.add_parameter('invert', type=int,
            flags=Instrument.FLAG_GETSET, units='', format='')  
            
        Nchannels=2*len(address)
        
        self._number_events=[0]*Nchannels
        self._number_events_PSB=[0]*Nchannels
        self._average=[0]*Nchannels
        self._min=[0]*Nchannels
        self._max=[0]*Nchannels
        self._saved_timing_data=[[]]*Nchannels
        self._saved_PSB_results=[[]]*Nchannels
        self._timing_data_already_read=False
        self._PSB_results_already_read=False
        self._PSB_mode = False
    
    def convert_bit_to_mV(self,bit):
        return bit*1
    def convert_mV_to_bit(self,mV):
        return mV/1
    
    # def serial(self,here,fpga=1):
        # if fpga==1:
            # self.serialconnection1.write(here)
        # elif fpga==2:
            # self.serialconnection2.write(here)

    # def read(self,packet,fpga=1):
        # if fpga==1:
            # readresult=self.serialconnection1.read(packet)
        # elif fpga==2:
            # readresult=self.serialconnection2.read(packet)
        
        # print 'readresult...'
        # print readresult
        # return readresult
        
    def write_to_serial(self,address,value,fpga=1):
        register=chr(address)
        first_byte=chr(value>>8)
        last_byte=chr(value&255)
        if fpga==1:
            self.serialconnection1.write(register)
            self.serialconnection1.write(first_byte)
            self.serialconnection1.write(last_byte)
        elif fpga==2:
            self.serialconnection2.write(register)
            self.serialconnection2.write(first_byte)
            self.serialconnection2.write(last_byte)
        
    def ask_from_serial(self,address,fpga=1):
        for iread in [1,2,3]:
            if fpga==1:
                self.serialconnection1.write(chr(255))
                self.serialconnection1.write(chr(0))
                self.serialconnection1.write(chr(address))
                readresult=self.serialconnection1.read(3) #aantal bites = 3
                self.serialconnection1.flushInput()
            elif fpga==2:
                self.serialconnection2.write(chr(255))
                self.serialconnection2.write(chr(0))
                self.serialconnection2.write(chr(address))
                readresult=self.serialconnection2.read(3)
                self.serialconnection2.flushInput()
        
            if ord(readresult[0])==address:
                result=(ord(readresult[1])<<8) | ord(readresult[2])
                return result
            else:
                print('Wrong address returned')
                qt.msleep(0.1)
        
        
        print('Wrong address returned for three times') 
        return False
    
    def ask_from_serial_big_data(self,address,ch=[1]):
        packet_volumne=[0]*len(self._address)
        fpga_lst=[]

        for chname in ch:
            if chname<3:
                fpga_lst+=[1]
                packet_volumne[0]=self.get_amount_of_packets(ch=chname)
            elif chname<5:
                fpga_lst+=[2]
                packet_volumne[1]=self.get_amount_of_packets(ch=chname)
            
        if 1 in fpga_lst:
            self.serialconnection1.write(chr(255))
            self.serialconnection1.write(chr(0))
            self.serialconnection1.write(chr(address))
        if 2 in fpga_lst:
            self.serialconnection2.write(chr(255))
            self.serialconnection2.write(chr(0))
            self.serialconnection2.write(chr(address))
        
        if self._PSB_mode:
            readresult1=[]
            readresult2=[]
            if 1 in ch:
                readresult1=self.serialconnection1.read(3*packet_volumne[0])
                self.serialconnection1.flushInput()
            if 2 in ch:
                readresult2=self.serialconnection1.read(3*packet_volumne[0])
                self.serialconnection1.flushInput()
                
            # print 'readresult1',[readresult1]
            # print 'readresult2',[readresult2]
            return [readresult1,readresult2]
            
        else:
            readresult1=[]
            readresult2=[]
            readresult3=[]
            readresult4=[]
            if 1 in ch:
                readresult1=self.serialconnection1.read(3*packet_volumne[0])
                self.serialconnection1.flushInput()
            if 2 in ch:
                readresult2=self.serialconnection1.read(3*packet_volumne[0])
                self.serialconnection1.flushInput()
            if 3 in ch:
                readresult3=self.serialconnection2.read(3*packet_volumne[1])
                self.serialconnection2.flushInput()
            if 4 in ch:
                readresult4=self.serialconnection2.read(3*packet_volumne[1])
                self.serialconnection2.flushInput()
                
            return [readresult1,readresult2,readresult3,readresult4]
            
            
    def ask_from_serial_signed(self,address,Nbit=16,fpga=1):
        # print 'Ben in functie SERIAL'
        if type(address)==type(1):
            vref=2**Nbit
            for iread in [1,2,3]:
                if fpga==1:
                    self.serialconnection1.write(chr(255))
                    self.serialconnection1.write(chr(0))
                    self.serialconnection1.write(chr(address))
                    readresult=self.serialconnection1.read(3)
                    self.serialconnection1.flushInput()
                elif fpga==2:
                    self.serialconnection2.write(chr(255))
                    self.serialconnection2.write(chr(0))
                    self.serialconnection2.write(chr(address))
                    readresult=self.serialconnection2.read(3)
                    self.serialconnection2.flushInput()
                
                if ord(readresult[0])==address:
                    unsigned=(ord(readresult[1])<<8) | ord(readresult[2])
                    signed=unsigned-vref if unsigned > (vref/2-1) else unsigned
                    return signed
                else:
                    print('Wrong address returned, read after 100ms...')
                    qt.msleep(0.1)
            print('Wrong address returned for three times')        
            return False

        elif type(address)==type([1]):
            # print 'Ben in 2e if SERIAL'
            # print 'hallo'
            vref=2**(Nbit*len(address))
            for addcnt,addname in enumerate(address):
                # print 'addname:',addname
                for iread in [1]:
                    if fpga==1:
                        # print 'Ben in fpga 1 SERIAL'
                        self.serialconnection1.write(chr(255))
                        self.serialconnection1.write(chr(0))
                        self.serialconnection1.write(chr(addname))
                        readresult=self.serialconnection1.read(3)
                        self.serialconnection1.flushInput()
                        # print 'readresult, ', readresult
                    elif fpga==2:
                        self.serialconnection2.write(chr(255))
                        self.serialconnection2.write(chr(0))
                        self.serialconnection2.write(chr(addname))
                        readresult=self.serialconnection2.read(3)                    
                        self.serialconnection2.flushInput()
                        
                    if ord(readresult[0])==addname:
                        if addcnt==0:
                            unsigned_12=(ord(readresult[1])<<8) | ord(readresult[2])
                            # print 'addname:',addname
                            # print 'iread:',iread
                            # print 'readresult[1]',readresult[1],'-------',ord(readresult[1])
                            # print 'readresult[2]',readresult[2],'-------',ord(readresult[2])
                            # print 'unsigned_12 --->',unsigned_12
                        elif not addcnt==(len(address)-1):
                            unsigned_new=(ord(readresult[1])<<8) | ord(readresult[2])
                            unsigned_tmp=unsigned
                            unsigned=(unsigned<<16)|unsigned_tmp
                            # print 'unsigned_new',unsigned_new
                            # print 'unsigned',unsigned
                        elif addcnt==(len(address)-1) :                     
                            unsigned_34=(ord(readresult[1])<<8) | ord(readresult[2])
                            # unsigned_tmp=unsigned
                            unsigned=(unsigned_12<<16)|unsigned_34
                            # print 'addname: (elif)',addname
                            # print 'readresult[1] ',readresult[1],'-------',ord(readresult[1])
                            # print 'readresult[2] ',readresult[2],'-------',ord(readresult[2])
                            # print 'unsigned_34 ---> ',unsigned_34
                            # print 'unsigned... ',unsigned
                            signed=unsigned-vref if unsigned > (vref/2-1) else unsigned
                            # print 'signed... ',signed
                            
                            
                            # print '-------------------- Natasja gaat iets proberen -----------------------------'
                            # tas_new= (15<<4|15)
                            # print 'tas_new, --> ', tas_new
                            # vrefn = 2**8
                            # signed_tas =tas_new-vrefn if tas_new > (vrefn/2-1) else tas_new
                            # print 'signed_tas --> ', signed_tas
                            # print '--------------------EINDE! -----------------------------'
                            
                            return signed
                    else:
                        print('Wrong address returned, read after 100ms...')
                        qt.msleep(0.1)
            print('Wrong address returned for three times')        
            return False
    def do_set_number_traces(self,value,ch=[1]):
        '''
        Set the number of traces over which pulse events will be counted
        ch=1 or ch=2 sets the number of traces for both channel 1 and channel 2 of fpga box 1
        ch=3 or ch=4 sets the number of traces for both channel 1 and channel 2 of fpga box 2
        In this way we can sync both boxes.
        '''
        
        if type(ch)==type(1):
            ch=[ch]
        
        if (1 in ch) or (2 in ch):
            self.write_to_serial(129,value,fpga=1)
            self.write_to_serial(160,value,fpga=1)
        if (3 in ch) or (4 in ch):
            self.write_to_serial(129,value,fpga=2)
            self.write_to_serial(160,value,fpga=2)
        
        
        
        self._timing_data_already_read=False
        self._PSB_results_already_read=False
        # print '_PSB_results_already_read after set_number_traces=',self._PSB_results_already_read
        # if (1 in ch) or (2 in ch):
            # self.write_to_serial(129,value,fpga=1)
        # if 2 in ch:
            # self.write_to_serial(160,value,fpga=1)
        # if 3 in ch:
            # self.write_to_serial(129,value,fpga=2)
        # if 4 in ch:
            # self.write_to_serial(160,value,fpga=2)
    
    def do_set_clock_speed(self,value,ch=1):
        '''
        Sets the sampling rate of the FPGA
        0 = 195 kHz
        1 = 390 kHz
        2= 781 kHz
        3 = 1.562 MHz
        4 = 3.124 MHz
        5 = 6.248 MHz
        6 = 12.496 MHz
        7 = 24.992 MHz
        Note: you have to specify the clockspeed of the FPGA in the matlab script for making histograms (see variable timing_scale)
        Note2: during each gating period the FPGA can acquire at most 1024 datapoints, this limits you're maximum gating time!
        '''
        
        if ch==1:
            self.write_to_serial(136,value,fpga=1)
        elif ch==2:
            self.write_to_serial(167,value,fpga=1)
        elif ch==3:
            self.write_to_serial(136,value,fpga=2)
        elif ch==4:
            self.write_to_serial(167,value,fpga=2)    
    
    def do_get_clock_speed(self,ch=1):
        ''' 
        Gets the sampling rate of the FPGA
        0 = 195 kHz
        1 = 390 kHz
        2= 781 kHz
        3 = 1562 kHz
        4 = 3.124 MHz
        5 = 6.248 MHz
        6 = 12.496 MHz
        7 = 24.992 MHz
        Note: you have to specify the clockspeed of the FPGA in the matlab script for making histograms (see variable timing_scale)
        Note2: during each gating period the FPGA can acquire at most 1024 datapoints, this limits you're maximum gating time!
        '''
        if ch==1:
            return self.ask_from_serial(136,fpga=1)
        elif ch==2:
            return self.ask_from_serial(167,fpga=1)
        elif ch==3:
            return self.ask_from_serial(136,fpga=2)
        elif ch==4:
            return self.ask_from_serial(167,fpga=2)
    
    def do_set_hysteresis(self,value,ch=1):
        '''
        Set hysteresis value for the pulse detection 
        '''
        if ch==1:
            self.write_to_serial(130,value,fpga=1)
        elif ch==2:
            self.write_to_serial(161,value,fpga=1)
        elif ch==3:
            self.write_to_serial(130,value,fpga=2)
        elif ch==4:
            self.write_to_serial(161,value,fpga=2)    
            
    def do_get_number_traces(self,ch=1):
        '''
        Set the number of traces over which pulse events will be counted
        '''
        if ch==1:
            return self.ask_from_serial(129,fpga=1)
        elif ch==2:
            return self.ask_from_serial(160,fpga=1)
        elif ch==3:
            return self.ask_from_serial(129,fpga=2)
        elif ch==4:
            return self.ask_from_serial(160,fpga=2)
        
    def do_get_hysteresis(self,ch=1):
        '''
        Get hysteresis value for the pulse detection
        '''
        if ch==1:
            return self.convert_bit_to_mV(self.ask_from_serial(130,fpga=1))
        elif ch==2:
            return self.convert_bit_to_mV(self.ask_from_serial(161,fpga=1))
        elif ch==3:
            return self.convert_bit_to_mV(self.ask_from_serial(130,fpga=2))
        elif ch==4:
            return self.convert_bit_to_mV(self.ask_from_serial(161,fpga=2))
        
    def do_get_number_events(self,ch=[1]):
        '''
        Get the number of traces per number_traces which contained an event
        Caution: there's no loop to retrieve data from both channels, need to fix this somewhere
        '''
        aevent=self._number_events[0]
        bevent=self._number_events[1]
        
        # cevent=self._number_events[2]
        # devent=self._number_events[3]
    
        
        if type(ch)==type(1):
            ch=[ch]

        # print 'entered FPGA.get_number_events'
        # print 'ch',ch
        # print 'PSB_mode = ',self._PSB_mode
        if self._PSB_mode:
            # print 'entered PSB_mode of FPGA.get_number_events'
            PSB_results_list=self.get_PSB_results(ch=ch)
            # print 'PSB_results_list',PSB_results_list
            if 1 in ch:
                aevent=sum(PSB_results_list[0])
            if 2 in ch:
                # bevent=sum(PSB_results_list[1])
                bevent=sum(PSB_results_list[0]) # 20160627 T.F. get_PSB_results gets only the single specified channel
        else:
            if 1 in ch:
                aevent=self.ask_from_serial(1,fpga=1)
            if 2 in ch:
                bevent=self.ask_from_serial(16,fpga=1)
            # if 3 in ch:
                # cevent=self.ask_from_serial(1,fpga=2)
        # if 1 in ch:
            # aevent=self.ask_from_serial(1,fpga=1)
        # if 2 in ch:
            # bevent=self.ask_from_serial(16,fpga=1)
        # if 3 in ch:
            # cevent=self.ask_from_serial(1,fpga=2)
        # if 4 in ch:
            # devent=self.ask_from_serial(16,fpga=2)    
        
        
        # self._number_events=[aevent,bevent,cevent,devent]
        # return [aevent,bevent,cevent,devent]
        
        self._number_events=[aevent,bevent]
        return [aevent,bevent]
    
    def do_get_measure_done(self,ch=1):
        '''
        Returns one if the measurement is done, else returns zero
        '''
        if self._PSB_mode:
            if ch == 1:
                return self.ask_from_serial(15,fpga=1)
            elif ch == 2:
                return self.ask_from_serial(30,fpga=1)
        else:
            if ch==1:
                return self.ask_from_serial(2,fpga=1)
            elif ch==2:
                return self.ask_from_serial(17,fpga=1)
            if ch==3:
                return self.ask_from_serial(2,fpga=2)
            elif ch==4:
                return self.ask_from_serial(17,fpga=2)
    
    def wait_till_measurement_is_done(self,ch=[1],printing=True,waitingTime=-1):
        '''
        Returns one if the measurement is done, else returns zero
        '''
        if type(ch)==type(1):
            ch=[ch]
        
        done=True
    
        if 1 in ch:
            done=done and (self.ask_from_serial(2,fpga=1)==1) 
        if 2 in ch:
            done=done and (self.ask_from_serial(17,fpga=1)==1)
        if 3 in ch:
            done=done and (self.ask_from_serial(2,fpga=2)==1) 
        if 4 in ch:
            done=done and (self.ask_from_serial(17,fpga=2)==1)
    
         ## this while loop stays active untill acquisition is finished
        printDone=0
        strtmp='0%[..........]100%'
        while not done:
            if printDone==0 and printing:
                if waitingTime>=1:
                    print('Acquiring from ch%s of FPGA %s'%(ch,strtmp))
                print('Acquiring from ch%s of FPGA   ['%ch, end=' ')
                sys.stdout.softspace=False
                printDone=1
                # print ' Done!'
            elif printing:
                print('.', end=' ')
                sys.stdout.softspace=False
            
            done=True
            if 1 in ch:
                done=done and (self.ask_from_serial(2,fpga=1)==1)
            if 2 in ch:
                done=done and (self.ask_from_serial(17,fpga=1)==1)
            if 3 in ch:
                done=done and (self.ask_from_serial(2,fpga=2)==1)
            if 4 in ch:
                done=done and (self.ask_from_serial(17,fpga=2)==1)    
                
            if done and printing:
                print('] Done!')
            if waitingTime==-1:
                qt.msleep(0.1)
            elif waitingTime<1:
                qt.msleep(waitingTime)
            elif waitingTime>=1:    
                qt.msleep(waitingTime/10.0)
            
            
        return True

    def do_get_amount_of_packets(self,ch=1):
        '''

        '''
        if ch==1:
            out=self.ask_from_serial(4,fpga=1)
        elif ch==2:
            out=self.ask_from_serial(19,fpga=1)
        elif ch==3:
            out=self.ask_from_serial(4,fpga=2)
        elif ch==4:
            out=self.ask_from_serial(19,fpga=2)
        
        return out

    def do_get_timing_data(self,ch=[1],outputAllchannels=False,printing=False):
        '''
            input : ch  :fpga channel, default value=[1]
            returns [wincount,flankcount,timecount] for each channel
        '''
        if type(ch)==type(1):
            ch=[ch]
        if self._timing_data_already_read:
            if outputAllchannels or len(ch)==4:
                if printing:
                    print('timing data is already read, so I return to you the previous saved value')
                return self._saved_timing_data
            else:
                if printing:
                    print('timing data is already read, so I return to you [selceted channels of] the previous saved value ')
                output=self._saved_timing_data
                finaloutput=[output[jcnt-1] for jcnt in ch]
                return finaloutput
        else:
            self._timing_data_already_read=True
            
        ascii1=''
        ascii2=''
        # ascii3=''
        # ascii4=''
        
        if 1 in ch:
            ascii1=self.ask_from_serial_big_data(5,ch=[1])[0]
        if 2 in ch:
            ascii2=self.ask_from_serial_big_data(20,ch=[2])[1]
        # if 3 in ch:
            # ascii3=self.ask_from_serial_big_data(5,ch=[3])[2]
        # if 4 in ch:
            # ascii4=self.ask_from_serial_big_data(20,ch=[4])[3]
        
        
        list1=[]
        wincount=[]
        flankcount=[]
        timecount=[]
        # we get 3 bytes, the first 10 bits (from MSB side) are the the window counter, then 3 bit flank counter, then one bit not used, the rest 10 bits are for the timing
        
        wincountMask=2**10-1
        flankcountMask=2**3-1
        timecountMask=2**10-1
        
        for icnt in arange(0,len(ascii1),3):
            temp=(ord(ascii1[icnt])<<16)|(ord(ascii1[icnt+1])<<8) | ord(ascii1[icnt+2])
            wincount.append(temp>>14 & wincountMask)
            flankcount.append(temp>>11 & flankcountMask)
            timecount.append(temp & timecountMask)
        
        list1=[wincount,flankcount,timecount]
        
        list2=[]
        wincount=[]
        flankcount=[]
        timecount=[]
        for icnt in arange(0,len(ascii2),3):
            temp=(ord(ascii2[icnt])<<16)|(ord(ascii2[icnt+1])<<8) | ord(ascii2[icnt+2])
            wincount.append(temp>>14 & wincountMask)
            flankcount.append(temp>>11 & flankcountMask)
            timecount.append(temp & timecountMask)
        
        list2=[wincount,flankcount,timecount]
        
        
        # list3=[]
        # wincount=[]
        # flankcount=[]
        # timecount=[]
        # for icnt in arange(0,len(ascii3),3):
            # temp=(ord(ascii3[icnt])<<16)|(ord(ascii3[icnt+1])<<8) | ord(ascii3[icnt+2])
            # wincount.append(temp>>14 & wincountMask)
            # flankcount.append(temp>>11 & flankcountMask)
            # timecount.append(temp & timecountMask)
        
        # list3=[wincount,flankcount,timecount]
        
        
        
        # list4=[]
        # wincount=[]
        # flankcount=[]
        # timecount=[]
        # for icnt in arange(0,len(ascii4),3):
            # temp=(ord(ascii4[icnt])<<16)|(ord(ascii4[icnt+1])<<8) | ord(ascii4[icnt+2])
            # wincount.append(temp>>14 & wincountMask)
            # flankcount.append(temp>>11 & flankcountMask)
            # timecount.append(temp & timecountMask)
        
        # list4=[wincount,flankcount,timecount]
        
        # self._saved_timing_data=[list1,list2,list3,list4]
        self._saved_timing_data=[list1,list2]
        
        if outputAllchannels:
            # return [list1,list2,list3,list4]
            return [list1,list2]
        else:
            # output=[list1,list2,list3,list4]
            output=[list1,list2]
            finaloutput=[output[jcnt-1] for jcnt in ch]
            return finaloutput
    def do_get_PSB_results(self,ch=[1],outputAllchannels=False,printing=False):
        '''
            input : ch  :fpga channel, default value=[1]
            returns binary result as a list for each channel
        '''
        # print 'entered do_get_PSB_results'
        if type(ch)==type(1):
            ch=[ch]
        # print 'self._PSB_results_already_read = ',self._PSB_results_already_read
        if self._PSB_results_already_read:
            if outputAllchannels or len(ch)==4:
                if printing:
                    print('PSB results is already read, so I return to you the previous saved value')
                return self._saved_PSB_results
            else:
                if printing:
                    print('PSB results is already read, so I return to you [selceted channels of] the previous saved value ')
                output=self._saved_PSB_results
                finaloutput=[output[jcnt-1] for jcnt in ch]
                return finaloutput
        else:
            self._PSB_results_already_read=True
            
        ascii1=''
        ascii2=''
        print('ch= ',ch)
        if 1 in ch:
            ascii1=self.ask_from_serial_big_data(14,ch=[1])[0]
        if 2 in ch:
            ascii2=self.ask_from_serial_big_data(29,ch=[2])[1]
        # print 'ascii1',ascii1
        # print 'ascii2',ascii2
        # make a list of binaries corresponding to PSB results
        list1=[]
        for i in arange(0,len(ascii1)): 
            bin1=format(ord(ascii1[i]),'008b')
            for j in arange(0,8):
                list1.append(int(bin1[j]))
        list1=list1[0:self.get_number_traces(1)]
        list2=[]
        for i in arange(0,len(ascii2)):
            bin2=format(ord(ascii2[i]),'008b')
            for j in arange(0,8):
                list2.append(int(bin2[j]))
        list2=list2[0:self.get_number_traces(2)]
        
        self._saved_PSB_results=[list1,list2]
        if outputAllchannels:
            # print 'outputAllchannels'
            return [list1,list2]
        else:
            # print 'not outputAllchannels'
            output=[list1,list2]
            finaloutput=[output[jcnt-1] for jcnt in ch]
            return finaloutput
    
    
    def do_get_pulse_detection_data(self,ch=[1]):
        '''
            outputs the pulse detection data (0 and 1 for every cycle where 0 means no tunnelling event, and 1 means one or more tunnelling events)
        '''
        # [ch1o,ch2o,ch3o,ch4o]=self.get_timing_data(ch=ch,outputAllchannels=True)
        # output=[[]]*4
        # for ich in ch:
            
            # current_ch_output=[]
            # if ich==1:
                # win_no_ch=ch1o[0]
                # if win_no_ch[0]==1:
                    # win_no_ch[0]=0
                # time_ch=ch1o[2]
            # elif ich==2:
                # win_no_ch=ch2o[0]
                # if win_no_ch[0]==1:
                    # win_no_ch[0]=0
                # time_ch=ch2o[2]
            # elif ich==3:
                # win_no_ch=ch3o[0]
                # if win_no_ch[0]==1:
                    # win_no_ch[0]=0
                # time_ch=ch3o[2]
            # elif ich==4:
                # win_no_ch=ch4o[0]
                # if win_no_ch[0]==1:
                    # win_no_ch[0]=0
                # time_ch=ch4o[2]
        
        [ch1o,ch2o]=self.get_timing_data(ch=ch,outputAllchannels=True)
        output=[[]]*2
        for ich in ch:
            
            current_ch_output=[]
            if ich==1:
                win_no_ch=ch1o[0]
                if win_no_ch[0]==1:
                    win_no_ch[0]=0
                time_ch=ch1o[2]
            elif ich==2:
                win_no_ch=ch2o[0]
                if win_no_ch[0]==1:
                    win_no_ch[0]=0
                time_ch=ch2o[2]

                
            timecnt=0
            for icnt in range(self.get_number_traces(ch=ich)):
                if win_no_ch.count(icnt)>1 or (win_no_ch.count(icnt)==1 and time_ch[timecnt]<1023): #-->1023 means no flank detected in the pulse
                    current_ch_output.append(ceil(win_no_ch.count(icnt)/2.0))
                elif win_no_ch.count(icnt)==1 and time_ch[timecnt]>=1023:
                    current_ch_output.append(0)
                else:
                    if win_no_ch.count(icnt)==0:
                        print('very strange> there is no cycle %s' %icnt)
                    print(icnt)
                    print('win_no_ch.count(icnt):',win_no_ch.count(icnt))
                    print('time_ch[icnt]',time_ch[timecnt])
                    print('strange that it entered to here')
                    current_ch_output.append(0)
                
                timecnt=timecnt+win_no_ch.count(icnt)
            output[ich-1]=current_ch_output
        finaloutput=[output[jcnt-1] for jcnt in ch]
        
        return finaloutput
    def do_get_pulse_detection_data_binary(self,ch=[1]):
        '''
            outputs the pulse detection data (0 and 1 for every cycle where 0 means no tunnelling event, and 1 means one or more tunnelling events)
        '''
        # [ch1o,ch2o,ch3o,ch4o]=self.get_timing_data(ch=ch,outputAllchannels=True)
        # output=[[]]*4
        # for ich in ch:
            
            # current_ch_output=[]
            # if ich==1:
                # win_no_ch=ch1o[0]
                # if win_no_ch[0]==1:
                    # win_no_ch[0]=0
                # time_ch=ch1o[2]
            # elif ich==2:
                # win_no_ch=ch2o[0]
                # if win_no_ch[0]==1:
                    # win_no_ch[0]=0
                # time_ch=ch2o[2]
            # elif ich==3:
                # win_no_ch=ch3o[0]
                # if win_no_ch[0]==1:
                    # win_no_ch[0]=0
                # time_ch=ch3o[2]
            # elif ich==4:
                # win_no_ch=ch4o[0]
                # if win_no_ch[0]==1:
                    # win_no_ch[0]=0
                # time_ch=ch4o[2]
                
        [ch1o,ch2o]=self.get_timing_data(ch=ch,outputAllchannels=True)
        output=[[]]*2
        for ich in ch:
            
            current_ch_output=[]
            if ich==1:
                win_no_ch=ch1o[0]
                if win_no_ch[0]==1:
                    win_no_ch[0]=0
                time_ch=ch1o[2]
            elif ich==2:
                win_no_ch=ch2o[0]
                if win_no_ch[0]==1:
                    win_no_ch[0]=0
                time_ch=ch2o[2]
                
            timecnt=0
            for icnt in range(self.get_number_traces(ch=ich)):
                if win_no_ch.count(icnt)>1 or (win_no_ch.count(icnt)==1 and time_ch[timecnt]<1023): #-->1023 means no flank detected in the pulse
                    current_ch_output.append(1)
                elif win_no_ch.count(icnt)==1 and time_ch[timecnt]>=1023:
                    current_ch_output.append(0)
                else:
                    if win_no_ch.count(icnt)==0:
                        print('very strange> there is no cycle %s' %icnt)
                    print(icnt)
                    print('win_no_ch.count(icnt):',win_no_ch.count(icnt))
                    print('time_ch[icnt]',time_ch[timecnt])
                    print('strange that it entered to here')
                    current_ch_output.append(0)
                
                timecnt=timecnt+win_no_ch.count(icnt)
            output[ich-1]=current_ch_output
        finaloutput=[output[jcnt-1] for jcnt in ch]
        
        return finaloutput    
        
    def do_get_pulse_detection_data_decimal(self,ch=[1]):
        '''
            similar to pulse detection, but it convert the pulse detection output to decimal.
            In this case the first channel input has the most significant bit last channel the least sig. bit.
            for example ch=[1,2] if for cycle number N an event happen in channel 1 and no event happen in channel 2 then the output for that cycle is 2.
        '''
        data_binary=self.get_pulse_detection_data_binary(ch=ch)
        output=[]
        Nelments=len(data_binary[0])
        Nchannel=len(data_binary)
        for icnt in range(Nelments):
            calc_dec=0
            for jcnt in range(Nchannel):
                p=Nchannel-jcnt-1
                calc_dec+=(2**p)*data_binary[jcnt][icnt]
            output.append(calc_dec)
        return output
    def do_get_threshold(self,ch=1):
        '''
        Returns one if the measurement is done, else returns zero
        '''
        if self._PSB_mode:
            if ch == 1:
                return self.convert_bit_to_mV(self.ask_from_serial(132,fpga=1)) 
            elif ch == 2:
                return self.convert_bit_to_mV(self.ask_from_serial(163,fpga=1)) 
        else:
            if ch==1:
                return self.convert_bit_to_mV(self.ask_from_serial(131,fpga=1)) 
            elif ch==2:
                return self.convert_bit_to_mV(self.ask_from_serial(162,fpga=1)) 
            elif ch==3:
                return self.convert_bit_to_mV(self.ask_from_serial(131,fpga=2)) 
            elif ch==4:
                return self.convert_bit_to_mV(self.ask_from_serial(162,fpga=2)) 
            
    def do_set_threshold(self,value,ch=1):
        '''
        Set hysteresis value for the pulse detection 
        '''
        if value <0:
            value=value+65536
        if self._PSB_mode:
            if ch == 1:
                self.write_to_serial(132,value,fpga=1)
            elif ch == 2:
                self.write_to_serial(163,value,fpga=2)
        else:    
            if ch==1:
                self.write_to_serial(131,value,fpga=1)
            elif ch==2:
                self.write_to_serial(162,value,fpga=1)
            elif ch==3:
                self.write_to_serial(131,value,fpga=2)
            elif ch==4:
                self.write_to_serial(162,value,fpga=2)
            
    def do_get_absolute_threshold(self,ch=1):
        '''
        Returns one if the measurement is done, else returns zero
        '''

        if ch==1:
            return self.convert_bit_to_mV(self.ask_from_serial(132,fpga=1)) 
        elif ch==2:
            return self.convert_bit_to_mV(self.ask_from_serial(163,fpga=1)) 
        elif ch==3:
            return self.convert_bit_to_mV(self.ask_from_serial(132,fpga=2)) 
        elif ch==4:
            return self.convert_bit_to_mV(self.ask_from_serial(163,fpga=2)) 
            
    def do_set_absolute_threshold(self,value,ch=1):
        '''
        Set hysteresis value for the pulse detection 
        '''
        if value <0:
            # value=value+32768
            value=value+65536
        
        if ch==1:
            self.write_to_serial(132,value,fpga=1)
        elif ch==2:
            self.write_to_serial(163,value,fpga=1)
        elif ch==3:
            self.write_to_serial(132,value,fpga=2)
        elif ch==4:
            self.write_to_serial(163,value,fpga=2)
        
    # def do_get_peak_value_of_noise(self,ch=1):
        # '''
        # Returns one if the measurement is done, else returns zero
        # Note: we found out on 27-11-2014 that this functionality has already been disabled
        # '''
        # if ch==1:
            # return self.convert_bit_to_mV(self.ask_from_serial_signed(3,fpga=1))      
        # elif ch==2:
            # return self.convert_bit_to_mV(self.ask_from_serial_signed(18,fpga=1))      
        # elif ch==3:
            # return self.convert_bit_to_mV(self.ask_from_serial_signed(3,fpga=2))      
        # elif ch==4:
            return self.convert_bit_to_mV(self.ask_from_serial_signed(18,fpga=2))          
        
    def do_get_average(self,ch=[1],Nbit=16):
        '''
        Returns the average of the signal over set number of cycles
        '''
        # print 'AVERAGE!'
        if type(ch)==type(1):
            ch=[ch]
        
        ch1_av=0
        ch2_av=0
        
        ch3_av=0
        ch4_av=0
        
        if 1 in ch:
            sum=self.ask_from_serial_signed([7,6],Nbit=Nbit,fpga=1) 
            # print 'sum', sum
            Npoints_in_a_window=self.ask_from_serial(8,fpga=1)
            # print 'Npoints_in_a_window', Npoints_in_a_window   
            # print 'Number of traces', self.get_number_traces(ch=1)     
            Npoints=Npoints_in_a_window*(self.get_number_traces(ch=1))
            if Npoints != 0:
                ch1_av=sum*1.0/Npoints
            # ch1_av = sum
        if 2 in ch:
            sum=self.ask_from_serial_signed([22,21],Nbit=Nbit,fpga=1)    
            Npoints_in_a_window=self.ask_from_serial(23,fpga=1) 
            Npoints=Npoints_in_a_window*(self.get_number_traces(ch=2))            
            if Npoints != 0:
                ch2_av=sum*1.0/Npoints
        
        if 3 in ch:
            sum=self.ask_from_serial_signed([7,6],Nbit=Nbit,fpga=2)    
            Npoints_in_a_window=self.ask_from_serial(8,fpga=2)    
            Npoints=Npoints_in_a_window*(self.get_number_traces(ch=3))
            if Npoints != 0:
                ch3_av=sum*1.0/Npoints
        if 4 in ch:
            sum=self.ask_from_serial_signed([22,21],Nbit=Nbit,fpga=2)    
            Npoints_in_a_window=self.ask_from_serial(23,fpga=2) 
            Npoints=Npoints_in_a_window*(self.get_number_traces(ch=4))            
            if Npoints != 0:
                ch4_av=sum*1.0/Npoints    
            
        if  (not (1 in ch)) and (not (2 in ch)) and (not (3 in ch)) and (not (4 in ch)):
            raise ValueError('channel is not defined properly')
            return False
        
        all=[ch1_av,ch2_av,ch3_av,ch4_av]
        finaloutput=[all[jcnt-1] for jcnt in ch]
        
        return finaloutput

    def do_get_Npoints_in_a_window(self,ch=[1]):
        '''
        Returns the points included in the gating
        '''
        if type(ch)==type(1):
            ch=[ch]
        
        Npoints_in_a_window1=0
        Npoints_in_a_window2=0
        Npoints_in_a_window3=0
        Npoints_in_a_window4=0
            
        if 1 in ch:
            Npoints_in_a_window1=self.ask_from_serial(8,fpga=1)
            # print 'Npoints_in_a_window', Npoints_in_a_window   
        if 2 in ch:
            Npoints_in_a_window2=self.ask_from_serial(23,fpga=1) 
        if 3 in ch:
            Npoints_in_a_window3=self.ask_from_serial(8,fpga=2)    
        if 4 in ch:
            Npoints_in_a_window4=self.ask_from_serial(23,fpga=2) 
            
        if  (not (1 in ch)) and (not (2 in ch)) and (not (3 in ch)) and (not (4 in ch)):
            raise ValueError('channel is not defined properly')
            return False
        
        all=[Npoints_in_a_window1,Npoints_in_a_window2,Npoints_in_a_window3,Npoints_in_a_window4]
        finaloutput=[all[jcnt-1] for jcnt in ch]
        
        return finaloutput
        
    def do_get_average_check_points(self,ch=[1],Nbit=16,Ntraces=1,Npoints_in_window=1):
        '''
        Returns the average of the signal over set number of cycles
        Commented out where this talks with the FPGA, to reduce the error rate
        '''
        # print 'AVERAGE!'
        if type(ch)==type(1):
            ch=[ch]
        
        ch1_av=0
        ch2_av=0
        
        ch3_av=0
        ch4_av=0
        
        if 1 in ch:
            sum=self.ask_from_serial_signed([7,6],Nbit=Nbit,fpga=1) 
            # print 'sum', sum
            Npoints_in_a_window=self.ask_from_serial(8,fpga=1)
            # print 'Npoints_in_a_window', Npoints_in_a_window   
            # print 'Number of traces', self.get_number_traces(ch=1)     
            if Npoints_in_a_window==Npoints_in_window:
                Npoints=Npoints_in_a_window*Ntraces
                # if Npoints != 0:
                    # ch1_av=sum*1.0/Npoints
                ch1_av=sum*1.0/Npoints
            # ch1_av = sum
        if 2 in ch:
            sum=self.ask_from_serial_signed([22,21],Nbit=Nbit,fpga=1)    
            Npoints_in_a_window=self.ask_from_serial(23,fpga=1) 
            if Npoints_in_a_window==Npoints_in_window:
                Npoints=Npoints_in_a_window*Ntraces     
                ch2_av=sum*1.0/Npoints
        
        if 3 in ch:
            sum=self.ask_from_serial_signed([7,6],Nbit=Nbit,fpga=2)    
            Npoints_in_a_window=self.ask_from_serial(8,fpga=2)    
            if Npoints_in_a_window==Npoints_in_window:
                Npoints=Npoints_in_a_window*Ntraces
                ch3_av=sum*1.0/Npoints
        if 4 in ch:
            sum=self.ask_from_serial_signed([22,21],Nbit=Nbit,fpga=2)    
            Npoints_in_a_window=self.ask_from_serial(23,fpga=2) 
            if Npoints_in_a_window==Npoints_in_window:
                Npoints=Npoints_in_a_window*Ntraces      
                ch4_av=sum*1.0/Npoints    
            
        if  (not (1 in ch)) and (not (2 in ch)) and (not (3 in ch)) and (not (4 in ch)):
            raise ValueError('channel is not defined properly')
            return False
        
        all=[ch1_av,ch2_av,ch3_av,ch4_av]
        finaloutput=[all[jcnt-1] for jcnt in ch]
        
        return finaloutput
        
    def do_get_min(self,ch=[1],Nbit=12):
        '''
        Returns one if the minimum of the FIFO within a certain amount of cycles
        '''
         
        if type(ch)==type(1):
            ch=[ch]
        
        ch1_min=0
        ch2_min=0
        
        ch3_min=0
        ch4_min=0
        
        if 1 in ch:
            ch1_min=self.ask_from_serial_signed(12,Nbit=Nbit,fpga=1)    
        if 2 in ch:
            ch2_min=self.ask_from_serial_signed(27,Nbit=Nbit,fpga=1)
        if 3 in ch:
            ch3_min=self.ask_from_serial_signed(12,Nbit=Nbit,fpga=2)    
        if 4 in ch:
            ch4_min=self.ask_from_serial_signed(27,Nbit=Nbit,fpga=2)
        if  (not (1 in ch)) and (not (2 in ch)) and (not (3 in ch)) and (not (4 in ch)):
            raise ValueError('channel is not defined properly')
            return False
        
        all=[ch1_min,ch2_min,ch3_min,ch4_min]
        finaloutput=[all[jcnt-1] for jcnt in ch]
        return finaloutput
    def do_get_max(self,ch=[1],Nbit=12):
        '''
        Returns one if the maximum of the FIFO within a certain amount of cycles
        '''
        if type(ch)==type(1):
            ch=[ch]
            
        ch1_max=0
        ch2_max=0
        ch3_max=0
        ch4_max=0
        
        if 1 in ch:
            ch1_max=self.ask_from_serial_signed(11,Nbit=Nbit,fpga=1)    
        if 2 in ch:
            ch2_max=self.ask_from_serial_signed(26,Nbit=Nbit,fpga=1)
        if 3 in ch:
            ch3_max=self.ask_from_serial_signed(11,Nbit=Nbit,fpga=2)    
        if 4 in ch:
            ch4_max=self.ask_from_serial_signed(26,Nbit=Nbit,fpga=2)
            
        
        all=[ch1_max,ch2_max,ch3_max,ch4_max]
        finaloutput=[all[jcnt-1] for jcnt in ch]
        
        return finaloutput
    def do_set_invert(self,value,ch=[1]):
        '''
        invert the signal
            value 0 : not inverted
            value 1 or higher : inverted
        '''
        if type(ch)==type(1):
            ch=[ch]
        if 1 in ch:
            self.write_to_serial(133,value,fpga=1)
        if 2 in ch:
            self.write_to_serial(164,value,fpga=1)
        if 3 in ch:
            self.write_to_serial(133,value,fpga=2)
        if 4 in ch:
            self.write_to_serial(164,value,fpga=2)
            
    def do_get_invert(self,ch=[1]):
        '''
        invert the signal
            value 0 : not inverted
            value 1 or higher : inverted
        '''
        if type(ch)==type(1):
            ch=[ch]
        ch1p=0
        ch2p=0
        
        ch3p=0
        ch4p=0
        
        if 1 in ch:
            ch1p=self.ask_from_serial(133,fpga=1)
        if 2 in ch:
            ch2p=self.ask_from_serial(164,fpga=1)
        
        if 3 in ch:
            ch3p=self.ask_from_serial(133,fpga=2)
        if 4 in ch:
            ch4p=self.ask_from_serial(164,fpga=2)
            
        # self._invert=[ch1p,ch2p,ch3p,ch4p]
        
        all=[ch1p,ch2p,ch3p,ch4p]
        finaloutput=[all[jcnt-1] for jcnt in ch]
        return finaloutput
    def close_serial_ports(self):
        self.serialconnection1.close()
        
        if len(self._address)==2:
            self.serialconnection2.close()
    def do_get_cycle_number(self,ch=1):
        ch=[ch]
        
        if 1 in ch:
            output=self.ask_from_serial(13,fpga=1)
        if 2 in ch:
            output=self.ask_from_serial(28,fpga=1)
        
        if 3 in ch:
            output=self.ask_from_serial(13,fpga=2)
        if 4 in ch:
            output=self.ask_from_serial(28,fpga=2)
        
        return output
    def do_get_first_point_detection_mode(self,ch=1):
        '''
        Returns one if the measurement is done, else returns zero
        '''
        if ch==1:
            return self.ask_from_serial(135,fpga=1)
        elif ch==2:
            return self.ask_from_serial(166,fpga=1)
        elif ch==3:
            return self.ask_from_serial(135,fpga=2)
        elif ch==4:
            return self.ask_from_serial(166,fpga=2)
            
    def do_set_first_point_detection_mode(self,value,ch=1):
        '''
        Set how the first point of FIFO is going to be analyzed
        input:
            4,2,1
            4:  THRESHOLD + hysteresis
            2:  THRESHOLD - hysteresis
            1 or any other value : THRESHOLD 
        '''
        
        if ch==1:
            self.write_to_serial(135,value,fpga=1)
        elif ch==2:
            self.write_to_serial(166,value,fpga=1)
        elif ch==3:
            self.write_to_serial(135,value,fpga=2)
        elif ch==4:
            self.write_to_serial(166,value,fpga=2)
    def do_set_PSB_mode(self,value,ch=1):
        '''
        0: FPGA runs in Elzerman read-out mode: so pulse detection during the read-out stage
        1: FPGA runs in PSB-mode: it will detect the average during the read-out window and determine if the average is below/above the threshold
        
        0 = off
        1 = on
        
        Stores True/False in _PSB_mode when 1/0
        
        '''
        
        if value==1:
            self.write_to_serial(168,1,fpga=1)
            self._PSB_mode = True
        elif value==0:
            self.write_to_serial(168,0,fpga=1)
            self._PSB_mode = False    
            
    def do_get_PSB_mode(self):
        '''
        
        0 = off
        1 = on
        
        '''
        return self.ask_from_serial(168,fpga=1)
            
    # def do_set_enable_PSB(self):
        # self.write_to_serial(168,1,fpga=1)
        # self._PSB_mode = True
        
    # def do_set_disable_PSB(self):
        # self.write_to_serial(168,0,fpga=1)
        # self._PSB_mode = False
      
    #CHANGE DON"T THINK IT IS CORRECT
    def do_import_PSB_data(self,ch = 1):
        if ch == 1:
            return ask_from_serial_big_data(14,ch = [1])
        elif ch == 2:
            return ask_from_serial_big_data(29, ch = [2])
        
    def do_get_Nwindows_PSB(self):
            return self.get_number_traces(ch=1)