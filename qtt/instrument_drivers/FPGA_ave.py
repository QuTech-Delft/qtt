import logging
import functools
import time
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.visa import visa


class FPGA_ave(VisaInstrument):
    '''
    This is the python driver for the FPGA averaging, it communicates with the FPGA to get an average of a pulse

    Usage:
    Initialize with
    <name> = instruments.create('name', 'FPGA_AVE', address='<COM PORT>')
    <COM PORT> = COM5 e.g.
    '''
    def __init__(self, name, address, mirrorfactors=[1, 1], verbose=1, **kwargs):
        logging.debug(__name__ + ' : Initializing instrument')
        super().__init__(name, address, **kwargs)

        self._address = address
        self._values = {}
        self.verbose = verbose
        self._total_cycle_num = 100
        self._sampling_frequency = 1000e3
        self.visa_handle.baud_rate = 57600
        self.set_sampling_frequency(self._sampling_frequency)
        self.mirrorfactors = mirrorfactors

        # Add parameters
        self.add_parameter('mode',
                           set_cmd=functools.partial(self.write_to_serial, 130))
        self.add_parameter('ch1_cycle_num',
                           get_cmd=functools.partial(self.ask_from_serial, 1))
        self.add_parameter('ch2_cycle_num',
                           get_cmd=functools.partial(self.ask_from_serial, 5))
        self.add_parameter('measurement_done',
                           get_cmd=self.get_measurement_done)
        self.add_parameter('total_cycle_num',
                           get_cmd=self.get_total_cycle_num,
                           set_cmd=self.set_total_cycle_num)
        self.add_parameter('ch1_datapoint_num',
                           get_cmd=self.get_ch1_datapoint_num)
        self.add_parameter('ch2_datapoint_num',
                           get_cmd=self.get_ch2_datapoint_num)
        self.add_parameter('data',
                           get_cmd=self.get_data)
        self.add_parameter('ch1_data',
                           get_cmd=self.get_ch1_data)
        self.add_parameter('ch2_data',
                           get_cmd=self.get_ch2_data)
        self.add_parameter('sampling_frequency',
                           get_cmd=self.get_sampling_frequency,
                           set_cmd=self.set_sampling_frequency)

        v = self.visa_handle
        # make sure term characters are ignored
        logging.debug(__name__ + ' : set termchar settings')
        v.set_visa_attribute(visa.constants.VI_ATTR_TERMCHAR_EN, 0)
        v.set_visa_attribute(visa.constants.VI_ATTR_ASRL_END_IN, 0)
        logging.debug(__name__ + ' : completed initialization')

    def get_idn(self):
        logging.debug(__name__ + ' : FPGA_ave: get_idn')
        IDN = {'vendor': None, 'model': 'FPGA',
               'serial': None, 'firmware': None}
        return IDN

#    def get_all(self):
#        for cnt in self.get_parameter_names():
#            self.get(cnt)

    def serial(self, here):
        self.visa_handle.write_raw(here)

    def start(self):
        self.write_to_serial(129, 1)

    def write_to_serial(self, address, value):
        #        register=chr(address)
        #        first_byte=chr(value>>8)
        #        last_byte=chr(value&255)
        self.visa_handle.write_raw(bytes([address]))
        self.visa_handle.write_raw(bytes([value >> 8]))
        self.visa_handle.write_raw(bytes([value & 255]))

    def ask_from_serial(self, address, register=255):
        self.visa_handle.write_raw(bytes([register]))
        self.visa_handle.write_raw(bytes([0]))
        self.visa_handle.write_raw(bytes([address]))
        readresult = self.read_raw_bytes(size=3)
        result = (int(readresult[1]) << 8) | int(readresult[2])

        return result

    def ask_from_serial_signed(self, address, register=255):
        self.visa_handle.write_raw(chr(register))
        self.visa_handle.write_raw(chr(0))
        self.visa_handle.write_raw(chr(address))

        readresult = self.read_raw_bytes(size=3)

        if int(readresult[0]) == address:
            unsigned = (int(readresult[1]) << 8) | int(readresult[2])
            signed = unsigned - 256 if unsigned > 127 else unsigned
            return signed
        else:
            raise Exception('Wrong address returned')

    def read_raw_bytes(self, size=None):
        '''
        Returns the values that are in the FPGA buffer. Replacement for 
        read_raw in messagebased.py, also see:
        https://github.com/hgrecco/pyvisa/issues/93
        https://github.com/hgrecco/pyvisa/issues/190
        '''
        size = self.visa_handle.chunk_size if size is None else size

        with self.visa_handle.ignore_warning(visa.constants.VI_SUCCESS_MAX_CNT):
            chunk, status = self.visa_handle.visalib.read(self.visa_handle.session, size)

        if not len(chunk) == size:
            raise Exception('Visa received incorrect number of bytes')

        return chunk

    def set_(self, value):
        '''
        Set the number of traces over which pulse events will be counted
        '''
        self.write_to_serial(129, value)

    def get_total_cycle_num(self):
        '''
        Get the total number of cycles which are averaged in FPGA
        '''
        return self._total_cycle_num

    def set_total_cycle_num(self, value):
        '''
        Set the total number of cycles which are averaged in FPGA
        '''
        self._total_cycle_num = value
        self.write_to_serial(131, value)

    def get_ch1_datapoint_num(self):
        '''
        Get the total number of cycles which are averaged in FPGA
        '''
        return self.ask_from_serial(3) - 1

    def get_ch2_datapoint_num(self):
        '''
        Get the total number of cycles which are averaged in FPGA
        '''
        return self.ask_from_serial(7) - 1

    def get_measurement_done(self, ch=[1, 2]):
        '''
        Returns one if the measurement is done, else returns zero
        '''
        meas_done = True

        for ch_name in ch:
            meas_tmp = self.get('ch%i_cycle_num' % ch_name)
            meas_done = meas_done and (meas_tmp == self._total_cycle_num)

        return meas_done

    def get_data(self, address=2):
        '''
        Read data ch1, unsigned
        '''
        if not self.get_measurement_done():
            return False

        self.visa_handle.write_raw(chr(255))
        self.visa_handle.write_raw(chr(0))
        self.visa_handle.write_raw(chr(address))

        result = []
        for x in range(0, 1000, 1):
            readresult = self.read_raw_bytes(size=3)
            result.append((int(readresult[1]) << 8) | int(readresult[2]))

        self._data = result

        return result

    def get_ch1_data(self, address=2, checkdone=True, buf=True):
        '''
        Reads signed data out of the FPGA
        '''
        Npoint = self.get_ch1_datapoint_num()

        if checkdone:
            if not self.get_measurement_done(ch=[1]):
                return False

        self.visa_handle.write_raw(bytes([255]))
        self.visa_handle.write_raw(bytes([0]))
        self.visa_handle.write_raw(bytes([address]))

        signed = []

        if Npoint == 0:
            raise ValueError('There is no fpga output, the number of data points recorded for ch1 is 0 ')

        if buf:
            readresultbuf = self.read_raw_bytes(3 * Npoint)

            if len(readresultbuf) != 3 * Npoint:
                print('Npoint %d' % Npoint)
                raise Exception('get_ch1_data: error reading data')
            for x in range(0, Npoint, 1):
                readresult = readresultbuf[3 * x:3 * (x + 1)]
                unsigned = (int(readresult[0]) << 16) | (int(readresult[1]) << 8) | int(readresult[2])
                signed_temp = unsigned - 16777215 if unsigned > 8388607 else unsigned
                if x < Npoint:
                    signed.append(signed_temp)
        else:
            for x in range(0, Npoint, 1):
                readresult = self.read_raw_bytes(size=3)
                unsigned = (int(readresult[0]) << 16) | (int(readresult[1]) << 8) | int(readresult[2])
                signed_temp = unsigned - 16777215 if unsigned > 8388607 else unsigned
                if x < Npoint:
                    signed.append(signed_temp)

        self._data_signed = signed
        self.visa_handle.flush(16)

        signed = [x * self.mirrorfactors[0] for x in signed]

        return signed

    def get_ch2_data(self, address=6, checkdone=True):
        '''
        Reads signed data out of the FPGA
        '''
        Npoint = self.get_ch2_datapoint_num()

        if checkdone:
            if not self.get_measurement_done(ch=[2]):
                return False

        self.visa_handle.write_raw(bytes([255]))
        self.visa_handle.write_raw(bytes([0]))
        self.visa_handle.write_raw(bytes([address]))

        signed = []

        if Npoint == 0:
            raise ValueError('There is no fpga output, the number of data points recorded for ch2 is 0 ')

        readresultbuf = self.read_raw_bytes(3 * Npoint)

        for x in range(0, Npoint, 1):
            readresult = readresultbuf[3 * x:3 * (x + 1)]
            unsigned = (int(readresult[0]) << 16) | (int(readresult[1]) << 8) | int(readresult[2])
            signed_temp = unsigned - 16777215 if unsigned > 8388607 else unsigned
            if x < Npoint:
                signed.append(signed_temp)

        self._data_signed = signed
        self.visa_handle.flush(16)

        signed = [x * self.mirrorfactors[1] for x in signed]

        return signed

    def set_sampling_frequency(self, value):
        '''
        Set the total number of cycles which are averaged in FPGA, maximum samp freq=1 MHz and minimum is freq=763.
        '''
        if value > 1e6:
            raise ValueError('The sampling frequency can not be set higher than 1 MHz.')
        internal_clock = 50e6
        Ndivision = round(internal_clock * 1.0 / value)

        fs = internal_clock / Ndivision
        self._sampling_frequency = fs
        if self.verbose:
            print('FPGA internal clock is 50MHz, dividing it by %d, yields samp. freq. is %d Hz' % (Ndivision, self._sampling_frequency))

        return self.write_to_serial(132, int(Ndivision))

    def get_sampling_frequency(self):
        '''
        Get the total number of cycles which are averaged in FPGA
        '''
        return self._sampling_frequency

    def get_sampling_frequency_ratio(self):
        '''
        Get the total number of cycles which are averaged in FPGA
        '''
        return self.ask_from_serial(132)

    def readFPGA(self, FPGA_mode=0, ReadDevice=['FPGA_ch1', 'FPGA_ch2'], Naverage=1, verbose=1, waittime=0):
        '''
        Basic function to read the data from the FPGA memory.
        '''
        t0 = time.clock()

        self.set('mode', FPGA_mode)
        self.set('total_cycle_num', Naverage)
        self.start()

        if verbose >= 2:
            print('  readFPGA: dt %.3f [ms], after start' % (1e3 * (time.clock() - t0)))

        time.sleep(waittime)

        if verbose >= 2:
            print('  readFPGA: dt %.3f [ms], after wait' % (1e3 * (time.clock() - t0)))

        cc = []
        if 'FPGA_ch1' in ReadDevice:
            cc += [1]
        if 'FPGA_ch2' in ReadDevice:
            cc += [2]

        for c in cc:
            loop = 0
            while not self.get_measurement_done(ch=[c]):
                if verbose >= 2:
                    print('readFPGA: waiting for FPGA for 7.5 ms longer time (channel %d)' % c)
                time.sleep(0.0075)
                loop = loop + 1
                if loop * .0075 > 1:
                    pass
                    raise Exception('readFPGA: error')

        if verbose >= 2:
            print('  readFPGA: dt %.3f [ms], after dynamic wait' % (1e3 * (time.clock() - t0)))

        DataRead_ch1 = []
        if 'FPGA_ch1' in ReadDevice:
            DataRead_ch1 = self.get_ch1_data(checkdone=False, buf=True)

        DataRead_ch2 = []
        if 'FPGA_ch2' in ReadDevice:
            DataRead_ch2 = self.get_ch2_data(checkdone=False)

        if verbose >= 2:
            print('  readFPGA: dt %.3f [ms]' % (1e3 * (time.clock() - t0)))

        if verbose >= 2:
            print('  readFPGA: DataRead_ch1 %d, DataRead_ch2 %d' % (len(DataRead_ch1), len(DataRead_ch2)))

        totalpoints = max(len(DataRead_ch1), len(DataRead_ch2))

        return totalpoints, DataRead_ch1, DataRead_ch2

#%% Testing driver functionality
if __name__ == '__main__':
    server_name = None
    fpga = FPGA_ave_x('FPGA', 'ASRL4::INSTR', server_name=server_name)
    self = fpga
