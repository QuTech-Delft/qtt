#%%
from qcodes import Instrument
from functools import partial
from zmqrpc.ZmqRpcClient import ZmqRpcClient
from zmqrpc.ZmqRpcServer import ZmqRpcServerThread

from datetime import datetime
import glob

# rename to DistributedInstrument

#--------------------------------------------------------------------------------------------------

'''Represents a client for collecting instrument measurable quantities from a server.'''
class InstrumentDataClient(Instrument):

    def __init__(self, name, address='localhost', port=8080, user=None, password=None, **kwargs):
        super().__init__(name, **kwargs)
        self._client_ = ZmqRpcClient(["tcp://{0}:{1}".format(address, port)], username=user, password=password)

    def __proxy_wrapper__(self, command_name, default_value, sec_time_out=3):
        try: return self._client_.invoke(command_name, time_out_waiting_for_response_in_sec=sec_time_out)
        except: return default_value

    def add_measurable_quantity(self, name='quantity', unit='arb.', default_value=None, doc_string='Unknown'):
        command = partial(self.__proxy_wrapper__, command_name=name, default_value=default_value)
        self.add_parameter(name, unit=unit, get_cmd=command, docstring=doc_string)

#--------------------------------------------------------------------------------------------------

class InstrumentDataServer():

    def __init__(self, functions, address='*', port=8080, user=None, password=None):
        self._server_ = ZmqRpcServerThread("tcp://{0}:{1}".format(address, port), rpc_functions=functions,
                                           username=user, password=password)

    def start(self):
        print('Enabled instrument server...')
        print('Press CTRL+C to quit!')
        try:
            self._server_.start()
            while(True) : continue
        except KeyboardInterrupt as ex:
            print('Done')
        finally:
            self._server_.stop()
            self._server_.join()

#--------------------------------------------------------------------------------------------------

class FridgeDataReveiver(InstrumentDataClient):

    def __init__(self, name, **kwargs):
       super().__init__(name, **kwargs)
       self.add_measurable_quantity('temperature', 'K', -1, 'Cold plate temperature')
       self.add_measurable_quantity('pressure', 'bar', -1, 'Maxigauge pressure')
       self.add_measurable_quantity('datetime', '', -1, 'Read time from server')

#--------------------------------------------------------------------------------------------------

class FridgeDataSender():

    _T_file_ext_ = "CH*T*.log"
    _P_file_ext_ = "maxigauge*.log"

    def __init__(self, folder_path):
        self._check_folder_path_(folder_path)
        quantities = { 'datetime' : self.get_datetime, 'temperature' : self.get_temperatures, 
                      'pressure' : self.get_pressures }
        self._data_server_ = InstrumentDataServer(quantities)
        self._data_server_.start()

    def __del__(self):
        self._data_server_.__del__()

    def _check_folder_path_(self, path):
        self._folder_path_ = path
        #check if folder and .../17-10-11/ exists..

    def _read_file_(self, file_path):
        with open(file_path, 'r') as fstream: 
            last_line = fstream.readlines()[-1]
        return last_line.strip().split(",")

    def _read_temperature_(self, file_path):
        temperature_data = self._read_file_(file_path)
        return float(temperature_data[-1])

    def get_temperatures(self):
        today = datetime.now().strftime('%y-%m-%d')
        T_directory = '{0}\\{1}\\{2}'.format(self._folder_path_, today, FridgeDataSender._T_file_ext_)
        temperature_files = glob.glob(T_directory)
        assert(len(temperature_files) == 5)
        T = [self._read_temperature_(file) for file in temperature_files]
        return {'1 - PT1':T[0], '2 - PT2':T[1], '3 - Magnet':T[2], '5 - Still':T[3], '6 - MC':T[4] }

    def _read_pressure_(self, file_path):
        P = self._read_file_(file_path)
        return { 'P1':float(P[5]),'P2':float(P[11]),'P3':float(P[17]),'P4':float(P[23]),'P5':float(P[29]), 'P6':float(P[35]) }

    def get_pressures(self):
        today = datetime.now().strftime('%y-%m-%d')
        P_directory = '{0}\\{1}\\{2}'.format(self._folder_path_, today, FridgeDataSender._P_file_ext_)
        pressure_files = glob.glob(P_directory)
        assert(len(pressure_files) == 1)
        return self._read_pressure_(pressure_files[0])

    def get_datetime(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def start_data_logger(self):
        self._data_server_.run()

    def stop_data_logger(self):
        self._data_server_.stop()

#--------------------------------------------------------------------------------------------------

#%%
# server

directory = 'C:\\Workspace\\Qutech\\Resources\\_Other\\TestData\\Fridge'
FridgeDataSender(folder_path=directory)


#%%
#client

reader = FridgeDataReveiver(name='fridge2')
print(reader.temperature())
print(reader.pressure())
print(reader.datetime())

#--------------------------------------------------------------------------------------------------
