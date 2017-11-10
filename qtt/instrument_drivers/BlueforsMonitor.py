import sys
import glob
import getopt

from datetime import datetime
from DistributedInstrument import InstrumentDataClient
from DistributedInstrument import InstrumentDataServer


# -----------------------------------------------------------------------------

class FridgeDataReceiver(InstrumentDataClient):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.add_measurable_quantity('temperature', 'K', -1,
                                     'Cold plate temperature')
        self.add_measurable_quantity('pressure', 'bar', -1,
                                     'Maxigauge pressure')
        self.add_measurable_quantity('datetime', '', -1,
                                     'Read time from server')


# -----------------------------------------------------------------------------

class FridgeDataSender():

    _T_file_ext_ = "CH*T*.log"
    _P_file_ext_ = "maxigauge*.log"

    def __init__(self, folder_path, **kwargs):
        self._check_folder_path_(folder_path)
        quantities = {'datetime': self.get_datetime,
                      'temperature': self.get_temperatures,
                      'pressure': self.get_pressures}
        _data_server_ = InstrumentDataServer(quantities, **kwargs)
        _data_server_.start()

    def _check_folder_path_(self, path):
        self._folder_path_ = path
        # check if folder and .../17-10-11/ exists..

    def _read_file_(self, file_path):
        with open(file_path, 'r') as fstream:
            last_line = fstream.readlines()[-1]
        return last_line.strip().split(",")

    def _read_temperature_(self, file_path):
        temperature_data = self._read_file_(file_path)
        return float(temperature_data[-1])

    def get_temperatures(self):
        today = datetime.now().strftime('%y-%m-%d')
        T_directory = '{0}\\{1}\\{2}'.format(self._folder_path_, today,
                                             FridgeDataSender._T_file_ext_)
        print(T_directory)
        temperature_files = glob.glob(T_directory)
        print(temperature_files)
        assert(len(temperature_files) == 5)
        T = [self._read_temperature_(file) for file in temperature_files]
        return {'1 - PT1': T[0], '2 - PT2': T[1], '3 - Magnet': T[2],
                '5 - Still': T[3], '6 - MC': T[4]}

    def _read_pressure_(self, file_path):
        P = self._read_file_(file_path)
        return {'P1': float(P[5]), 'P2': float(P[11]), 'P3': float(P[17]),
                'P4': float(P[23]), 'P5': float(P[29]), 'P6': float(P[35])}

    def get_pressures(self):
        today = datetime.now().strftime('%y-%m-%d')
        P_directory = '{0}\\{1}\\{2}'.format(self._folder_path_, today,
                                             FridgeDataSender._P_file_ext_)
        pressure_files = glob.glob(P_directory)
        assert(len(pressure_files) == 1)
        return self._read_pressure_(pressure_files[0])

    def get_datetime(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def start_data_logger(self):
        self._data_server_.run()

    def stop_data_logger(self):
        self._data_server_.stop()


# -----------------------------------------------------------------------------

class BlueforsApp():

    _short_options_ = "?hd:u:p:n:"
    _long_options_ = ['--help', '--dir', '--un', '--pw', '--port']

    def __init__(self):
        self._directory_ = ''  # todo: set default folder...
        self._username_ = None
        self._password_ = None
        self._port_ = 8080

    def print_header(self):
        print("\n### BLUEFORS Data Proxy ###")
        print(" QuTech 2017, Delft, The Netherlands, www.qutech.nl\n")

    def print_usage(self, status=0):
        self.print_header()
        print(' BlueForsMonitor.exe [options]')
        print('    [-h, -?, --help]   print usage information')
        print('    [-d, --dir]        sets the log-data directory')
        print('    [-n, --port]       sets the proxy port number\n')
        print('    [-u, --user]       sets the server username')
        print('    [-p, --pw]         sets the server password')
        sys.exit(status)

    def print_settings(self):
        self.print_header()
        print(' Starting proxy with settings:')
        print("    directory : {0}".format(self._directory_))
        print("    portnumber : {0}".format(self._port_))
        print("    username : {0}".format(self._username_))
        print("    password : {0}\n".format(self._password_))

    def set_password(self, password: str):
        self._password_ = password

    def set_username(self, username: str):
        self._username_ = username

    def set_portnumber(self, port: str):
        try:
            self._port_ = int(port)
        except ValueError:
            self.print_usage(-3)

    def set_directory(self, directory: str):
        self._directory_ = directory

    def check_directory(self):
        pass  # todo check if exists

    def main(self, argv):
        if len(argv) < 2:
            self.print_usage(-1)
        try:
            options, arguments = getopt.getopt(argv[1:],
                                               BlueforsApp._short_options_,
                                               BlueforsApp._long_options_)
        except getopt.GetoptError:
            self.print_usage(-2)
        for option, argument in options:
            if option in ('-?', '-h', '--help'):
                self.print_usage()
            elif option in ("-u", "--un"):
                self.set_username(argument)
            elif option in ("-p", "--pw"):
                    self.set_password(argument)
            elif option in ("-d", "--dir"):
                self.set_directory(argument)
            elif option in ("-n", "--port"):
                self.set_portnumber(argument)
        self.print_settings()
        self.check_directory()
        FridgeDataSender(self._directory_, port=self._port_,
                         user=self._username_, password=self._password_)


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    BlueforsApp().main(sys.argv)


'''
#Base examples

## Server
directory = 'D:\\workspace\\QuTech\\data\\Fridge\\17-11-10'
FridgeDataSender(folder_path=directory)

## Client
client = FridgeDataReceiver(name='fridge')
print(client.temperature())
print(client.pressure())
print(client.datetime())

'''
