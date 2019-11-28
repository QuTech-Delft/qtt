import sys
from functools import partial

from qtt.instrument_drivers.BlueforsMonitorApp import BlueforsApp, FridgeDataSender
from qtt.instrument_drivers.DistributedInstrument import InstrumentDataClient

# -----------------------------------------------------------------------------


class FridgeDataReceiver(InstrumentDataClient):
    '''
    Receives temperature and pressure data from the Bluefors fridge with
    server connection.
    '''

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.add_measurable_quantity('temperatures', 'K', -1,
                                     'The CH temperature values')
        self.add_measurable_quantity('pressures', 'bar', -1,
                                     'The maxigauge pressures values')
        self.add_measurable_quantity('cpatempwo', '°C', None,
                                     'The compressor output water temperature',
                                     command_name='status',
                                     params={'item': 'cpatempwo'})
        self.add_measurable_quantity('cpatempwi', '°C', None,
                                     'The compressor input water temperature',
                                     command_name='status',
                                     params={'item': 'cpatempwi'})
        self.add_measurable_quantity('cpawarn', 'arb.', None,
                                     'The compressor status',
                                     command_name='status',
                                     params={'item': 'cpawarn'})
        self.add_measurable_quantity('datetime', '', -1,
                                     'The server date and time (for testing)')
        self.temperatures.get_latest.max_val_age = 1

        if self.temperatures() == -1:
            raise ConnectionError('Could not connect to the server!')

        def get_temp(key):
            return self.temperatures.get_latest()[key][0]

        for key in self.temperatures().keys():
            self.add_parameter('T' + key.lower(), unit='K',
                               get_cmd=partial(get_temp, key))


# -----------------------------------------------------------------------------
# Main block for creating py-installer


if __name__ == '__main__':
    BlueforsApp().main(sys.argv)

# -----------------------------------------------------------------------------
# Sample for local testing

# Python console 1: server
if None:
    from qtt.instrument_drivers.BlueforsMonitor import BlueforsApp

    argv = ['', '-d', '<fridge_data_dir>']
    BlueforsApp().main(argv)

# Python console 2: client
if None:
    from qtt.instrument_drivers.BlueforsMonitor import FridgeDataReceiver

    client = FridgeDataReceiver(name='dummy_fridge')
    print(client.temperatures())
    print(client.pressures())
    print(client.cpatempwo())
    print(client.cpatempwi())
    print(client.cpawarn())
    print(client.datetime())
    client.close()

# -----------------------------------------------------------------------------
