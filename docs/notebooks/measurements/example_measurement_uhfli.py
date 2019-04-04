from qcodes import Station, Measure
from qcodes.instrument_drivers.ZI.ZIUHFLI import ZIUHFLI

from qtt.measurements.new import UhfliScopeReader
from qtt.measurements.new.configuration_storage import load_adapter, save_adapter


# STATION

device_id = 'DEV2338'
uhfli = ZIUHFLI('uhfli', device_id)
station = Station(uhfli, update_snapshot=False)


# SCOPE READER

file_path = 'D:\\Users\\lcblom\\data\\InstrumentAdapters\\uhfli.ia'

scope_reader = UhfliScopeReader(device_id)

configuration = load_adapter(file_path)
scope_reader.initialize(configuration)
# save_adapter(file_path, scope_reader._adapter)


# PREPAIR

scope_reader.set_scope_signals([1], ['Demod 1 R'])
scope_reader.set_trigger_enabled(False)

scope_reader.prepare_acquisition()


# READOUT

for i in range(10):
    result = scope_reader.acquire()
    # print(result)
