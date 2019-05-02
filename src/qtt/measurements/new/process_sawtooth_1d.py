from qtt.measurements.new.interfaces.signal_processor_interface import SignalProcessorInterface

from qilib.data_set import DataSet, DataArray


class ProcessSawtooth1D(SignalProcessorInterface):
    def run_process(self, signal_data: DataSet) -> DataSet:
        data_set = DataSet(user_data=signal_data.user_data)
        data_array = next(iter(signal_data.data_arrays.values()))

        width = data_set.user_data['width']
        sample_count = len(data_array)

        sliced_data = data_array[:int(sample_count * width)]
        identifier = f'{data_array.name}_SawtoothProcessed1D'
        data_set.add_array(DataArray(identifier, data_array.label, preset_data=sliced_data))

        return data_set

