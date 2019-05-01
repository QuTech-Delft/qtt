from qtt.measurements.new.interfaces.signal_processor_interface import SignalProcessorInterface

from qilib.data_set import DataSet


class ProcessSawtooth1D(SignalProcessorInterface):
    def run_process(self, signal_data: DataSet) -> DataSet:
        return signal_data
