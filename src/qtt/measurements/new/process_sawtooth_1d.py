from qtt.measurements.new.interfaces.signal_processor_interface import SignalProcessorInterface

from qilib.data_set import DataSet, DataArray


class ProcessSawtooth1D(SignalProcessorInterface):

    def run_process(self, signal_data: DataSet) -> DataSet:
        """ Extracts a 1D image from a readout dot responce measured with an acquisition device

        Args:
            signal_data: The readout dot reponse data coming from the acquisition device. The data
                         user data of the data set should contain the width and resolution settings.

        Returns:
            A data set which contains a 1D image with the charge stability diagram
        """

        data_set = DataSet(user_data=signal_data.user_data)
        width = data_set.user_data['width']

        for data_array in signal_data.data_arrays.values():
            sample_count = len(data_array)

            sliced_data = data_array[:int(sample_count * width)]
            identifier = f'{data_array.name}_SawtoothProcessed1D'
            data_set.add_array(DataArray(identifier, data_array.label, preset_data=sliced_data))

        return data_set
