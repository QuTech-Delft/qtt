from typing import Sequence, Callable, Optional
import numpy as np
import copy

import qcodes
import qtt.data
from qtt.data import DataSet

# %%


def process_dataarray(dataset: DataSet, input_array_name: str, output_array_name: str, processing_function: Callable, label: Optional[str] = None, unit: Optional[str] = None, ) -> DataSet:
    """ Apply a function to a DataArray in a DataSet  """
    array = dataset.default_parameter_array(input_array_name)
    data = processing_function(np.array(array))
    if label is None:
        label = array.label
    if unit is None:
        unit = array.unit
    data_array = qcodes.DataArray(array_id=output_array_name, name=output_array_name, label=label,
                                  set_arrays=array.set_arrays, preset_data=data, unit=unit)
    dataset.add_array(data_array)
    return dataset


def dataset_dimension(dataset) -> int:
    """ Return dimension of DataSet """
    return len(dataset.default_parameter_array().set_arrays)


def average_dataset(dataset: qtt.data.DataSet, axis='vertical') -> qtt.data.DataSet:
    """ Calculate the mean signal of a 2D dataset over the specified axis

    Args:
        dataset: DataSet to be processed
        axis: Specification of the axis

    """

    if dataset_dimension(dataset) != 2:
        raise Exception('average_dataset only implemented for 2D datasets')

    if axis == 'vertical':
        axis = 0
    if axis == 1 or axis == 'horizontal':
        raise Exception('average_dataset not implemented for horizontal axis')

    zarray = dataset.default_parameter_array()
    set_arrays = zarray.set_arrays
    xarray = set_arrays[1]

    data = zarray
    averaged_signal = data.mean(axis=0)

    dataset_averaged = qtt.data.makeDataSet1Dplain(xarray.name, xarray[0], yname='signal', y=averaged_signal,
                                                   xunit=xarray.unit, yunit=zarray.unit)

    return dataset_averaged


def calculate_averaged_dataset(dataset: DataSet, number_of_repetitions: int) -> DataSet:
    """ Calculate the averaged signal from a 2D dataset with repeated rows """
    zarray = dataset.default_parameter_array()
    set_arrays = zarray.set_arrays
    xarray = set_arrays[1]
    yarray = set_arrays[0]

    unique_detunings = np.array(dataset.metadata['detunings'])
    data = zarray
    ncolumns = data.shape[1]
    averaged_signal = data.transpose().reshape(-1, number_of_repetitions).mean(1).reshape(ncolumns, -1).transpose()

    dataset_averaged = qtt.data.makeDataSet2Dplain(xarray.name, xarray[0], yarray.name, unique_detunings, zname='signal',
                                                   z=averaged_signal, xunit=xarray.unit, yunit=yarray.unit, zunit=zarray.unit)

    return dataset_averaged


# %%


def slice_dataset(dataset: DataSet, window: Sequence[float], axis: int = 0, verbose: int = 0, copy_metadata: bool = False) -> DataSet:
    """ Given a dataset and a window for the horizontal axis return the dataset with selected window """
    zarray = dataset.default_parameter_array()

    set_arrays = zarray.set_arrays
    yarray = set_arrays[0]

    scan_dimension = len(set_arrays)
    is_1d_dataset = len(set_arrays) == 1

    if is_1d_dataset:
        if not axis==0:
            raise AssertionError('for a 1D dataset axis should be 0')
    else:
        xarray = set_arrays[1]

    if axis == 0:
        slice_array = yarray
        start_idx = int(np.floor(np.interp(window[0], slice_array.ndarray, np.arange(slice_array.ndarray.size))))
        end_idx = int(np.interp(window[1], slice_array.ndarray, np.arange(slice_array.ndarray.size)))
    else:
        slice_array = xarray

        start_idx = int(np.floor(np.interp(window[0], slice_array.ndarray[0], np.arange(slice_array.ndarray[0].size))))
        end_idx = int(np.interp(window[1], slice_array.ndarray[0], np.arange(slice_array.ndarray[0].size)))

    if verbose:
        print(f'slice_dataset: dimension {scan_dimension} start_idx {start_idx}, end_idx {end_idx}')

    if axis == 0:
        if is_1d_dataset:
            signal_window = zarray[start_idx:end_idx]
            dataset_window = qtt.data.makeDataSet1Dplain(yarray.name, yarray[start_idx:end_idx], yname='signal',
                                                         y=signal_window, xunit=yarray.unit, yunit=zarray.unit)
        else:
            signal_window = zarray[start_idx:end_idx, :]
            dataset_window = qtt.data.makeDataSet2Dplain(xarray.name, xarray[0], yarray.name, yarray[start_idx:end_idx], zname='signal',
                                                         z=signal_window, xunit=xarray.unit, yunit=yarray.unit, zunit=zarray.unit)
    else:
        signal_window = zarray[:, start_idx:end_idx]
        dataset_window = qtt.data.makeDataSet2Dplain(xarray.name, xarray[0][start_idx:end_idx], yarray.name, yarray, zname='signal',
                                                     z=signal_window, xunit=xarray.unit, yunit=yarray.unit, zunit=zarray.unit)

    if copy_metadata:
        dataset_window.metadata = copy.deepcopy(dataset.metadata)

    return dataset_window
