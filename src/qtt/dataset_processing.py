import copy
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from qcodes_loop.data.data_array import DataArray
from qcodes_loop.data.data_set import DataSet

import qtt.data

# %%


def process_dataarray(dataset: DataSet, input_array_name: str, output_array_name: Optional[str],
                      processing_function: Callable, label: Optional[str] = None,
                      unit: Optional[str] = None, ) -> DataSet:
    """ Apply a function to a DataArray in a DataSet

    Args:
        dataset: Input dataset containing the data array
        input_array_name: Name of the data array to be processed
        output_array_nane: Name of the output array or None to operate in place
        processing_function: Method to apply to the data array
        label: Label for the output array
        unit: Unit for the output array
    """
    array = dataset.default_parameter_array(input_array_name)
    data = processing_function(np.array(array))
    if label is None:
        label = array.label
    if unit is None:
        unit = array.unit
    if output_array_name is None:
        array.ndarray[:] = data
    else:
        data_array = DataArray(array_id=output_array_name, name=output_array_name, label=label,
                                      set_arrays=array.set_arrays, preset_data=data, unit=unit)
        dataset.add_array(data_array)
    return dataset


def dataset_dimension(dataset: DataSet) -> int:
    """ Return dimension of DataSet """
    return len(dataset.default_parameter_array().set_arrays)


def average_dataset(dataset: qtt.data.DataSet, axis: Union[str, int] = 'vertical') -> qtt.data.DataSet:
    """ Calculate the mean signal of a 2D dataset over the specified axis

    Args:
        dataset: DataSet to be processed
        axis: Specification of the axis

    Returns:
        Dataset with averaged signal
    """

    if dataset_dimension(dataset) != 2:
        raise Exception('average_dataset only implemented for 2D datasets')

    if axis == 'vertical':
        axis = 0
    if axis == 'horizontal':
        axis = 1

    zarray = dataset.default_parameter_array()
    set_arrays = zarray.set_arrays
    if axis == 0:
        new_setpoint_array = set_arrays[1]
        new_setpoint_array_data = new_setpoint_array[0]
        averaged_signal = zarray.mean(axis=0)
    else:
        new_setpoint_array = set_arrays[0]
        new_setpoint_array_data = new_setpoint_array
        averaged_signal = zarray.mean(axis=1)

    dataset_averaged = qtt.data.makeDataSet1Dplain(new_setpoint_array.name, new_setpoint_array_data, yname=zarray.name,
                                                   y=averaged_signal,
                                                   xunit=new_setpoint_array.unit, yunit=zarray.unit)

    return dataset_averaged


def average_multirow_dataset(dataset: DataSet, number_of_repetitions: int, new_values=None,
                             parameter_name: str = 'signal', output_parameter_name: str = 'signal') -> DataSet:
    """ Calculate the averaged signal from a 2D dataset with repeated rows

    Args:
        dataset: Dataset containing the data to be averaged
        number_of_repetitions: Number of rows over which to average
        new_values: Optional new values for the averaged axis
        parameter_name: Name of data array to process
        output_parameter_name: Name of output array

    Returns:
        Averaged dataset
    """
    zarray = dataset.default_parameter_array(parameter_name)
    set_arrays = zarray.set_arrays
    xarray = set_arrays[1]
    yarray = set_arrays[0]

    if new_values is None:
        number_of_blocks = int(zarray.shape[0] / number_of_repetitions)
        new_values = np.linspace(yarray[0], yarray[-1], number_of_blocks)
    data = zarray
    ncolumns = data.shape[1]
    averaged_signal = data.transpose().reshape(-1, number_of_repetitions).mean(1).reshape(ncolumns, -1).transpose()

    dataset_averaged = qtt.data.makeDataSet2Dplain(xarray.name, xarray[0], yarray.name, new_values,
                                                   zname=output_parameter_name,
                                                   z=averaged_signal, xunit=xarray.unit, yunit=yarray.unit,
                                                   zunit=zarray.unit)

    return dataset_averaged


# %%


def slice_dataset(dataset: DataSet, window: Sequence[float], axis: int = 0,
                  verbose: int = 0, copy_metadata: bool = False, output_parameter_name=None) -> DataSet:
    """ Given a dataset and a window for the horizontal axis return the dataset with selected window

    Args:
        dataset: Dataset to be slice
        window: Specification of the window to be selected
        axis: Axis used for slicing
        verbose: Verbosity level
        copy_metadata: If True then copy the metadata of the input dataset
        output_parameter_name: Name of the output array
    Returns:
        Dataset with sliced data

    """
    zarray = dataset.default_parameter_array()
    if output_parameter_name is None:
        output_parameter_name = zarray.name

    set_arrays = zarray.set_arrays
    yarray = set_arrays[0]
    scan_dimension = dataset_dimension(dataset)
    is_1d_dataset = scan_dimension == 1

    if is_1d_dataset:
        if not axis == 0:
            raise AssertionError('for a 1D dataset axis should be 0')
    else:
        xarray = set_arrays[1]

    slice_objects = [slice(0, size) for jj, size in enumerate(zarray.shape)]

    if axis == 0:
        slice_array = yarray
        start_idx = int(np.floor(np.interp(window[0], slice_array.ndarray, np.arange(slice_array.ndarray.size))))
        end_idx = int(np.interp(window[1], slice_array.ndarray, np.arange(slice_array.ndarray.size)))
        slice_objects[0] = slice(start_idx, end_idx)
    else:
        slice_array = xarray

        start_idx = int(np.floor(np.interp(window[0], slice_array.ndarray[0], np.arange(slice_array.ndarray[0].size))))
        end_idx = int(np.interp(window[1], slice_array.ndarray[0], np.arange(slice_array.ndarray[0].size)))

        slice_objects[1] = slice(start_idx, end_idx)

    return _slice_dataset(dataset, tuple(slice_objects), output_parameter_name, copy_metadata=copy_metadata, verbose=0)


def _slice_dataset(dataset: DataSet, slice_objects: Sequence[slice], output_parameter_name: Optional[str],
                   copy_metadata: bool, verbose: int = 0):
    """ Slice the measurement array of a dataset and adjust the setpoints arrays accordingly """
    zarray = dataset.default_parameter_array()
    if output_parameter_name is None:
        output_parameter_name = zarray.name

    set_arrays = zarray.set_arrays
    yarray = set_arrays[0]

    scan_dimension = dataset_dimension(dataset)
    is_1d_dataset = scan_dimension == 1
    is_2d_dataset = scan_dimension == 2

    if verbose:
        print(f'slice_dataset: dimension {scan_dimension} slice_objects {slice_objects}')

    if is_1d_dataset:
        signal_window = zarray[tuple(slice_objects)]
        dataset_window = qtt.data.makeDataSet1Dplain(yarray.name, yarray[slice_objects[0]], yname=output_parameter_name,
                                                     y=signal_window, xunit=yarray.unit, yunit=zarray.unit)
    elif is_2d_dataset:
        xarray = set_arrays[1]
        signal_window = zarray[tuple(slice_objects)]
        dataset_window = qtt.data.makeDataSet2Dplain(xarray.name, xarray[0][slice_objects[1]], yarray.name,
                                                     yarray[slice_objects[0]], zname=output_parameter_name,
                                                     z=signal_window, xunit=xarray.unit, yunit=yarray.unit,
                                                     zunit=zarray.unit)
    else:
        raise NotImplementedError('slicing a multi-dimensional dataset of dimension {scan_dimension} is not supported')

    if copy_metadata:
        dataset_window.metadata = copy.deepcopy(dataset.metadata)
    return dataset_window


def resample_dataset(dataset: DataSet, sample_rate: Tuple[int], copy_metadata: bool = False,
                     output_parameter_name: Optional[str] = None) -> DataSet:
    """ Given a dataset resample the measurement array

    Args:
        dataset: Dataset to be slice
        sample_rate: Tuple with for each axis the sample rate. Must be a postive integer
        copy_metadata: If True then copy the metadata of the input dataset
        output_parameter_name: Name of the output array
    Returns:
        Dataset with sliced data

    """
    zarray = dataset.default_parameter_array()
    if output_parameter_name is None:
        output_parameter_name = zarray.name

    slice_objects = tuple(slice(0, size, sample_rate[jj]) for jj, size in enumerate(zarray.shape))

    return _slice_dataset(dataset, slice_objects, output_parameter_name, copy_metadata=copy_metadata, verbose=0)
