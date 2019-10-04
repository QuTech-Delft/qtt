from typing import Dict, Any, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

from qcodes import DataSet, DataArray
from qtt.utilities.visualization import plot_vertical_line


def generate_allxy_combinations() -> List[Any]:
    """ Generate all combinations of the AllXY sequence from Reed 2013 """
    xymapping = {'I': 'I', 'x': 'X90', 'y': 'Y90', 'X': 'X180', 'Y': 'Y180'}
    ground_state_rotations = ['II', 'XX', 'YY', 'XY', 'YX']
    equator_state_rotations = ['xI', 'yI', 'xy', 'yx', 'xY', 'yX', 'Xy', 'Yx', 'xX', 'Xx', 'yY', 'Yy']
    excited_state_rotations = ['XI', 'YI', 'xx', 'yy']
    allxy_combinations_input = ground_state_rotations + equator_state_rotations + excited_state_rotations
    allxy_combinations = [(xymapping[x[0]], xymapping[x[1]]) for x in allxy_combinations_input]

    return allxy_combinations


def allxy_model(indices: Union[float, np.ndarray], offset0: float, slope0: float, offset1: float, slope1: float, offset2: float, slope2: float) -> Union[float, np.ndarray]:
    """ Model for AllXY experiment

    The model consists of three linear segments. The segments correspond to the pairs of gates that result in
    fraction 0, 0.5 and 1 in the AllXY experiment.

    Args:
        index: Indices of the allxy pairs or a single index
        offset0: Offset of first segment
        slope0: Slope of first segment
        offset1: Offset of second segment
        slope1: Slope of second segment
        offset2: Offset of last segment
        slope2: Slope of last segment
    Returns:
        Fractions for the allxy pairs
    """
    indices = np.array(indices)
    x0 = indices < 4.5
    x1 = np.logical_and(indices >= 4.5, indices < 16.5)
    x2 = (indices >= 16.5)

    v1 = x0 * (offset0 + (indices - 2.) * slope0)
    v2 = x1 * (offset1 + (indices - 10.5) * slope1)
    v3 = x2 * (offset2 + (indices - 19.) * slope2)

    return v1 + v2 + v3


def _estimate_allxy_parameters(allxy_data: np.ndarray) -> List[float]:
    """ Return estimate of allxy model parameters """
    p = [np.mean(allxy_data[0:5]), 0, np.mean(allxy_data[5:17]), 0, np.mean(allxy_data[17:]), 0]
    return p


def _default_measurement_array(dataset: DataSet) -> DataArray:
    mm = [name for (name, a) in dataset.arrays.items() if not a.is_setpoint]
    return dataset.arrays[mm[0]]


def fit_allxy(dataset: DataSet, initial_parameters: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """ Fit AllXY measurement to piecewise linear model

    Args:
        dataset: Dataset containing the AllXY measurement
        initial_parameters: Optional set of initialization parameters
    Returns:
        Dictionary with the fitting results
    """
    allxy_data = _default_measurement_array(dataset)

    x_data = np.arange(21)
    lmfit_model = Model(allxy_model)

    if initial_parameters is None:
        initial_parameters = _estimate_allxy_parameters(allxy_data)
    param_names = lmfit_model.param_names
    result = lmfit_model.fit(allxy_data, indices=x_data, **dict(zip(param_names, initial_parameters)), verbose=0)
    fitted_parameters = np.array([result.best_values[p] for p in param_names])
    fitted_parameters_covariance = np.diag(result.covar)
    chi_squared = result.chisqr

    return {'fitted_parameters': fitted_parameters, 'description': 'allxy fit', 'initial_parameters': initial_parameters, 'fitted_parameters_covariance': fitted_parameters_covariance, 'chi_squared': chi_squared}


def plot_allxy(dataset: DataSet, result: Dict[str, Any], fig: int = 1, plot_initial_estimate: bool = False):
    """ Plot the results of an AllXY fit

    Args:
        dataset: Dataset containing the measurement data
        result: Fitting result of fit_allxy
        int: Figure handle
        plot_initial_guess: If True, then plot the initial estimate of the model
    """
    allxy_data = _default_measurement_array(dataset)
    xy_pairs = generate_allxy_combinations()
    x_data = np.arange(21)

    plt.figure(fig)
    plt.clf()
    fitted_parameters = result['fitted_parameters']
    xfine = np.arange(0, 21, 1e-3)
    plt.plot(xfine, allxy_model(xfine, *fitted_parameters), 'm', label='fitted allxy', alpha=.5)

    plt.plot(x_data, allxy_data, '.b', label='allxy data')

    if plot_initial_estimate:
        p = [0, 0, .5, 0, 1, 0]
        plt.plot(xfine, allxy_model(xfine, *p), 'c', label='baseline', alpha=.5)

        initial_params = _estimate_allxy_parameters(allxy_data)
        plt.plot(xfine, allxy_model(xfine, *initial_params), 'g', label='initial estimate', alpha=.35)

        initial_parameters = result['initial_parameters']
        plt.plot(xfine, allxy_model(xfine, *initial_parameters), ':g', label='initial estimate', alpha=.35)

    plt.xticks(x_data, [v[0] + "," + v[1] for v in xy_pairs], rotation='vertical')
    vl = plot_vertical_line(4.5)
    vl.set_linestyle(':')
    vl = plot_vertical_line(16.5)
    vl.set_linestyle(':')
    plt.title('AllXY')

    plt.legend()
