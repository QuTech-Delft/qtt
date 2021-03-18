from typing import Optional, Any, Union

import matplotlib.pyplot as plt
import numpy as np

import qtt.algorithms.functions


def plot_horizontal_line(x: float, color: str = 'c', alpha: float = .5, label: Optional[str] = None) -> Any:
    """ Plot vertical alignment line

    Args:
        x: Position on horizontal axis to plot the line
        color: Color specification of the line
        alpha: Value to use for the transparency of the line
        label: Label for the line
    Return:
        Handle to the plotted line
    """
    vline = plt.axhline(x, label=label)
    vline.set_alpha(alpha)
    vline.set_color(color)
    vline.set_linestyle('--')
    return vline


def plot_vertical_line(x: float, color: str = 'c', alpha: float = .5, label: Optional[str] = None) -> Any:
    """ Plot vertical alignment line

    Args:
        x: Position on horizontal axis to plot the line
        color: Color specification of the line
        alpha: Value to use for the transparency of the line
        label: Label for the line
    Return:
        Handle to the plotted line

    """
    vline = plt.axvline(x, label=label)
    vline.set_alpha(alpha)
    vline.set_color(color)
    vline.set_linestyle('--')
    return vline


def plot_double_gaussian_fit(result_dict: dict, xdata: np.ndarray) -> None:
    """ Plot a two Gaussians from a double Gaussian fit

    Args:
        result_dict: Result of the double Gaussian fitting
        xdata: Independent data
    """
    plt.plot(xdata, qtt.algorithms.functions.gaussian(xdata, *result_dict['left']), 'g', label='left')
    _ = plt.plot(xdata, qtt.algorithms.functions.gaussian(xdata, *result_dict['right']), 'r', label='right')


def plot_single_traces(traces: np.ndarray, time: Optional[np.ndarray] = None, trace_color: Optional[np.ndarray]
                       = None, offset: Union[None, bool, float] = None, fig: int = 1, maximum_number_of_traces: int = 40):
    """ Plot single traces with offset for separation

    Args:
       traces: Array with single traces in the rows
       time: Option array for time axis
       trace_color: Specification of trace color
       offset: Offset to use between traces. For None automatically determine the offset
       fig: Specification of Matplotlib window
       maximum_number_of_traces: Maximum number of traces to plot
    """
    if time is None:
        time = np.arange(traces.shape[1])
    if trace_color is None:
        trace_color = np.zeros(traces.shape[0])
    if offset is False:
        offset = 0
    if offset is None:
        offset = (np.percentile(traces, 99) - np.percentile(traces, 1)) * 1.95
    maximum_number_of_traces = min(maximum_number_of_traces, traces.shape[0])

    color_map = {0: 'b', 1: 'r', 2: 'm'}
    plt.figure(fig)
    plt.clf()
    for ii in range(maximum_number_of_traces):
        trace_offset = ii * offset

        color = color_map.get(trace_color[ii], 'c')
        plt.plot(time, traces[ii] + trace_offset, color=color)
    plt.xlabel('Time [us]')
    plt.ylabel('Signal [a.u.]')
    _ = plt.title('Elzerman traces (spin-down in blue, spin-up in red)')
