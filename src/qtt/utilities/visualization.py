from typing import Any, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import qtt.algorithms.functions


def get_axis(handle: Union[int, Axes, Figure, None]) -> Axes:
    """ Create or return matplotlib axis object

    Args:
        handle: Specification of how to obtain the axis object. For an integer, generate a new figure.
            For an Axis object, return the handle.
            For a Figure, return the default axis of the figure.
            For None, use the matplotlib current axis.
    Returns:
        Axis object
    """
    if handle is None:
        return plt.gca()
    elif isinstance(handle, Axes):
        return handle
    elif isinstance(handle, int):
        plt.figure(handle)
        plt.clf()
        return plt.gca()
    elif isinstance(handle, Figure):
        plt.figure(handle)
        return plt.gca()
    else:
        raise NotImplementedError('handle {handle} of type {type(handle)}  is not implemented')


def combine_legends(axis_list: List[matplotlib.axes.Axes], target_ax: Optional[matplotlib.axes.Axes] = None):
    """ Combine legends of a list of matplotlib axis objects into a single legend

    Args:
        axis_list: List of matplotlib axis containing legends
        target_ax: Axis to add the combined legend to. If None, use the first axis from the `axis_list`

    Example:
        import matplotlib.pyplot as plt
        ax1=plt.gca()
        plt.plot([1,2,3], [.1,.2,.3], '.b', label='X')
        plt.legend()
        ax2=ax1.twinx()
        ax2.plot([1,2,3], [1, 2, 3], '-r', label='miliX' )
        plt.legend()
        combine_legends([ax1, ax2])

    """
    lines: List[Any] = []
    labels: List[Any] = []
    for ax in axis_list:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines.extend(lines1)
        labels.extend(labels1)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    if target_ax is None:
        target_ax = next(iter(axis_list), None)

    if target_ax is not None:
        target_ax.legend(lines, labels)


def plot_horizontal_line(x: float, color: str = 'c', alpha: float = .5, label: Optional[str] = None,
                         ax: Optional[Axes] = None) -> Any:
    """ Plot vertical alignment line

    Args:
        x: Position on horizontal axis to plot the line
        color: Color specification of the line
        alpha: Value to use for the transparency of the line
        label: Label for the line
        ax: Matplotlib axis handle to plot to. If None, select the default handle

    Returns:
        Handle to the plotted line
    """
    if ax is None:
        ax = plt.gca()
    vline = ax.axhline(x, label=label)
    vline.set_alpha(alpha)
    vline.set_color(color)
    vline.set_linestyle('--')
    return vline


def plot_vertical_line(x: float, color: str = 'c', alpha: float = .5, label: Optional[str] = None, ax: Optional[Axes] = None) -> Any:
    """ Plot vertical alignment line

    Args:
        x: Position on horizontal axis to plot the line
        color: Color specification of the line
        alpha: Value to use for the transparency of the line
        label: Label for the line
        ax: Matplotlib axis handle to plot to. If None, select the default handle

    Returns:
        Handle to the plotted line

    """
    if ax is None:
        ax = plt.gca()
    vline = ax.axvline(x, label=label)
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
        p1, p99 = np.percentile(traces, [1, 99])
        offset = (p99-p1) * 1.95
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
