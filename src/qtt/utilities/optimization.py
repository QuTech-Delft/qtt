import datetime
import logging
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


class AverageDecreaseTermination:

    def __init__(self, N: int, tolerance: float = 0.):
        """ Callback to terminate optimization based the average decrease

        The average decrease over the last N data points is compared to the specified tolerance.
        The average decrease is determined by a linear fit (least squares) to the data.

        This class can be used as an argument to the Qiskit SPSA optimizer.

        Args:
            N: Number of data points to use
            tolerance: Abort if the average decrease is smaller than the specified tolerance

        """
        self.N = N
        self.tolerance = tolerance
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()

    @property
    def parameters(self):
        return self._parameters

    @property
    def values(self):
        return self._values

    def reset(self):
        """ Reset the data """
        self._values = []
        self._parameters = []

    def __call__(self, nfev, parameters, value, update, accepted) -> bool:
        """
        Args:
            nfev: Number of evaluations
            parameters: Current parameters in the optimization
            value: Value of the objective function
            update: Update step
            accepted: Whether the update was accepted

        Returns:
            True if the optimization loop should be aborted
        """
        self._values.append(value)
        self._parameters.append(parameters)

        if len(self._values) > self.N:
            last_values = self._values[-self.N:]
            pp = np.polyfit(range(self.N), last_values, 1)
            slope = pp[0] / self.N

            self.logger.debug(f'AverageDecreaseTermination(N={self.N}): slope {slope}, tolerance {self.tolerance}')
            if slope > self.tolerance:
                self.logger.info(
                    f'AverageDecreaseTermination(N={self.N}): terminating with slope {slope}, tolerance {self.tolerance}')
                return True
        return False


class OptimizerCallback:

    _column_names = ['iteration', 'timestamp', 'residual']

    def __init__(self, show_progress=False, store_data=True):
        """ Class to collect data of optimization procedures

        The class contains methods that can be used as callbacks on several well-known optimization packages.
        """
        self.show_progress = show_progress
        self.store_data = store_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self.clear()

    @property
    def data(self) -> pd.DataFrame:
        """ Return data gathered by callback """

        df = pd.DataFrame(self._data, columns=self._column_names)

        return df

    def _append(self, d: Tuple):
        """ Apppend a row of data """
        self._data.append(d)

    def clear(self):
        """ Clear the data from this instance """
        self.parameters = []
        self._data = []
        self._number_of_evaluations = 0

    def number_of_evaluations(self) -> int:
        """ Return the number of callback evaluations

        Note: this can differ from the number of objective evaluations
        """
        return self._number_of_evaluations

    def optimization_time(self) -> float:
        """ Return time difference between the first and the last invocation of the callback

        Returns:
            Time in seconds
        """
        if len(self.data) > 0:
            delta_t = self.data.iloc[-1]['timestamp']-self.data.iloc[0]['timestamp']
            dt = delta_t.total_seconds()
        else:
            dt = 0
        return dt

    def plot(self, ax: Optional[Axes] = None, **kwargs):
        """ Plot optimization results """
        if ax is None:
            ax = plt.gca()

        self.data.plot('iteration', 'residual', ax=ax, **kwargs)
        dt = self.optimization_time()
        ax.set_title(f'Optimization total time {dt:.2f} [s]')

    def data_callback(self, iteration: int, parameters: Any, residual: float):
        """ Callback used to store data

        Args:
            iteration: Iteration on the optimization procedure
            parameters: Current values of the parameters to be optimized
            residual: Current resisual (value of the objective function)

        """
        self._number_of_evaluations = self._number_of_evaluations + 1
        if self.store_data:
            self.logger.info('data_callback: {iteration} {parameters} {residual}')
            self.parameters.append(parameters)

            ts = datetime.datetime.now()  # .isoformat()
            d = (int(iteration), ts, float(residual))
            self._append(d)

    def qiskit_callback(self, number_evaluations, parameters, value, stepsize, accepted):
        """ Callback method for Qiskit optimizers """
        if self.show_progress:
            print(f'#{number_evaluations}, {parameters}, {value}, {stepsize}, {accepted}')
        self.data_callback(number_evaluations, parameters, value)

    def lmfit_callback(self, parameters, iteration, residual, *args, **kws):
        """ Callback method for lmfit optimizers """
        residual = np.linalg.norm(residual)
        if self.show_progress:
            print(f'#{iteration}, {parameters}, {residual}')
        self.data_callback(iteration, parameters, residual)

    def scipy_callback(self, parameters):
        """ Callback method for scipy optimizers """
        number_evaluations = self.number_of_evaluations()
        value = np.NaN
        if self.show_progress:
            print(f'#{number_evaluations}, {parameters}, {value}')
        self.data_callback(number_evaluations, parameters, value)
