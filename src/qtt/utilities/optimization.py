import datetime
import logging
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AverageDecreaseTermination:

    def __init__(self, N: int, tolerance:  float = 0):
        """ Callback to terminate optimization based the average decrease

        The average decrease over the last N data points is compared to the specified tolerance

        Args:
            N: Number of data points to use
            tolerance: Abort if the average decrease is smaller than the specified tolerance

        """
        self.N = N
        self.tolerance = tolerance
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()

    def reset(self):
        """ Reset the data """
        self.values = []

    def __call__(self, nfev, parameters, value, update, accepted) -> bool:
        """
        Returns:
            True if the optimization loop should be aborted
        """
        self.values.append(value)

        if len(self.values) > self.N:
            last_values = self.values[-self.N:]
            pp = np.polyfit(range(self.N), last_values, 1)
            slope = pp[0] / self.N

            self.logger.debug(f'AverageDecreaseTermination: slope {slope}, tolerance {self.tolerance}')
            if slope > self.tolerance:
                self.logger.info(
                    f'AverageDecreaseTermination(N={self.N}): terminating with slope {slope}, tolerance {self.tolerance}')
                return True
        return False


class OptimizerCallback:

    _column_names = ['iteration', 'timestamp', 'residual']

    def __init__(self, show_progress=False, store_data=True):
        """ Class to collect data of optimization procedures

        The class contains methods that can be used as callbacks is several well-known packages.
        """
        self.show_progress = show_progress
        self.store_data = store_data
        self.logger = logging.getLogger(self.__class__.__name__)

        self.clear()

    def decorate(self, *args):
        """  Decorate an objective function """
        raise NotImplementedError('')

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

    def number_of_evaluations(self) -> int:
        """ Return the number of callback evaluations

        Note: this can differ from the number of objective evaluations
        """
        return len(self.data)

    def optimization_time(self) -> float:
        """ Return time difference between the first and the last invocation of the callback"

        Returns:
            Time in seconds
        """
        if len(self.data) > 1:
            delta_t = self.data.iloc[-1]['timestamp']-self.data.iloc[0]['timestamp']
            dt = delta_t.total_seconds()
        else:
            dt = 0
        return dt

    def plot(self, ax=None, **kwargs):
        """ Plot optimization results """
        if ax is None:
            ax = plt.gca()

        self.data.plot('iteration', 'residual', ax=ax, **kwargs)
        dt = self.optimization_time()
        ax.set_title(f'Optimization total time {dt:.2f} [s]')

    def data_callback(self, iteration: int, parameters: Any, residual: float):
        """ Callback used to store data """
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
        if self.show_progress:
            print(f'#{iteration}, {parameters}, {residual}')
        residual = np.linalg.norm(residual)
        self.data_callback(iteration, parameters, residual)

    def scipy_callback(self, parameters):
        """ Callback method for scipy optimizers """
        number_evaluations = self.number_of_evaluations()
        value = np.NaN
        if self.show_progress:
            print(f'#{number_evaluations}, {parameters}, {value}')
        self.data_callback(number_evaluations, parameters, value)


if __name__ == '__main__':
    import scipy.optimize
    from qiskit.algorithms.optimizers import SPSA

    def rosen(params, a=1, b=100, noise=0):
        """ Rosenbrock function """
        v = (a-params[0])**2+b*(params[1]-params[0]**2)**2
        v += noise*(np.random.rand()-.5)
        return v

    def objective(x):
        return rosen(x, 0.01, 1)

    oc = OptimizerCallback(show_progress=True)

    result = scipy.optimize.minimize(objective, [.5, .9], callback=oc.scipy_callback)

    oc.clear()
    optimizer = SPSA(maxiter=90, callback=oc.qiskit_callback, termination_callback=AverageDecreaseTermination(10))
    point, value, eval = optimizer.optimize(2, objective, initial_point=[0.4, 0.98])
    print(f'point {point}, {value}')
    plt.figure(100)
    plt.clf()
    oc.plot(logy=True)

    print(f'number_of_evaluations {oc.number_of_evaluations()} dt {oc.optimization_time()}')
    from qtt.utilities.tools import measure_time

    with measure_time():
        for ii in range(20):
            oc.data_callback(0, [2, 3], .1)
    with measure_time():
        for ii in range(1000):
            oc.data_callback(0, [2, 3], .1)
    with measure_time():
        for ii in range(20):
            oc.data_callback(0, [2, 3], .1)

    ts = datetime.datetime.now()
    data = [[1, ts, 2], [3, ts, 3]]
    pd.DataFrame(data, columns=['a', 'b', 'c'])
