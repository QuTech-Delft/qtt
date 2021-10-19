import unittest

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.optimize

from qtt.algorithms.functions import linear_function
from qtt.utilities.optimization import (AverageDecreaseTermination,
                                        OptimizerCallback)


class TestOptimizationUtilities(unittest.TestCase):

    def test_OptimizerCallback(self):

        def rosen(params, a=1, b=100, noise=0):
            """ Rosenbrock function """
            v = (a-params[0])**2+b*(params[1]-params[0]**2)**2
            v += noise*(np.random.rand()-.5)
            return v

        def objective(x):
            return rosen(x, 0.01, 1)

        oc = OptimizerCallback(show_progress=False)

        result = scipy.optimize.minimize(objective, [.5, .9], callback=oc.scipy_callback)
        self.assertEqual(result.success, True)
        self.assertEqual(oc.number_of_evaluations(), result.nit)
        self.assertIsInstance(oc.data, pandas.DataFrame)

        oc.clear()

        lmfit_model = lmfit.Model(linear_function, independent_vars=['x'])
        xdata = np.linspace(0, 10, 40)
        data = 10+5*xdata
        lmfit_model.fit(data, x=xdata, a=1, b=1, iter_cb=oc.lmfit_callback)
        plt.figure(100)
        plt.clf()
        oc.plot(logy=True)
        plt.close(100)

    def test_AverageDecreaseTermination(self):
        tc = AverageDecreaseTermination(4)
        results = [tc(0, None, value, None, None) for value in [4, 3, 2, 1, .1, 0, 0, 0, 0, 0.01, 0]]
        self.assertEqual(results, [False, False, False, False, False, False, False, False, False, True, True])

        tc = AverageDecreaseTermination(4, tolerance=1.)
        results = [tc(0, None, value, None, None) for value in [4, 3, 2, 1, .1, 0, 0, 0, 0, 0.01, 1., 20.]]
        self.assertEqual(results, [False, False, False, False, False, False, False, False, False, False, False, True])
