"""
Unit test for constraint conversion
"""

import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_allclose, assert_warns, suppress_warnings)
import pytest
from trust_constr import (NonlinearConstraint, LinearConstraint,
                            OptimizeWarning, minimize, BFGS)
from test_minimize_constrained import (Maratos, HyperbolicIneq, Rosenbrock,
                                        IneqRosenbrock, EqIneqRosenbrock,
                                        BoundedRosenbrock, Elec)


class TestOldToNew(object):
    x0 = (2, 0)
    bnds = ((0, None), (0, None))
    method = "trust-constr"

    def test_constraint_dictionary_1(self):
        fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
                {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
                {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.0")
            res = minimize(fun, self.x0, method=self.method,
                           bounds=self.bnds, constraints=cons)
        assert_allclose(res.x, [1.4, 1.7], rtol=1e-4)
        assert_allclose(res.fun, 0.8, rtol=1e-4)

    def test_constraint_dictionary_2(self):
        fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
        cons = {'type': 'eq',
                'fun': lambda x, p1, p2: p1*x[0] - p2*x[1],
                'args': (1, 1.1),
                'jac': lambda x, p1, p2: np.array([[p1, -p2]])}
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.0")
            res = minimize(fun, self.x0, method=self.method,
                           bounds=self.bnds, constraints=cons)
        assert_allclose(res.x, [1.7918552, 1.62895927])
        assert_allclose(res.fun, 1.3857466063348418)

    def test_constraint_dictionary_3(self):
        fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
        cons = [{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
                NonlinearConstraint(lambda x: x[0] - x[1], 0, 0)]

        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.0")
            res = minimize(fun, self.x0, method=self.method,
                           bounds=self.bnds, constraints=cons)
        assert_allclose(res.x, [1.75, 1.75], rtol=1e-4)
        assert_allclose(res.fun, 1.125, rtol=1e-4)


class TestNewToOld(object):

    def test_multiple_constraint_objects(self):
        fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2 + (x[2] - 0.75)**2
        x0 = [2, 0, 1]
        coni = []  # only inequality constraints (can use cobyla)
        methods = ["trust-constr",]

        # mixed old and new
        coni.append([{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
                     NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])

        coni.append([LinearConstraint([1, -2, 0], -2, np.inf),
                     NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])

        coni.append([NonlinearConstraint(lambda x: x[0] - 2 * x[1] + 2, 0, np.inf),
                     NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])

        for con in coni:
            funs = {}
            for method in methods:
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    result = minimize(fun, x0, method=method, constraints=con)
                    funs[method] = result.fun
            assert_allclose(funs['trust-constr'], .8, rtol=1e-4)

    def test_individual_constraint_objects(self):
        fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2 + (x[2] - 0.75)**2
        x0 = [2, 0, 1]

        cone = []  # with equality constraints (can't use cobyla)
        coni = []  # only inequality constraints (can use cobyla)
        methods = ["trust-constr",]

        # nonstandard data types for constraint equality bounds
        cone.append(NonlinearConstraint(lambda x: x[0] - x[1], 1, 1))
        cone.append(NonlinearConstraint(lambda x: x[0] - x[1], [1.21], [1.21]))
        cone.append(NonlinearConstraint(lambda x: x[0] - x[1],
                                        1.21, np.array([1.21])))

        # multiple equalities
        cone.append(NonlinearConstraint(
                    lambda x: [x[0] - x[1], x[1] - x[2]],
                    1.21, 1.21))  # two same equalities
        cone.append(NonlinearConstraint(
                    lambda x: [x[0] - x[1], x[1] - x[2]],
                    [1.21, 1.4], [1.21, 1.4]))  # two different equalities
        cone.append(NonlinearConstraint(
                    lambda x: [x[0] - x[1], x[1] - x[2]],
                    [1.21, 1.21], 1.21))  # equality specified two ways
        cone.append(NonlinearConstraint(
                    lambda x: [x[0] - x[1], x[1] - x[2]],
                    [1.21, -np.inf], [1.21, np.inf]))  # equality + unbounded

        # nonstandard data types for constraint inequality bounds
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], 1.21, np.inf))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], [1.21], np.inf))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1],
                                        1.21, np.array([np.inf])))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], -np.inf, -3))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1],
                                        np.array(-np.inf), -3))

        # multiple inequalities/equalities
        coni.append(NonlinearConstraint(
                    lambda x: [x[0] - x[1], x[1] - x[2]],
                    1.21, np.inf))  # two same inequalities
        cone.append(NonlinearConstraint(
                    lambda x: [x[0] - x[1], x[1] - x[2]],
                    [1.21, -np.inf], [1.21, 1.4]))  # mixed equality/inequality
        coni.append(NonlinearConstraint(
                    lambda x: [x[0] - x[1], x[1] - x[2]],
                    [1.1, .8], [1.2, 1.4]))  # bounded above and below
        coni.append(NonlinearConstraint(
                    lambda x: [x[0] - x[1], x[1] - x[2]],
                    [-1.2, -1.4], [-1.1, -.8]))  # - bounded above and below

        # quick check of LinearConstraint class (very little new code to test)
        cone.append(LinearConstraint([1, -1, 0], 1.21, 1.21))
        cone.append(LinearConstraint([[1, -1, 0], [0, 1, -1]], 1.21, 1.21))
        cone.append(LinearConstraint([[1, -1, 0], [0, 1, -1]],
                                     [1.21, -np.inf], [1.21, 1.4]))

        # Solutions for SciPy 1.6.0 trust-constr algorithm
        solutions = [3.672050010240001, 3.672050010240001,
                     3.672050010240001, 1.1250000102400013,
                     1.1250000102400013, 4.114869685317061,
                     3.4616666889075245, 3.7616666954827016]
        for i, con in enumerate(coni):
            funs = {}
            for method in methods:
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    result = minimize(fun, x0, method=method, constraints=con)
                    funs[method] = result.fun
            assert_allclose(funs['trust-constr'], solutions[i], rtol=1e-3)

        # Solutions from SciPy 1.6.0 trust-constr algorithm
        solutions = [3.125000000000001, 3.67205, 3.67205,
                     4.114866666666668, 4.345399999999999,
                     4.114866666666668, 3.67205, 3.67205,
                     3.672050000000001, 4.114866666666666,
                     3.6720500000000005]
        for i, con in enumerate(cone):
            funs = {}
            for method in methods[::2]:  # skip cobyla
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    result = minimize(fun, x0, method=method, constraints=con)
                    funs[method] = result.fun
            assert_allclose(funs['trust-constr'], solutions[i], rtol=1e-5)
