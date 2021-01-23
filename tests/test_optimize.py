"""
Unit tests for optimization routines from optimize.py

Authors:
   Ed Schofield, Nov 2005
   Andrew Straw, April 2008

To run it in its simplest form::
  nosetests test_optimize.py

"""
import itertools
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
                           assert_, assert_almost_equal,
                           assert_no_warnings, assert_warns,
                           assert_array_less, suppress_warnings)
import pytest
from pytest import raises as assert_raises

import trust_constr as optimize

from trust_constr._differentiable_functions import ScalarFunction
from trust_constr import MemoizeJac, show_options

MINIMIZE_METHODS = ['trust-constr',]

def test_check_grad():
    # Verify if check_grad is able to estimate the derivative of the
    # logistic function.

    def logit(x):
        return 1 / (1 + np.exp(-x))

    def der_logit(x):
        return np.exp(-x) / (1 + np.exp(-x))**2

    x0 = np.array([1.5])

    r = optimize.check_grad(logit, der_logit, x0)
    assert_almost_equal(r, 0)

    r = optimize.check_grad(logit, der_logit, x0, epsilon=1e-6)
    assert_almost_equal(r, 0)

    # Check if the epsilon parameter is being considered.
    r = abs(optimize.check_grad(logit, der_logit, x0, epsilon=1e-1) - 0)
    assert_(r > 1e-7)


class CheckOptimize(object):
    """ Base test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """

    def setup_method(self):
        self.F = np.array([[1, 1, 1],
                           [1, 1, 0],
                           [1, 0, 1],
                           [1, 0, 0],
                           [1, 0, 0]])
        self.K = np.array([1., 0.3, 0.5])
        self.startparams = np.zeros(3, np.float64)
        self.solution = np.array([0., -0.524869316, 0.487525860])
        self.maxiter = 1000
        self.funccalls = 0
        self.gradcalls = 0
        self.trace = []

    def func(self, x):
        self.funccalls += 1
        if self.funccalls > 6000:
            raise RuntimeError("too many iterations in optimization routine")
        log_pdot = np.dot(self.F, x)
        logZ = np.log(sum(np.exp(log_pdot)))
        f = logZ - np.dot(self.K, x)
        self.trace.append(np.copy(x))
        return f

    def grad(self, x):
        self.gradcalls += 1
        log_pdot = np.dot(self.F, x)
        logZ = np.log(sum(np.exp(log_pdot)))
        p = np.exp(log_pdot - logZ)
        return np.dot(self.F.transpose(), p) - self.K

    def hess(self, x):
        log_pdot = np.dot(self.F, x)
        logZ = np.log(sum(np.exp(log_pdot)))
        p = np.exp(log_pdot - logZ)
        return np.dot(self.F.T,
                      np.dot(np.diag(p), self.F - np.dot(self.F.T, p)))

    def hessp(self, x, p):
        return np.dot(self.hess(x), p)


class TestOptimizeSimple(CheckOptimize):
    def test_custom(self):
        # This function comes from the documentation example.
        def custmin(fun, x0, args=(), maxfev=None, stepsize=0.1,
                    maxiter=100, callback=None, **options):
            bestx = x0
            besty = fun(x0)
            funcalls = 1
            niter = 0
            improved = True
            stop = False

            while improved and not stop and niter < maxiter:
                improved = False
                niter += 1
                for dim in range(np.size(x0)):
                    for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                        testx = np.copy(bestx)
                        testx[dim] = s
                        testy = fun(testx, *args)
                        funcalls += 1
                        if testy < besty:
                            besty = testy
                            bestx = testx
                            improved = True
                    if callback is not None:
                        callback(bestx)
                    if maxfev is not None and funcalls >= maxfev:
                        stop = True
                        break

            return optimize.OptimizeResult(fun=besty, x=bestx, nit=niter,
                                           nfev=funcalls, success=(niter > 1))

        x0 = [1.35, 0.9, 0.8, 1.1, 1.2]
        res = optimize.minimize(optimize.rosen, x0, method=custmin,
                                options=dict(stepsize=0.05))
        assert_allclose(res.x, 1.0, rtol=1e-4, atol=1e-4)

    def test_gh10771(self):
        # check that minimize passes bounds and constraints to a custom
        # minimizer without altering them.
        bounds = [(-2, 2), (0, 3)]
        constraints = 'constraints'

        def custmin(fun, x0, **options):
            assert options['bounds'] is bounds
            assert options['constraints'] is constraints
            return optimize.OptimizeResult()

        x0 = [1, 1]
        optimize.minimize(optimize.rosen, x0, method=custmin,
                          bounds=bounds, constraints=constraints)


    @pytest.mark.parametrize('method',
                             MINIMIZE_METHODS)
    def test_minimize_callback_copies_array(self, method):
        # Check that arrays passed to callbacks are not modified
        # inplace by the optimizer afterward

        # cobyla doesn't have callback
        if method == 'cobyla':
            return

        if method in ('fmin_tnc', 'fmin_l_bfgs_b'):
            func = lambda x: (optimize.rosen(x), optimize.rosen_der(x))
        else:
            func = optimize.rosen
            jac = optimize.rosen_der
            hess = optimize.rosen_hess

        x0 = np.zeros(10)

        # Set options
        kwargs = {}
        if method.startswith('fmin'):
            routine = getattr(optimize, method)
            if method == 'fmin_slsqp':
                kwargs['iter'] = 5
            elif method == 'fmin_tnc':
                kwargs['maxfun'] = 100
            else:
                kwargs['maxiter'] = 5
        else:
            def routine(*a, **kw):
                kw['method'] = method
                return optimize.minimize(*a, **kw)

            if method == 'tnc':
                kwargs['options'] = dict(maxfun=100)
            else:
                kwargs['options'] = dict(maxiter=5)

        if method in ('fmin_ncg',):
            kwargs['fprime'] = jac
        elif method in ('newton-cg',):
            kwargs['jac'] = jac
        elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg',
                        'trust-constr'):
            kwargs['jac'] = jac
            kwargs['hess'] = hess

        # Run with callback
        results = []

        def callback(x, *args, **kwargs):
            results.append((x, np.copy(x)))

        routine(func, x0, callback=callback, **kwargs)

        # Check returned arrays coincide with their copies
        # and have no memory overlap
        assert_(len(results) > 2)
        assert_(all(np.all(x == y) for x, y in results))
        assert_(not any(np.may_share_memory(x[0], y[0])
                        for x, y in itertools.combinations(results, 2)))


    @pytest.mark.parametrize('method', ['trust-constr',])
    def test_respect_maxiter(self, method):
        # Check that the number of iterations equals max_iter, assuming
        # convergence doesn't establish before
        MAXITER = 4

        x0 = np.zeros(10)

        sf = ScalarFunction(optimize.rosen, x0, (), optimize.rosen_der,
                            optimize.rosen_hess, None, None)

        # Set options
        kwargs = {'method': method, 'options': dict(maxiter=MAXITER)}

        if method in ('Newton-CG',):
            kwargs['jac'] = sf.grad
        elif method in ('trust-krylov', 'trust-exact', 'trust-ncg', 'dogleg',
                        'trust-constr'):
            kwargs['jac'] = sf.grad
            kwargs['hess'] = sf.hess

        sol = optimize.minimize(sf.fun, x0, **kwargs)
        assert sol.nit == MAXITER
        assert sol.nfev >= sf.nfev
        if hasattr(sol, 'njev'):
            assert sol.njev >= sf.ngev

        # method specific tests
        if method == 'SLSQP':
            assert sol.status == 9  # Iteration limit reached

    def test_respect_maxiter_trust_constr_ineq_constraints(self):
        # special case of minimization with trust-constr and inequality
        # constraints to check maxiter limit is obeyed when using internal
        # method 'tr_interior_point'
        MAXITER = 4
        f = optimize.rosen
        jac = optimize.rosen_der
        hess = optimize.rosen_hess

        fun = lambda x: np.array([0.2 * x[0] - 0.4 * x[1] - 0.33 * x[2]])
        cons = ({'type': 'ineq',
                 'fun': fun},)

        x0 = np.zeros(10)
        sol = optimize.minimize(f, x0, constraints=cons, jac=jac, hess=hess,
                                method='trust-constr',
                                options=dict(maxiter=MAXITER))
        assert sol.nit == MAXITER

    def test_minimize_automethod(self):
        def f(x):
            return x**2

        def cons(x):
            return x - 2

        x0 = np.array([10.])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.*")

            sol_0 = optimize.minimize(f, x0)
            sol_1 = optimize.minimize(f, x0, constraints=[{'type': 'ineq',
                                                        'fun': cons}])
            sol_2 = optimize.minimize(f, x0, bounds=[(5, 10)])
            sol_3 = optimize.minimize(f, x0,
                                    constraints=[{'type': 'ineq', 'fun': cons}],
                                    bounds=[(5, 10)])
            sol_4 = optimize.minimize(f, x0,
                                    constraints=[{'type': 'ineq', 'fun': cons}],
                                    bounds=[(1, 10)])

        for sol in [sol_0, sol_1, sol_2, sol_3, sol_4]:
            assert_(sol.success)
        assert_allclose(sol_0.x, 0, atol=1e-7)
        # These last four were opened up from atol=1e-7 for scipy 1.6.0
        # likely do to a different algorithm being used instead of trust-constr
        # only methods SLSQP, L-BFGS-B, and BFGS are used by minimize
        # automatically. Here, we're forced to used trust-constr for 
        # everything since that's all that is available.
        assert_allclose(sol_1.x, 2, atol=1e-4, rtol=1e-4)
        assert_allclose(sol_2.x, 5, atol=1e-4, rtol=1e-4)
        assert_allclose(sol_3.x, 5, atol=1e-4, rtol=1e-4)
        assert_allclose(sol_4.x, 2, atol=1e-4, rtol=1e-4)


    @pytest.mark.parametrize('method', ['trust-constr',])
    def test_nan_values(self, method):
        # Check nan values result to failed exit status
        np.random.seed(1234)

        count = [0]

        def func(x):
            return np.nan

        def func2(x):
            count[0] += 1
            if count[0] > 2:
                return np.nan
            else:
                return np.random.rand()

        def grad(x):
            return np.array([1.0])

        def hess(x):
            return np.array([[1.0]])

        x0 = np.array([1.0])

        needs_grad = method in ('newton-cg', 'trust-krylov', 'trust-exact',
                                'trust-ncg', 'dogleg')
        needs_hess = method in ('trust-krylov', 'trust-exact', 'trust-ncg',
                                'dogleg')

        funcs = [func, func2]
        grads = [grad] if needs_grad else [grad, None]
        hesss = [hess] if needs_hess else [hess, None]

        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            sup.filter(UserWarning, "delta_grad == 0.*")
            sup.filter(RuntimeWarning, ".*does not use Hessian.*")
            sup.filter(RuntimeWarning, ".*does not use gradient.*")

            for f, g, h in itertools.product(funcs, grads, hesss):
                count = [0]
                sol = optimize.minimize(f, x0, jac=g, hess=h, method=method,
                                        options=dict(maxiter=20))
                assert_equal(sol.success, False)

    @pytest.mark.parametrize('method', ['trust-constr',])
    def test_duplicate_evaluations(self, method):
        # check that there are no duplicate evaluations for any methods
        jac = hess = None
        if method in ('newton-cg', 'trust-krylov', 'trust-exact',
                      'trust-ncg', 'dogleg'):
            jac = self.grad
        if method in ('trust-krylov', 'trust-exact', 'trust-ncg',
                      'dogleg'):
            hess = self.hess

        with np.errstate(invalid='ignore'), suppress_warnings() as sup:
            # for trust-constr
            sup.filter(UserWarning, "delta_grad == 0.*")
            optimize.minimize(self.func, self.startparams,
                              method=method, jac=jac, hess=hess)

        for i in range(1, len(self.trace)):
            if np.array_equal(self.trace[i - 1], self.trace[i]):
                raise RuntimeError(
                    "Duplicate evaluations made by {}".format(method))


class TestRosen(object):

    def test_hess(self):
        # Compare rosen_hess(x) times p with rosen_hess_prod(x,p). See gh-1775.
        x = np.array([3, 4, 5])
        p = np.array([2, 2, 2])
        hp = optimize.rosen_hess_prod(x, p)
        dothp = np.dot(optimize.rosen_hess(x), p)
        assert_equal(hp, dothp)


def himmelblau(p):
    """
    R^2 -> R^1 test function for optimization. The function has four local
    minima where himmelblau(xopt) == 0.
    """
    x, y = p
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b


def himmelblau_grad(p):
    x, y = p
    return np.array([4*x**3 + 4*x*y - 42*x + 2*y**2 - 14,
                     2*x**2 + 4*x*y + 4*y**3 - 26*y - 22])


def himmelblau_hess(p):
    x, y = p
    return np.array([[12*x**2 + 4*y - 42, 4*x + 4*y],
                     [4*x + 4*y, 4*x + 12*y**2 - 26]])


himmelblau_x0 = [-0.27, -0.9]
himmelblau_xopt = [3, 2]
himmelblau_min = 0.0


class TestOptimizeResultAttributes(object):
    # Test that all minimizers return an OptimizeResult containing
    # all the OptimizeResult attributes
    def setup_method(self):
        self.x0 = [5, 5]
        self.func = optimize.rosen
        self.jac = optimize.rosen_der
        self.hess = optimize.rosen_hess
        self.hessp = optimize.rosen_hess_prod
        self.bounds = [(0., 10.), (0., 10.)]

    def test_attributes_present(self):
        attributes = ['nit', 'nfev', 'x', 'success', 'status', 'fun',
                      'message']
        skip = {'cobyla': ['nit']}
        for method in MINIMIZE_METHODS:
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning,
                           ("Method .+ does not use (gradient|Hessian.*)"
                            " information"))
                res = optimize.minimize(self.func, self.x0, method=method,
                                        jac=self.jac, hess=self.hess,
                                        hessp=self.hessp)
            for attribute in attributes:
                if method in skip and attribute in skip[method]:
                    continue

                assert hasattr(res, attribute)
                assert_(attribute in dir(res))

            # gh13001, OptimizeResult.message should be a str
            assert isinstance(res.message, str)


def f1(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)


def f2(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-g*np.exp(-((x-h)**2 + (y-i)**2) / scale))


def f3(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-j*np.exp(-((x-k)**2 + (y-l)**2) / scale))


def brute_func(z, *params):
    return f1(z, *params) + f2(z, *params) + f3(z, *params)

   
class TestIterationLimits(object):
    # Tests that optimisation does not give up before trying requested
    # number of iterations or evaluations. And that it does not succeed
    # by exceeding the limits.
    def setup_method(self):
        self.funcalls = 0

    def slow_func(self, v):
        self.funcalls += 1
        r, t = np.sqrt(v[0]**2+v[1]**2), np.arctan2(v[0], v[1])
        return np.sin(r*20 + t)+r*0.5

    def check_limits(self, method, default_iters):
        for start_v in [[0.1, 0.1], [1, 1], [2, 2]]:
            for mfev in [50, 500, 5000]:
                self.funcalls = 0
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxfev": mfev})
                assert_(self.funcalls == res["nfev"])
                if res["success"]:
                    assert_(res["nfev"] < mfev)
                else:
                    assert_(res["nfev"] >= mfev)
            for mit in [50, 500, 5000]:
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxiter": mit})
                if res["success"]:
                    assert_(res["nit"] <= mit)
                else:
                    assert_(res["nit"] >= mit)
            for mfev, mit in [[50, 50], [5000, 5000], [5000, np.inf]]:
                self.funcalls = 0
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxiter": mit,
                                                 "maxfev": mfev})
                assert_(self.funcalls == res["nfev"])
                if res["success"]:
                    assert_(res["nfev"] < mfev and res["nit"] <= mit)
                else:
                    assert_(res["nfev"] >= mfev or res["nit"] >= mit)
            for mfev, mit in [[np.inf, None], [None, np.inf]]:
                self.funcalls = 0
                res = optimize.minimize(self.slow_func, start_v,
                                        method=method,
                                        options={"maxiter": mit,
                                                 "maxfev": mfev})
                assert_(self.funcalls == res["nfev"])
                if res["success"]:
                    if mfev is None:
                        assert_(res["nfev"] < default_iters*2)
                    else:
                        assert_(res["nit"] <= default_iters*2)
                else:
                    assert_(res["nfev"] >= default_iters*2 or
                        res["nit"] >= default_iters*2)


class FunctionWithGradient(object):
    def __init__(self):
        self.number_of_calls = 0

    def __call__(self, x):
        self.number_of_calls += 1
        return np.sum(x**2), 2 * x


@pytest.fixture
def function_with_gradient():
    return FunctionWithGradient()


def test_memoize_jac_function_before_gradient(function_with_gradient):
    memoized_function = MemoizeJac(function_with_gradient)

    x0 = np.array([1.0, 2.0])
    assert_allclose(memoized_function(x0), 5.0)
    assert function_with_gradient.number_of_calls == 1

    assert_allclose(memoized_function.derivative(x0), 2 * x0)
    assert function_with_gradient.number_of_calls == 1, \
        "function is not recomputed " \
        "if gradient is requested after function value"

    assert_allclose(
        memoized_function(2 * x0), 20.0,
        err_msg="different input triggers new computation")
    assert function_with_gradient.number_of_calls == 2, \
        "different input triggers new computation"


def test_memoize_jac_gradient_before_function(function_with_gradient):
    memoized_function = MemoizeJac(function_with_gradient)

    x0 = np.array([1.0, 2.0])
    assert_allclose(memoized_function.derivative(x0), 2 * x0)
    assert function_with_gradient.number_of_calls == 1

    assert_allclose(memoized_function(x0), 5.0)
    assert function_with_gradient.number_of_calls == 1, \
        "function is not recomputed " \
        "if function value is requested after gradient"

    assert_allclose(
        memoized_function.derivative(2 * x0), 4 * x0,
        err_msg="different input triggers new computation")
    assert function_with_gradient.number_of_calls == 2, \
        "different input triggers new computation"


def test_gh12696():
    # Test that optimize doesn't throw warning gh-12696
    with assert_no_warnings():
        optimize.fminbound(
            lambda x: np.array([x**2]), -np.pi, np.pi, disp=False)

        
