from itertools import combinations_with_replacement
from functools import partial

import numpy as np
from numpy.linalg import pinv
from numpy.random import uniform
from numpy.testing import assert_allclose, suppress_warnings

import pytest

from trust_constr import minimize, NonlinearConstraint, Bounds, check_grad

from response_surface_data import bottle_data_set, panel_data_set, brake_data_set


def get_response_surface(inputs, outputs):

    response_surfaces = []

    terms = list(combinations_with_replacement(range(inputs.shape[1]), 1))
    terms.extend(combinations_with_replacement(range(inputs.shape[1]), 2))

    for current_output in range(outputs.shape[1]):
        A = np.ones((inputs.shape[0], len(terms) + 1))
        rhs = np.zeros(inputs.shape[0])

        for i, row in enumerate(outputs):
            rhs[i] = row[current_output]
            for j, term in enumerate(terms):
                A[i, j + 1] = inputs[i].take(term).prod()

        response_surfaces.append(pinv(A) @ rhs)

    # first coefficient is the constant coefficient
    return terms, response_surfaces


def evaluate_response_surface(
    term_indices_list, rs_coefficients, x, factor=1.0, offset=0.0
):
    terms = np.ones(len(term_indices_list) + 1)

    for i, term_indices in enumerate(term_indices_list):
        terms[i + 1] = x.take(term_indices).prod()

    return factor * (terms.dot(rs_coefficients) + offset)


def evaluate_response_surface_grad(
    term_indices_list, rs_coefficients, x, factor=1.0, offset=0.0
):

    grad_terms = np.zeros((len(x), len(term_indices_list) + 1))

    for i, term_indices in enumerate(term_indices_list):
        for j in range(len(x)):
            if len(term_indices) == 1 and term_indices[0] == j:
                grad_terms[j, i + 1] = 1.0
            if len(term_indices) == 2:
                if term_indices[0] == j and term_indices[1] == j:
                    grad_terms[j, i + 1] = 2.0 * x[j]
                elif term_indices[0] == j:
                    grad_terms[j, i + 1] = x[term_indices[1]]
                elif term_indices[1] == j:
                    grad_terms[j, i + 1] = x[term_indices[0]]

    return factor * grad_terms.dot(rs_coefficients)


@pytest.mark.parametrize(
    "data,factorization_method,rtol",
    [
        (bottle_data_set, "SVDFactorization", 1e-5),
        (bottle_data_set, "AugmentedSystem", 1e-5),
        (panel_data_set, "SVDFactorization", 1e-5),
        (panel_data_set, "AugmentedSystem", 1e-5),
        (brake_data_set, "SVDFactorization", 2e-4),
        (brake_data_set, "AugmentedSystem", 2e-4),
    ],
)
def test_generate_pareto_data(data, factorization_method, rtol):
    inputs = np.array(data["inputs"])
    outputs = np.array(data["outputs"])
    y_axis_index = data["y_axis_index"]
    y_axis_goal = data["y_axis_goal"]
    x_axis_index = data["x_axis_index"]
    x_axis_goal = data["x_axis_goal"]
    bounds = data["bounds"]
    output_targets = data["output_targets"]
    baseline_pareto_points = np.array(data["pareto_points"])
    baseline_total_iterations = data["total_iterations"]

    terms, response_surfaces = get_response_surface(inputs, outputs)

    # check analytical gradients against numerical gradients
    for i in range(len(response_surfaces)):
        test_points = uniform(
            low=[pair[0] for pair in bounds],
            high=[pair[1] for pair in bounds],
            size=(10, len(bounds)),
        )
        for test_point in test_points:
            difference = check_grad(
                partial(evaluate_response_surface, terms, response_surfaces[i]),
                partial(evaluate_response_surface_grad, terms, response_surfaces[i]),
                test_point,
            )
            assert (
                difference
                / evaluate_response_surface(terms, response_surfaces[i], test_point)
                < 1e-4
            )

    # add equality constraints for any outputs with targets
    constraints = []
    for index, target in enumerate(output_targets):
        if target:
            constraints.append(
                NonlinearConstraint(
                    partial(evaluate_response_surface, terms, response_surfaces[index]),
                    target,
                    target,
                    jac=partial(
                        evaluate_response_surface_grad, terms, response_surfaces[index]
                    ),
                )
            )

    def objective_func(x, index, sign=1.0):
        return evaluate_response_surface(
            terms, response_surfaces[index], x, factor=sign
        )

    def objective_func_grad(x, index, sign=1.0):
        return evaluate_response_surface_grad(
            terms, response_surfaces[index], x, factor=sign
        )

    x0 = np.array([(pair[0] + pair[1]) / 2 for pair in bounds])

    # get the x-axis range for the pareto optimization
    with suppress_warnings() as sup:
        sup.filter(UserWarning, "delta_grad == 0.*")
        # purposely leaving out jac here, will test below
        res = minimize(
            objective_func,
            x0,
            args=(x_axis_index, -1.0),
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, "factorization_method": factorization_method},
        )
        x_max = -res.fun

        # x_min_starting_point = X[y[:,x_axis_index].argmin(),:]
        res = minimize(
            objective_func,
            x0,
            args=(x_axis_index, 1.0),
            bounds=bounds,
            constraints=constraints,
            method="trust-constr",
            options={"disp": False, "factorization_method": factorization_method},
        )
    x_min = res.fun

    pareto_points = np.linspace(x_min, x_max, num=baseline_pareto_points.shape[0])

    # find the actual pareto points
    pareto_input_values = []

    y_axis_sign = 1 if y_axis_goal == "min" else -1

    total_iterations = 0
    for x_value in pareto_points:
        if x_axis_goal == "min":
            limits = (-np.inf, x_value)
        else:
            limits = (x_value, np.inf)

        current_constraint = NonlinearConstraint(
            partial(evaluate_response_surface, terms, response_surfaces[x_axis_index]),
            *limits,
            jac=partial(
                evaluate_response_surface_grad, terms, response_surfaces[x_axis_index]
            )
        )

        res = minimize(
            objective_func,
            x0,
            args=(y_axis_index, y_axis_sign),
            bounds=bounds,
            constraints=constraints + [current_constraint,],
            jac=objective_func_grad,
            method="trust-constr",
            options={"disp": False, "factorization_method": factorization_method},
        )

        total_iterations += res.nit

        pareto_input_values.append(res.x)

    pareto_output_values = []
    for current_input in pareto_input_values:
        current_output = []
        for i in range(outputs.shape[1]):
            current_output.append(
                evaluate_response_surface(terms, response_surfaces[i], current_input)
            )

        pareto_output_values.append(current_output)

    pareto_output_values = np.array(pareto_output_values)

    assert total_iterations < baseline_total_iterations + 10
    assert_allclose(pareto_output_values, baseline_pareto_points, rtol=rtol)


def rosenbrock_function(x):
    result = 0

    for i in range(len(x) - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

    return result


def test_no_constraints():
    res = minimize(
        rosenbrock_function, np.array([0.1, -0.5, -5.0]), options={"disp": False}
    )

    assert res.niter < 61  # SciPy version 1.5.0 needs 56 iterations
    assert_allclose(res.x, [1, 1, 1], rtol=1e-4)


def test_bounds_class():
    res = minimize(
        rosenbrock_function,
        np.array([0.1, -0.5, -5.0]),
        bounds=Bounds([-10, -10, -np.inf], [10, np.inf, np.inf]),
        method="trust-constr",
        options={"disp": False},
    )

    assert res.niter < 110  # SciPy version 1.5.0 needs 105 iterations
    assert_allclose(res.x, [1, 1, 1], rtol=1e-4)
