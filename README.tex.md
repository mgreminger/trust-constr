# trust-constr

trust-constr optimization algorithm from the SciPy project that was originally implemented by [Antonio Horta Ribeiro](https://github.com/antonior92). This is a version of the trust-constr algorithm that does not depend on the rest of SciPy. The only dependency is NumPy. The goal is to have a version of the trust-constr algorithm that can run within the Pyodide environment.

# Installation

`pip install trust-constr`

# Examples Using trust-constr

Since the trust-constr algorithm was extracted from the `scipy.optimize` library, it uses the [same interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize) as `scipy.optimize.minimize`. The main different is that everything is imported from `trust_constr` rather than from `scipy.optimize`. The other difference is that the only optimization method available is 'trust-const'. The examples below show how to use trust-constr with a variety of different types of constraints.


```python
import numpy as np
from trust_constr import minimize, NonlinearConstraint, LinearConstraint, Bounds, check_grad
```

## Example 1: Nonlinear Inequality Constraint with Variable Bounds
Example 15.1 from [1]

Solve:
$$\min_{x,y} f(x,y)=\frac{1}{2}(x-2)^2+\frac{1}{2}\left(y-\frac{1}{2}\right)^2$$
Subject to:
$$(x+1)^{-1}-y-\frac{1}{4}\ge0$$
$$x\ge0$$
$$y\ge0$$

Solution: $$(x,y) = (1.953, 0.089)$$


First solve without defining gradient (finite difference gradient will be used):


```python
def objective(x):
    return 0.5*(x[0]-2)**2+0.5*(x[1]-0.5)**2

def ineq_constraint(x):
    return 1/(x[0]+1)-x[1]-0.25

# Use np.inf of -np.inf to define a single sided constraint
# If there are more than one constraint, that constraints will 
# be a list containing all of the constraints
constraints = NonlinearConstraint(ineq_constraint, 0, np.inf)

# set bounds on the variables
# only a lower bound is needed so the upper bound for both variables is set to np.inf
bounds = Bounds([0,0], [np.inf, np.inf])

# define starting point for optimization
x0 = np.array([5.0, 1.0])

res = minimize(objective, x0, bounds=bounds, constraints=constraints)

print("Solution =", res.x)
print(f"Obtained using {res.nfev} objective function evaluations.")
```

    Solution = [1.95282327 0.08865882]
    Obtained using 42 objective function evaluations.


Now define the gradient for objective and constraint and check gradients:


```python
def objective_gradient(x):
    return np.array([(x[0]-2), (x[1]-0.5)])

def ineq_gradient(x):
    return np.array([-1/((x[0]+1)**2), -1])

# check analytical gradients against finite difference gradient
# an incorrect analytical gradient is a common cause for lack of convergence to a true minimum
for x in np.random.uniform(low=[0,0], high=[10,10], size=(5,2)):
    print("objective difference: ", check_grad(objective, objective_gradient, x))
    print("constraint difference:", check_grad(ineq_constraint, ineq_gradient, x))


```

    objective difference:  7.24810320719611e-08
    constraint difference: 2.1805555505335916e-08
    objective difference:  1.5409355031965243e-08
    constraint difference: 1.8387489794657874e-10
    objective difference:  8.16340974645582e-08
    constraint difference: 2.2211865402521624e-08
    objective difference:  1.51975085661403e-07
    constraint difference: 5.070987015715067e-10
    objective difference:  1.7113557964841567e-07
    constraint difference: 4.981334539820581e-08


Finally, minimize using the gradient functions that were just test:


```python
constraints = NonlinearConstraint(ineq_constraint, 0, np.inf, jac=ineq_gradient)

res = minimize(objective, x0, jac=objective_gradient, bounds=bounds, constraints=constraints)

print("Solution =", res.x)
print(f"Obtained using {res.nfev} objective function evaluations.")
```

    Solution = [1.95282328 0.08865881]
    Obtained using 14 objective function evaluations.


## Example 2: Nonlinear Equality Constraint
Example 15.2 from [1]

Solve:

$$\min_{x,y} x^2+y^2$$

Subject to:

$$(x-1)^3 - y^2 = 0$$

Solution:
$$(x,y)=(1,0)$$




```python
objective2 = lambda x: x[0]**2 + x[1]**2
objective2_gradient = lambda x: np.array([2*x[0], 2*x[1]])

eq_constraint = lambda x: (x[0]-1)**3 - x[1]**2
eq_gradient = lambda x: np.array([3*(x[0]-1)**2, -2*x[1]]) 

# Make the upper and lower bound both zero to define an equality constraint
constraints = NonlinearConstraint(eq_constraint, 0, 0, jac=eq_gradient) 

x0 = np.array([5, 2])

res = minimize(objective2, x0, jac=objective2_gradient, constraints=constraints)

print("Solution =", res.x)
print(f"Obtained using {res.nfev} objective function evaluations.")
```

    Solution = [9.99966899e-01 3.36074169e-09]
    Obtained using 181 objective function evaluations.


## Example 3: Linear Constraint
Example problem from [2]

Solve:

$$\min_{x,y} 100\left(y-x^2\right)^2+\left(1-x\right)^2$$

Subject to:

$$x + 2y \le 1$$


Solution:

$$(x,y)=(0.5022, 0.2489)$$


```python
objective3 = lambda x: 100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2

objective3_gradient = lambda x: np.array([-400*(x[1]-x[0]**2)*x[0]-2*(1-x[0]),
                                          200*(x[1]-x[0]**2)])

# define the linear constraint
A = np.array([[1,2]])
constraints = LinearConstraint(A, [-np.inf], [1])

x0 = np.array([-1, 2])

res = minimize(objective3, x0, jac=objective3_gradient, constraints=constraints)

print("Solution =", res.x)
print(f"Obtained using {res.nfev} objective function evaluations.")
```

    Solution = [0.50220246 0.24889838]
    Obtained using 45 objective function evaluations.


## Example 4: Unconstrained Optimization
Example problem from [3]

Solve:

$$\min_{\mathbf{x}} \sum^{N/2}_{i=1}\left[100\left(x^2_{2i-1}-x_{2i}\right)^2+\left(x_{2i-1}-1\right)^2\right]$$

Solution for $N=3$:

$$\mathbf{x} = (1,1,1)$$



```python
def rosenbrock_function(x):
    result = 0

    for i in range(len(x) - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

    return result

x0 = np.array([0.1, -0.5, -5.0])

res = minimize(rosenbrock_function, x0)

print("Solution =", res.x)
print(f"Obtained using {res.nfev} objective function evaluations.")
```

    Solution = [0.99999729 0.99999458 0.99998915]
    Obtained using 224 objective function evaluations.


## References
[1] Nocedal, Jorge, and Stephen J. Wright. *Numerical Optimization*. 2nd ed. Springer Series in Operations Research. New York: Springer, 2006.

[2] https://www.mathworks.com/help/optim/ug/fmincon.html

[3] https://en.wikipedia.org/wiki/Rosenbrock_function
