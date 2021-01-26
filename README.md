# trust-constr

trust-constr optimization algorithm from the SciPy project that was originally implemented by [Antonio Horta Ribeiro](https://github.com/antonior92). This is a version of the trust-constr algorithm that does not depend on the rest of SciPy. The only dependency is NumPy. The goal is to have a version of the trust-constr algorithm that can run within the Pyodide environment.

# Examples Using trust-constr

Since the trust-constr algorithm was extracted from the `scipy.optimize` library, it uses the [same interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize) as `scipy.optimize.minimize`. The main different is that everything is imported from `trust_constr` rather than from `scipy.optimize`. The other difference is that the only optimization method available is 'trust-const'. The examples below show how to use trust-constr with a variety of different types of constraints.


```python
import numpy as np
from trust_constr import minimize, NonlinearConstraint, LinearConstraint, Bounds, check_grad
```

## Example 1: Nonlinear Inequality Constraint with Variable Bounds
Example 15.1 from [1]

Solve:
<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/28fdfb7a1d96af9198fc716e27c095ae.svg?invert_in_darkmode" align=middle width=276.69084134999997pt height=42.80407395pt/></p>
Subject to:
<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/fdf4422614269144ef7f80731ca33e4a.svg?invert_in_darkmode" align=middle width=159.27210584999997pt height=32.990165999999995pt/></p>
<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/565e576a78a7b581fe3c9ecf27b229d3.svg?invert_in_darkmode" align=middle width=39.5318286pt height=12.82874835pt/></p>
<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/224ba9af64deada0cfd312a4fea665df.svg?invert_in_darkmode" align=middle width=38.78604675pt height=13.789957499999998pt/></p>

Solution: <p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/babd0d08a2217bc8ce1f60222e098e93.svg?invert_in_darkmode" align=middle width=155.03058119999997pt height=16.438356pt/></p>


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

<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/2fee2fcbc6493f2dfed3044b1532bbbe.svg?invert_in_darkmode" align=middle width=82.19939475pt height=26.303252249999996pt/></p>

Subject to:

<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/3bedd9434e5c8c145c46787c6cd9af74.svg?invert_in_darkmode" align=middle width=124.1169798pt height=18.312383099999998pt/></p>

Solution:
<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/6e7b6b6a241ed9fb63a931886262f6a4.svg?invert_in_darkmode" align=middle width=96.58287705pt height=16.438356pt/></p>




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

<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/224a59c0b33c11009e0b5de29effd0e4.svg?invert_in_darkmode" align=middle width=202.6216566pt height=29.654885699999998pt/></p>

Subject to:

<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/278cd1acde20484153c1e2b358a714e0.svg?invert_in_darkmode" align=middle width=76.49143425pt height=13.789957499999998pt/></p>


Solution:

<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/313de0b35c1215a65b80c1a3a53e9d32.svg?invert_in_darkmode" align=middle width=171.4689999pt height=16.438356pt/></p>


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

<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/68eac75e5d74e3f0cbaf01f89339a552.svg?invert_in_darkmode" align=middle width=308.680977pt height=49.9887465pt/></p>

Solution for <img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/9aad22a1f10eb2f672ffc52c46eac498.svg?invert_in_darkmode" align=middle width=45.13680929999999pt height=22.465723500000017pt/>:

<p align="center"><img src="https://raw.githubusercontent.com/mgreminger/trust-constr/1849afa5af222a69e25679b5336a48e6b58a2853/svgs/2320eb38cb2a4f6c8ab1fe49826b4749.svg?invert_in_darkmode" align=middle width=83.9495745pt height=16.438356pt/></p>



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
