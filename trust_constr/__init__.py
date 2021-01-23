from ._minimize import minimize
from ._constraints import NonlinearConstraint, LinearConstraint, Bounds
from .optimize import (check_grad, MemoizeJac, show_options, OptimizeResult, 
                       rosen, rosen_hess, rosen_hess_prod, rosen_der, fminbound,
                       OptimizeWarning)
from ._hessian_update_strategy import BFGS, SR1
# from ._differentiable_functions import ScalarFunction
