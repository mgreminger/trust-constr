"""Basic linear factorizations needed by the solver."""

from ..interface import LinearOperator
import numpy as np
from warnings import warn

__all__ = [
    'orthogonality',
    'projections',
]


def orthogonality(A, g):
    """Measure orthogonality between a vector and the null space of a matrix.

    Compute a measure of orthogonality between the null space
    of the (possibly sparse) matrix ``A`` and a given vector ``g``.

    The formula is a simplified (and cheaper) version of formula (3.13)
    from [1]_.
    ``orth =  norm(A g, ord=2)/(norm(A, ord='fro')*norm(g, ord=2))``.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
           "On the solution of equality constrained quadratic
            programming problems arising in optimization."
            SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    # Compute vector norms
    norm_g = np.linalg.norm(g)
    # Compute Froebnius norm of the matrix A
    norm_A = np.linalg.norm(A, ord='fro')

    # Check if norms are zero
    if norm_g == 0 or norm_A == 0:
        return 0

    norm_A_g = np.linalg.norm(A.dot(g))
    # Orthogonality measure
    orth = norm_A_g / (norm_A*norm_g)
    return orth


def augmented_system_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A - ``AugmentedSystem``."""
    # Form augmented system
    if A.size != 0:
        K = np.block([[np.eye(n), A.T], [A, np.zeros((m,m))]])
    else:
        K = np.eye(n)
    # LU factorization

    # z = x - A.T inv(A A.T) A x
    # is computed solving the extended system:
    # [I A.T] * [ z ] = [x]
    # [A  O ]   [aux]   [0]
    def null_space(x):
        # v = [x]
        #     [0]
        v = np.hstack([x, np.zeros(m)])
        # lu_sol = [ z ]
        #          [aux]
        try:
            lu_sol = np.linalg.solve(K, v)
        except np.linalg.LinAlgError:
            warn("Jacobian matrix singular, switching to least squares solve.")
            lu_sol = np.linalg.lstsq(K, v)

        z = lu_sol[:n]

        # Iterative refinement to improve roundoff
        # errors described in [2]_, algorithm 5.2.
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            # new_v = [x] - [I A.T] * [ z ]
            #         [0]   [A  O ]   [aux]
            new_v = v - K.dot(lu_sol)
            # [I A.T] * [delta  z ] = new_v
            # [A  O ]   [delta aux]
            try:
                lu_update = np.linalg.solve(K, new_v)
            except np.linalg.LinAlgError:
                lu_update = np.linalg.lstsq(K, new_v)
            #  [ z ] += [delta  z ]
            #  [aux]    [delta aux]
            lu_sol += lu_update
            z = lu_sol[:n]
            k += 1

        # return z = x - A.T inv(A A.T) A x
        return z

    # z = inv(A A.T) A x
    # is computed solving the extended system:
    # [I A.T] * [aux] = [x]
    # [A  O ]   [ z ]   [0]
    def least_squares(x):
        # v = [x]
        #     [0]
        v = np.hstack([x, np.zeros(m)])
        # lu_sol = [aux]
        #          [ z ]
        try:
            lu_sol = np.linalg.solve(K, v)
        except np.linalg.LinAlgError:
            warn("Jacobian matrix singular, switching to least squares solve.")
            lu_sol = np.linalg.lstsq(K, v)
        # return z = inv(A A.T) A x
        return lu_sol[n:m+n]

    # z = A.T inv(A A.T) x
    # is computed solving the extended system:
    # [I A.T] * [ z ] = [0]
    # [A  O ]   [aux]   [x]
    def row_space(x):
        # v = [0]
        #     [x]
        v = np.hstack([np.zeros(n), x])
        # lu_sol = [ z ]
        #          [aux]
        try:
            lu_sol = np.linalg.solve(K, v)
        except np.linalg.LinAlgError:
            warn("Jacobian matrix singular, switching to least squares solve.")
            lu_sol = np.linalg.lstsq(K, v)

        
        # return z = A.T inv(A A.T) x
        return lu_sol[:n]

    return null_space, least_squares, row_space


def svd_factorization_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A using ``SVDFactorization`` approach.
    """
    # SVD Factorization
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Remove dimensions related with very small singular values
    U = U[:, s > tol]
    Vt = Vt[s > tol, :]
    s = s[s > tol]

    # z = x - A.T inv(A A.T) A x
    def null_space(x):
        # v = U 1/s V.T x = inv(A A.T) A x
        aux1 = Vt.dot(x)
        aux2 = 1/s*aux1
        v = U.dot(aux2)
        z = x - A.T.dot(v)

        # Iterative refinement to improve roundoff
        # errors described in [2]_, algorithm 5.1.
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            # v = U 1/s V.T x = inv(A A.T) A x
            aux1 = Vt.dot(z)
            aux2 = 1/s*aux1
            v = U.dot(aux2)
            # z_next = z - A.T v
            z = z - A.T.dot(v)
            k += 1

        return z

    # z = inv(A A.T) A x
    def least_squares(x):
        # z = U 1/s V.T x = inv(A A.T) A x
        aux1 = Vt.dot(x)
        aux2 = 1/s*aux1
        z = U.dot(aux2)
        return z

    # z = A.T inv(A A.T) x
    def row_space(x):
        # z = V 1/s U.T x
        aux1 = U.T.dot(x)
        aux2 = 1/s*aux1
        z = Vt.T.dot(aux2)
        return z

    return null_space, least_squares, row_space


def projections(A, method=None, orth_tol=1e-12, max_refin=3, tol=1e-15):
    """Return three linear operators related with a given matrix A.

    Parameters
    ----------
    A : matrix (or ndarray), shape (m, n)
        Matrix ``A`` used in the projection.
    method : string, optional
        Method used for compute the given linear
        operators. Should be one of:

            - 'AugmentedSystem': The operators
               will be computed using the
               so-called augmented system approach
               explained in [1]_. Exclusive
               for sparse matrices. Required for
               unconstrained problems.
            - 'SVDFactorization': Compute projections
               using SVD factorization. Exclusive for
               dense matrices.

    orth_tol : float, optional
        Tolerance for iterative refinements.
    max_refin : int, optional
        Maximum number of iterative refinements.
    tol : float, optional
        Tolerance for singular values.

    Returns
    -------
    Z : LinearOperator, shape (n, n)
        Null-space operator. For a given vector ``x``,
        the null space operator is equivalent to apply
        a projection matrix ``P = I - A.T inv(A A.T) A``
        to the vector. It can be shown that this is
        equivalent to project ``x`` into the null space
        of A.
    LS : LinearOperator, shape (m, n)
        Least-squares operator. For a given vector ``x``,
        the least-squares operator is equivalent to apply a
        pseudoinverse matrix ``pinv(A.T) = inv(A A.T) A``
        to the vector. It can be shown that this vector
        ``pinv(A.T) x`` is the least_square solution to
        ``A.T y = x``.
    Y : LinearOperator, shape (n, m)
        Row-space operator. For a given vector ``x``,
        the row-space operator is equivalent to apply a
        projection matrix ``Q = A.T inv(A A.T)``
        to the vector.  It can be shown that this
        vector ``y = Q x``  the minimum norm solution
        of ``A y = x``.

    Notes
    -----
    Uses iterative refinements described in [1]
    during the computation of ``Z`` in order to
    cope with the possibility of large roundoff errors.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
        "On the solution of equality constrained quadratic
        programming problems arising in optimization."
        SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    m, n = np.shape(A)

    # The factorization of an empty matrix
    # only works with the AugmentedSystem method
    if m*n == 0:
        if method == None:
            method = "AugmentedSystem"
        elif method != "AugmentedSystem":
            raise ValueError(f"factorization_method {method} not supported for "
                              "unconstrained problems.")

    if method is None:
        method = "SVDFactorization" # default for constrained problems

    if method == "AugmentedSystem":
        null_space, least_squares, row_space \
            = augmented_system_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == "SVDFactorization":
        null_space, least_squares, row_space \
            = svd_factorization_projections(A, m, n, orth_tol, max_refin, tol)
    else:
        raise ValueError("Unknown factorization_method:", method)

    Z = LinearOperator((n, n), null_space)
    LS = LinearOperator((m, n), least_squares)
    Y = LinearOperator((n, m), row_space)

    return Z, LS, Y
