import numpy as np
from trust_constr._trustregion_constr.projections \
    import projections, orthogonality
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_equal, assert_allclose)

available_dense_methods = ('SVDFactorization', 'AugmentedSystem')


class TestProjections(TestCase):
    def test_nullspace_and_least_squares_dense(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        At = A.T
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8])

        for method in available_dense_methods:
            Z, LS, _ = projections(A, method)
            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)
                assert_array_almost_equal(A.dot(x), 0)
                # Test orthogonality
                assert_array_almost_equal(orthogonality(A, x), 0)
                # Test if x is the least square solution
                x = LS.matvec(z)
                x2 = np.linalg.lstsq(At, z, rcond=None)[0]
                assert_array_almost_equal(x, x2)

    def test_iterative_refinements_dense(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1, 0, 0, 0, 0, 1, 2, 3+1e-10])

        for method in available_dense_methods:
            Z, LS, _ = projections(A, method, orth_tol=1e-18, max_refin=10)
            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)
                assert_allclose(A.dot(x), 0, rtol=0, atol=2.5e-14)
                # Test orthogonality
                assert_allclose(orthogonality(A, x), 0, rtol=0, atol=5e-16)

    def test_rowspace_dense(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        test_points = ([1, 2, 3],
                       [1, 10, 3],
                       [1.12, 10, 0])

        for method in available_dense_methods:
            _, _, Y = projections(A, method)
            for z in test_points:
                # Test if x is solution of A x = z
                x = Y.matvec(z)
                assert_array_almost_equal(A.dot(x), z)
                # Test if x is in the return row space of A
                A_ext = np.vstack((A, x))
                assert_equal(np.linalg.matrix_rank(A),
                             np.linalg.matrix_rank(A_ext))


class TestOrthogonality(TestCase):

    def test_dense_matrix(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        test_vectors = ([-1.98931144, -1.56363389,
                         -0.84115584, 2.2864762,
                         5.599141, 0.09286976,
                         1.37040802, -0.28145812],
                        [697.92794044, -4091.65114008,
                         -3327.42316335, 836.86906951,
                         99434.98929065, -1285.37653682,
                         -4109.21503806, 2935.29289083])
        test_expected_orth = (0, 0)

        for i in range(len(test_vectors)):
            x = test_vectors[i]
            orth = test_expected_orth[i]
            assert_array_almost_equal(orthogonality(A, x), orth)
