import random
import numpy as np

from matrix import (
    Matrix, gauss_solver, center_data, covariance_matrix,
    find_eigenvalues, find_eigenvectors, explained_variance_ratio,
    pca, reconstruction_error, auto_select_k, handle_missing_values,
    add_noise_and_compare,
)

TOL = 1e-6


def allclose(mat, expected, tol=TOL):
    for i in range(mat.rows):
        for j in range(mat.cols):
            assert abs(mat.data[i][j] - expected[i][j]) < tol


def col(values):
    return Matrix([[v] for v in values])


def to_list(vec):
    return [vec.data[i][0] for i in range(vec.rows)]

# Базовые операции с матрицами
def test_add():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    allclose(A + B, [[6, 8], [10, 12]])


def test_mul():
    A = Matrix([[1, 2], [3, 4]])
    allclose(A * 3, [[3, 6], [9, 12]])


def test_matmul():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    B = Matrix([[1, 0], [0, 1], [1, 1]])
    allclose(A @ B, (np.array(A.data) @ np.array(B.data)).tolist())


def test_matmul_dimension_error():
    try:
        Matrix([[1, 2]]) @ Matrix([[1, 2]])
        assert False
    except ValueError:
        pass


def test_transpose():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    allclose(A.transpose(), [[1, 4], [2, 5], [3, 6]])


def test_det():
    data = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    assert abs(Matrix(data).det() - float(np.linalg.det(data))) < TOL


def test_det_singular():
    assert abs(Matrix([[1, 2], [2, 4]]).det()) < TOL


def test_trace():
    assert Matrix([[1, 2], [3, 4]]).trace() == 5

# gauss_solver
def test_gauss_unique():
    A = [[2, 1], [1, 3]]
    b = [5, 10]
    sol = gauss_solver(Matrix(A), col(b))
    assert len(sol) == 1
    allclose(sol[0], [[x] for x in np.linalg.solve(A, b)])


def test_gauss_3x3():
    A = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    b = [1, 2, 3]
    sol = gauss_solver(Matrix(A), col(b))
    allclose(sol[0], [[x] for x in np.linalg.solve(A, b)])


def test_gauss_diagonal():
    sol = gauss_solver(Matrix([[2, 0], [0, 5]]), col([6, 10]))
    allclose(sol[0], [[3], [2]])


def test_gauss_1x1():
    sol = gauss_solver(Matrix([[2]]), col([6]))
    allclose(sol[0], [[3]])


def test_gauss_inconsistent():
    try:
        gauss_solver(Matrix([[1, 1], [1, 1]]), col([1, 2]))
        assert False
    except ValueError:
        pass


def test_gauss_infinite_solutions():
    A = Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    for vec in gauss_solver(A, col([0, 0, 0])):
        Av = A @ vec
        assert all(abs(Av.data[i][0]) < TOL for i in range(Av.rows))

#X_center
def test_center_data():
    X = center_data(Matrix([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]))
    for j in range(X.cols):
        mean = sum(X.data[i][j] for i in range(X.rows)) / X.rows
        assert abs(mean) < TOL

# covariance mat
def test_covariance():
    np.random.seed(0)
    data = np.random.randn(20, 3).tolist()
    C = covariance_matrix(center_data(Matrix(data)))
    allclose(C, np.cov(np.array(data).T).tolist())

#find_eigenvalues
def test_eigenvalues_2x2():
    eigs = find_eigenvalues(Matrix([[3.0, 1.0], [1.0, 3.0]]))
    exp = sorted(np.linalg.eigvalsh([[3, 1], [1, 3]]).tolist(), reverse=True)
    assert all(abs(g - e) < 1e-3 for g, e in zip(eigs, exp))


def test_eigenvalues_3x3():
    data = [[4.0, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
    eigs = find_eigenvalues(Matrix(data))
    exp = sorted(np.linalg.eigvalsh(data).tolist(), reverse=True)
    assert all(abs(g - e) < 1e-3 for g, e in zip(eigs, exp))


def test_eigenvalue_zero():
    eigs = find_eigenvalues(Matrix([[1.0, 1.0], [1.0, 1.0]]))
    assert len(eigs) == 2
    assert abs(eigs[0] - 2.0) < 1e-3
    assert abs(eigs[1] - 0.0) < 1e-3


def test_eigenvalues_diagonal():
    eigs = find_eigenvalues(Matrix([[5.0, 0.0], [0.0, 2.0]]))
    assert abs(eigs[0] - 5.0) < 1e-3
    assert abs(eigs[1] - 2.0) < 1e-3

#find_eigenvectors
def test_eigenvectors():
    C = Matrix([[4.0, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    eigs = find_eigenvalues(C)
    for val, vec in zip(eigs, find_eigenvectors(C, eigs)):
        Cv = C @ vec
        for i in range(vec.rows):
            assert abs(Cv.data[i][0] - val * vec.data[i][0]) < 1e-4


def test_eigenvectors_2():
    C = Matrix([[4.0, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    for vec in find_eigenvectors(C, find_eigenvalues(C)):
        norm = sum(v[0] ** 2 for v in vec.data) ** 0.5
        assert abs(norm - 1.0) < TOL


def test_eigenvectors_diagonal():
    C = Matrix([[5.0, 0.0], [0.0, 2.0]])
    vecs = find_eigenvectors(C, find_eigenvalues(C))
    assert abs(abs(vecs[0].data[0][0]) - 1.0) < TOL
    assert abs(vecs[0].data[1][0]) < TOL


#ratio
def test_explained_variance_ratio():
    assert abs(explained_variance_ratio([4.0, 1.0], 1) - 0.8) < TOL
    assert abs(explained_variance_ratio([4.0, 1.0], 2) - 1.0) < TOL

#auto_select
def test_auto_select_k():
    eigs = [4.0, 1.0, 0.2]
    assert auto_select_k(eigs, threshold=0.95) == 2
    assert auto_select_k(eigs, threshold=0.99) == 3

#pca
def test_pca_shape():
    np.random.seed(1)
    proj, gamma = pca(Matrix(np.random.randn(15, 4).tolist()), 2)
    assert proj.rows == 15 and proj.cols == 2
    assert 0.0 < gamma <= 1.0 + 1e-9


def test_pca_full_variance():
    np.random.seed(2)
    _, gamma = pca(Matrix(np.random.randn(15, 3).tolist()), 3)
    assert abs(gamma - 1.0) < TOL


def test_pca_variance_matches_eigenvalue():
    np.random.seed(3)
    X = Matrix(np.random.randn(40, 3).tolist())
    eigs = find_eigenvalues(covariance_matrix(center_data(X)))
    proj, _ = pca(X, 2)
    pc1 = [proj.data[i][0] for i in range(proj.rows)]
    mean = sum(pc1) / len(pc1)
    var_pc1 = sum((t - mean) ** 2 for t in pc1) / (len(pc1) - 1)
    assert abs(var_pc1 - eigs[0]) < 1e-3


def test_reconstruction_zero():
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    assert abs(reconstruction_error(A, A)) < TOL


def test_reconstruction_value():
    A = Matrix([[1.0, 2.0], [3.0, 4.0]])
    B = Matrix([[1.0, 2.0], [3.0, 5.0]])
    assert abs(reconstruction_error(A, B) - 0.25) < TOL


def test_handle_missing_values():
    nan = float('nan')
    out = handle_missing_values(Matrix([[1.0, 2.0], [nan, 4.0], [3.0, nan]]))
    assert abs(out.data[1][0] - 2.0) < TOL
    assert abs(out.data[2][1] - 3.0) < TOL
    assert out.data[0][0] == 1.0


def test_zero_noise():
    random.seed(0)
    X = Matrix([[float(i), float(2 * i + 1), float(i % 3)] for i in range(12)])
    before, after = add_noise_and_compare(X, noise_level=0.0)
    assert abs(before[1] - after[1]) < TOL


if __name__ == "__main__":
    tests = [
        test_add,
        test_mul, 
        test_matmul, 
        test_matmul_dimension_error, 
        test_transpose,
        test_det, 
        test_det_singular, 
        test_trace,
        test_gauss_unique, 
        test_gauss_3x3, 
        test_gauss_diagonal, 
        test_gauss_1x1,
        test_gauss_inconsistent, 
        test_gauss_infinite_solutions,
        test_center_data, 
        test_covariance,
        test_eigenvalues_2x2, 
        test_eigenvalues_3x3, 
        test_eigenvalue_zero, 
        test_eigenvalues_diagonal,
        test_eigenvectors, 
        test_eigenvectors_2, 
        test_eigenvectors_diagonal,
        test_explained_variance_ratio, 
        test_auto_select_k,
        test_pca_shape, 
        test_pca_full_variance, 
        test_pca_variance_matches_eigenvalue,
        test_reconstruction_zero, 
        test_reconstruction_value,
        test_handle_missing_values, 
        test_zero_noise,
    ]
    for t in tests:
        t()
    print(f"{len(tests)} из 32 тестов")
