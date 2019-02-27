import numpy as np


def dct_matrix(size):
    N = size
    a = np.zeros((N, N))

    for k in range(N):
        args = [np.pi / N * (n + 0.5) * k for n in range(N)]
        a[k, :] = np.cos(np.array(args))
    return a


def dct_matrix_normalized(size):
    w = dct_matrix(size)

    for k in range(size):
        w[k] /= np.linalg.norm(w[k])

    return w


def normalization_matrix(size):
    a = dct_matrix(size)
    quotients = 1.0 / np.linalg.norm(a, axis=1)
    return np.diag(quotients)


def dct1d(x):
    assert x.ndim == 1
    N = x.shape[0]
    return dct_matrix(N).dot(x)


def dct1d_inverse(x):
    assert x.ndim == 1
    N = x.shape[0]

    W = dct_matrix_normalized(N).transpose()
    Dinv = normalization_matrix(N)
    return W.dot(Dinv.dot(x))


def transform_matrix(matrix, transformation=dct1d):
    res = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        res[i] = transformation(matrix[i])
    return res


def dct2d(a):
    assert a.ndim == 2
    assert a.shape[0] == a.shape[1]

    # 2d DCT consists of 2 sets of 1-dimensional DCTs

    # 1-dimensional DCT for each row of matrix a
    # each resulting row is vector of DCT coefficients of row of a
    M = transform_matrix(a, dct1d)

    # 1-dimensional DCT of each column of matrix Tx
    # each resulting column is vector of DCT coefficients of column of M
    return transform_matrix(M.T, dct1d).T


def dct2d_inverse(a):
    assert a.ndim == 2
    assert a.shape[0] == a.shape[1]

    # similar to dct2d
    # performing 1d DCT inverse for each column of matrix a
    M = transform_matrix(a.T, dct1d_inverse).T

    # performing 1d DCT inverse for each row of matrix M
    return transform_matrix(M, dct1d_inverse)
