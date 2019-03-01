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


class DCT:
    def __init__(self, size):
        self._size = size
        self._dct_matrix = dct_matrix(size)
        self._dct_normalized = dct_matrix_normalized(size)
        self._normalization_matrix = normalization_matrix(size)

    def transform_1d(self, x):
        assert x.ndim == 1
        return self._dct_matrix.dot(x)

    def transform_1d_inverse(self, x):
        assert x.ndim == 1
        W = self._dct_normalized.transpose()
        Dinv = self._normalization_matrix
        return W.dot(Dinv.dot(x))

    def transform_2d(self, a):
        assert a.ndim == 2
        assert a.shape[0] == a.shape[1]

        # 2d DCT consists of 2 sets of 1-dimensional DCTs

        # 1-dimensional DCT for each row of matrix a
        # each resulting row is vector of DCT coefficients of row of a
        M = self._transform_matrix(a, self.transform_1d)

        # 1-dimensional DCT of each column of matrix Tx
        # each resulting column is vector of DCT coefficients of column of M
        return self._transform_matrix(M.T, self.transform_1d).T

    def transform_2d_inverse(self, a):
        assert a.ndim == 2
        assert a.shape[0] == a.shape[1]

        # similar to dct2d
        # performing 1d DCT inverse for each column of matrix a
        M = self._transform_matrix(a.T, self.transform_1d_inverse).T

        # performing 1d DCT inverse for each row of matrix M
        return self._transform_matrix(M, self.transform_1d_inverse)

    def _transform_matrix(self, matrix, transformation):
        res = np.zeros(matrix.shape)
        for i in range(matrix.shape[0]):
            res[i] = transformation(matrix[i])
        return res
