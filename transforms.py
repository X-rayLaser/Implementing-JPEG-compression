import numpy as np
import math


def dct_quotient(x, k1, k2):
    N = x.shape[0]

    s = 0
    for n1 in range(N):
        for n2 in range(N):
            s += x[n1, n2] * math.cos(np.pi / N * (n1 + 0.5) * k1) * math.cos(np.pi / N * (n2 + 0.5) * k2 )

    return s


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


def dct2d(a):
    assert a.ndim == 2
    assert a.shape[0] == a.shape[1]
    res = np.zeros(a.shape)

    N = a.shape[0]
    for k1 in range(N):
        for k2 in range(N):
            res[k1, k2] = dct_quotient(a, k1, k2)

    return res


def dct2d_inverse(a):
    assert a.ndim == 2
    assert a.shape[0] == a.shape[1]
    res = np.zeros(a.shape)
    return res
