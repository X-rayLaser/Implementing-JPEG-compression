import unittest
import sys
import numpy as np
sys.path.insert(0, '../')
from transforms import DCT


class DctTests(unittest.TestCase):
    def test_dct1d(self):

        a = 255 * np.cos(np.array(range(100)) * 1)

        for i in range(a.shape[0]):
            a[i] = round(a[i])

        dct = DCT(a.shape[0])
        res = dct.transform_1d_inverse(dct.transform_1d(a))

        self.assertTrue(np.allclose(a, res, rtol=0.01))

    def test_dct2d(self):

        a = np.array([[1, 2],
                      [3, 4]])

        dct = DCT(a.shape[0])

        res = dct.transform_2d_inverse(dct.transform_2d(a))

        self.assertTrue(np.allclose(a, res, rtol=0.01))

    def test_large_dct2d(self):
        a = np.arange(64).reshape((8, 8))

        dct = DCT(a.shape[0])
        res = dct.transform_2d_inverse(dct.transform_2d(a))

        self.assertTrue(np.allclose(a, res, rtol=0.01))
