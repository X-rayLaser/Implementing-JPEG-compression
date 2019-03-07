import unittest
import sys
import numpy as np
sys.path.insert(0, '../')
from util import zigzag


class ZigzagOrderTests(unittest.TestCase):
    def test_using_4x4_matrix(self):
        a = np.arange(16).reshape(4, 4)

        res = zigzag(a)

        expected_res = np.array(
            [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
        )

        self.assertEqual(res.tolist(), expected_res.tolist())

    def test_using_3x3_matrix(self):
        a = np.arange(9).reshape(3, 3)

        res = zigzag(a)

        expected_res = np.array(
            [0, 1, 3, 6, 4, 2, 5, 7, 8]
        )

        print(expected_res.reshape(3, 3))

        self.assertEqual(res.tolist(), expected_res.tolist())
