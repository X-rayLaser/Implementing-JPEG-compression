import unittest
import sys
import numpy as np
sys.path.insert(0, '../')
from util import split_into_blocks, BadArrayShapeError, EmptyArrayError
from util import pad_array, padded_size


class PaddingTests(unittest.TestCase):
    def test_with_1d_array(self):
        def f():
            split_into_blocks(np.array([32, 31]), block_size=2)

        self.assertRaises(BadArrayShapeError, f)

    def test_with_3d_array(self):
        def f():
            split_into_blocks(np.array([[[32]]]), block_size=2)

        self.assertRaises(BadArrayShapeError, f)

    def test_with_empty_matrix(self):
        def f():
            a = np.array([[]])
            res = split_into_blocks(a, block_size=3)

        self.assertRaises(EmptyArrayError, f)

    def test_pad_array(self):
        a = np.array([[20], [10]])

        res = pad_array(a, block_size=3)

        expected = [[20, 20, 20], [10, 10, 10], [10, 10, 10]]
        self.assertEqual(res.shape, (3, 3))
        self.assertEqual(res.tolist(), expected)

    def test_pad_array_when_no_padding_is_required(self):
        a = np.array([[20, 3], [10, 9]])

        res = pad_array(a, block_size=2)

        expected = [[20, 3], [10, 9]]
        self.assertEqual(res.shape, (2, 2))
        self.assertEqual(res.tolist(), expected)

    def test_padded_size(self):
        self.assertEqual(padded_size(3, 3), 3)

        self.assertEqual(padded_size(4, 3), 6)
        self.assertEqual(padded_size(5, 3), 6)
        self.assertEqual(padded_size(6, 3), 6)

        self.assertEqual(padded_size(7, 3), 9)
