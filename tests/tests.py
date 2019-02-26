import unittest
import sys
import numpy as np
sys.path.insert(0, '../')
from compress import split_into_blocks, BadArrayShapeError, EmptyArrayError
from compress import pad_array, padded_size


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


class SplitIntoBlocksTests(unittest.TestCase):
    def test_for_matrix_smaller_than_block_size(self):
        a = np.array([[20], [10]])

        res = split_into_blocks(a, block_size=3)

        expected = [[20, 20, 20], [10, 10, 10], [10, 10, 10]]
        self.assertEqual(res.shape, (1, 1, 3, 3))
        self.assertEqual(res[0, 0].tolist(), expected)

    def test_with_nice_matrix(self):
        a = np.arange(16).reshape((4, 4))
        print(a)

        blocks = split_into_blocks(a, block_size=2)
        self.assertEqual(blocks.shape, (2, 2, 2, 2))

        self.assertEqual(blocks[0, 0].ravel().tolist(), [0, 1, 4, 5])
        self.assertEqual(blocks[0, 1].ravel().tolist(), [2, 3, 6, 7])
        self.assertEqual(blocks[1, 0].ravel().tolist(), [8, 9, 12, 13])
        self.assertEqual(blocks[1, 1].ravel().tolist(), [10, 11, 14, 15])


if __name__ == '__main__':
    unittest.main()