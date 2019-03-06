import unittest
import sys
import numpy as np
sys.path.insert(0, '../')
from util import split_into_blocks


class SplitIntoBlocksTests(unittest.TestCase):
    def test_for_matrix_smaller_than_block_size(self):
        a = np.array([[20], [10]])

        res = split_into_blocks(a, block_size=3)

        expected = [[20, 20, 20], [10, 10, 10], [10, 10, 10]]
        self.assertEqual(res.shape, (1, 1, 3, 3))
        self.assertEqual(res[0, 0].tolist(), expected)

    def test_with_nice_matrix(self):
        a = np.arange(16).reshape((4, 4))

        blocks = split_into_blocks(a, block_size=2)
        self.assertEqual(blocks.shape, (2, 2, 2, 2))

        self.assertEqual(blocks[0, 0].ravel().tolist(), [0, 1, 4, 5])
        self.assertEqual(blocks[0, 1].ravel().tolist(), [2, 3, 6, 7])
        self.assertEqual(blocks[1, 0].ravel().tolist(), [8, 9, 12, 13])
        self.assertEqual(blocks[1, 1].ravel().tolist(), [10, 11, 14, 15])

    def test_with_complex_matrix(self):
        a = np.array([[3 - 2j]])
        blocks = split_into_blocks(a, block_size=1)
        self.assertEqual(blocks[0, 0].ravel().tolist(), [3 - 2j])
