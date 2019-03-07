import unittest
import sys
import numpy as np
sys.path.insert(0, '../')
from util import Zigzag, BadArrayShapeError


class ZigzagOrderTests(unittest.TestCase):
    def test_making_zigzag_order_using_4x4_matrix(self):
        a = np.arange(16).reshape(4, 4)
        zig = Zigzag(block_size=4)
        res = zig.zigzag_order(a)

        expected_res = np.array(
            [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
        )

        self.assertEqual(res.tolist(), expected_res.tolist())

    def test_making_zigzag_order_using_3x3_matrix(self):
        a = np.arange(9).reshape(3, 3)
        zig = Zigzag(block_size=3)

        res = zig.zigzag_order(a)

        expected_res = np.array(
            [0, 1, 3, 6, 4, 2, 5, 7, 8]
        )

        print(expected_res.reshape(3, 3))

        self.assertEqual(res.tolist(), expected_res.tolist())

    def test_restore_block_from_zigzag_order(self):
        a = np.arange(16).reshape(4, 4)
        zig = Zigzag(block_size=4)
        res = zig.restore(zig.zigzag_order(a))
        self.assertEqual(res.tolist(), a.tolist())

    def test_using_malformed_arrays(self):
        a = np.arange(12).reshape(3, 4)
        zig = Zigzag(block_size=3)
        self.assertRaises(BadArrayShapeError, lambda: zig.zigzag_order(a))

        a = np.arange(12)
        zig = Zigzag(block_size=3)
        self.assertRaises(BadArrayShapeError, lambda: zig.zigzag_order(a))

        a = np.arange(16).reshape(4, 4)
        zig = Zigzag(block_size=3)
        self.assertRaises(BadArrayShapeError, lambda: zig.zigzag_order(a))

        a = np.arange(16).reshape(4, 4)
        zig = Zigzag(block_size=4)
        self.assertRaises(BadArrayShapeError, lambda: zig.restore(a))

        a = np.arange(23)
        zig = Zigzag(block_size=4)
        self.assertRaises(BadArrayShapeError, lambda: zig.restore(a))
