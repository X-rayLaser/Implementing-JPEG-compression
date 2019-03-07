import unittest
import sys
import numpy as np
sys.path.insert(0, '../')
from util import Zigzag, BadArrayShapeError
from pipeline import ZigzagOrder, Configuration


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

    def test_zigzag_order_step(self):
        a = np.arange(16).reshape(4, 4)
        config = Configuration(width=4, height=4, block_size=1, dct_size=2)
        zigzag_step = ZigzagOrder(config)
        res = zigzag_step.execute(a)

        expected = np.array([
            [[0, 1, 4, 5], [2, 3, 6, 7]],
            [[8, 9, 12, 13], [10, 11, 14, 15]]
        ])

        self.assertTupleEqual(res.shape, (2, 2, 4))
        self.assertEqual(res.tolist(), expected.tolist())

    def test_restore_zigzag(self):
        a = np.arange(32).reshape(4, 8)

        config = Configuration(width=8, height=4, block_size=1, dct_size=2)
        zigzag_step = ZigzagOrder(config)

        res = zigzag_step.invert(zigzag_step.execute(a))

        self.assertEqual(res.shape, a.shape)
        self.assertEqual(res.tolist(), a.tolist())

    def test_restore_using_complex_numbers(self):
        a = np.arange(32).reshape(4, 8)
        a = a * 2j

        config = Configuration(width=8, height=4, block_size=1, dct_size=2)
        zigzag_step = ZigzagOrder(config)

        res = zigzag_step.invert(zigzag_step.execute(a))

        self.assertEqual(res.shape, a.shape)
        self.assertEqual(res.tolist(), a.tolist())
