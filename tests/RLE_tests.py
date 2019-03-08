import unittest
import sys
import numpy as np
sys.path.insert(0, '../')
from util import RunLengthBlock, RunLengthCode
from pipeline import ZigzagOrder, Configuration


class RunLengthBlockTests(unittest.TestCase):
    def setUp(self):
        self.array = np.array([-15, 0, 0, 0, 3, 2, 0, 0, 0, 0, 120, 0, 0, 0, 0])

    def test_encode_some_ordinary_array(self):
        a = self.array

        rle_block = RunLengthBlock(block_size=a.shape[0])
        result = rle_block.encode(a)

        self.assertEqual(result[0], RunLengthCode(0, 5, -15))
        self.assertEqual(result[1], RunLengthCode(3, 3, 3))
        self.assertEqual(result[2], RunLengthCode(0, 3, 2))
        self.assertEqual(result[3], RunLengthCode(4, 8, 120))
        self.assertTrue(result[4].is_EOB)

    def test_decoding_RLE_block(self):
        a = self.array

        rle_block = RunLengthBlock(block_size=a.shape[0])
        result = rle_block.decode(rle_block.encode(a))

        self.assertEqual(a.tolist(), result.tolist())

    def test_using_long_sequences_of_zeros(self):
        a = np.array([0, 2] + [0] * 32 + [5] + [0] * 5)

        rle_block = RunLengthBlock(block_size=a.shape[0])
        result = rle_block.encode(a)

        self.assertEqual(result[0], RunLengthCode(1, 3, 2))
        self.assertEqual(result[1], RunLengthCode(15, 0, 0))
        self.assertEqual(result[2], RunLengthCode(15, 0, 0))
        self.assertEqual(result[3], RunLengthCode(2, 4, 5))
        self.assertTrue(result[4].is_EOB)

        result = rle_block.decode(rle_block.encode(a))
        self.assertEqual(a.tolist(), result.tolist())
