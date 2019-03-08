import unittest
import sys
import numpy as np
sys.path.insert(0, '../')
from util import RunLengthBlock, RunLengthCode
from pipeline import RunLengthEncoding, Configuration


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

    def test_sequence_of_all_zeros(self):
        a = np.array([0] * 9)
        rle_block = RunLengthBlock(block_size=a.shape[0])

        res = rle_block.encode(a)

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], RunLengthCode.EOB())

        decoded = rle_block.decode(res)

        self.assertEqual(a.tolist(), decoded.tolist())


class RunLengthEncodingTests(unittest.TestCase):
    def setUp(self):
        b1 = np.array([21, 3, 0, 0, 0, 0, 2, 0, 0])
        b2 = np.array([0, 0, 0, 15, 0, 0, 0, 0, 9])
        b3 = np.array([0] * 9)
        a = np.zeros((3, 1, 9))

        a[0, 0] = b1
        a[1, 0] = b2
        a[2, 0] = b3
        self.array = a

    def test_encode_few_blocks(self):
        a = self.array
        expected = [(0, 6, 21), (0, 3, 3), (4, 3, 2), (0, 0),
                    (3, 5, 15), (4, 5, 9), (0, 0),
                    (0, 0)]

        rle = RunLengthEncoding(config=None)
        res = rle.execute(a)

        self.assertEqual(expected, res)

    def test_decode_encoded_blocks(self):
        a = self.array
        config = Configuration(width=3, height=9, block_size=1, dct_size=3)
        rle = RunLengthEncoding(config=config)

        res = rle.invert(rle.execute(a))

        self.assertEqual(res.tolist(), a.tolist())
