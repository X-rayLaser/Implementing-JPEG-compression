import unittest
import sys
import json
import numpy as np
sys.path.insert(0, '../')

from pipeline import compress_band, decompress_band, \
    Configuration, CompressionResult, QuantizationMethod, CompressedData

import file_format


class FileFormatTests(unittest.TestCase):
    def test_create_and_read_header(self):
        q = QuantizationMethod('qtable')
        config = Configuration(width=320, height=400, block_size=4,
                               dct_size=8, transform='DFT', quantization=q)
        res = file_format.get_header(file_format.create_header(config))

        self.assertEqual(config.width, res.width)
        self.assertEqual(config.height, res.height)

        self.assertEqual(config.block_size, res.block_size)
        self.assertEqual(config.dct_size, res.dct_size)
        self.assertEqual(config.transform, res.transform)

        self.assertEqual(res.quantization.name, 'qtable')

    def test_create_with_different_quantization_method(self):
        q = QuantizationMethod('divide', divisor=93)
        config = Configuration(width=320, height=400, block_size=44,
                               dct_size=16, transform='DCT', quantization=q)
        res = file_format.get_header(file_format.create_header(config))

        self.assertEqual(res.width, 320)
        self.assertEqual(res.height, 400)

        self.assertEqual(res.block_size, 44)
        self.assertEqual(res.dct_size, 16)
        self.assertEqual(res.transform, 'DCT')

        self.assertEqual(res.quantization.name, 'divide')
        self.assertEqual(res.quantization.params, {'divisor': 93})

    def test_generate_and_read_data(self):
        q = QuantizationMethod('divide', divisor=93)
        config = Configuration(width=320, height=400, block_size=44,
                               dct_size=16, transform='DCT', quantization=q)

        data = CompressedData(y=bytes([4, 8, 15, 16, 23, 42]),
                              cb=bytes([1, 2, 3, 4, 5]),
                              cr=bytes([10]))
        res = file_format.generate_data(config, data)

        read_config, read_data = file_format.read_data(res)

        self.assertEqual(read_config.dct_size, 16)
        self.assertEqual(read_data.y, bytes([4, 8, 15, 16, 23, 42]))
        self.assertEqual(read_data.cb, bytes([1, 2, 3, 4, 5]))
        self.assertEqual(read_data.cr, bytes([10]))
