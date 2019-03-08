import unittest
import sys
import json
import numpy as np
sys.path.insert(0, '../')

from pipeline import compress_band, decompress_band, \
    Configuration, CompressionResult, QuantizationMethod


class PipelineTests(unittest.TestCase):
    def test_compress_and_decompress_on_array(self):
        original = np.arange(128).reshape(8, 16)

        restored = decompress_band(compress_band(
            original, Configuration(width=16, height=8, block_size=3)
        ))
        self.assertTrue(np.allclose(original, restored, rtol=1))

    def test_preserves_allowed_range(self):
        original = np.array([[220, 255, 123, 205],
                             [255, 255, 112, 10],
                             [15, 51, 83, 221],
                             [239, 73, 62, 22]])

        self.assertTrue(np.all(original < 256))
        self.assertTrue(np.all(original > -1))

        restored = decompress_band(compress_band(
            original, Configuration(
                width=4, height=4, block_size=1, dct_size=2,
                quantization=QuantizationMethod('divide', divisor=129))
        ))

        self.assertTrue(np.all(restored < 256))
        self.assertTrue(np.all(restored > -1))

    def test_fourier_transform_option(self):
        original = np.arange(128).reshape(8, 16)

        restored = decompress_band(compress_band(
            original, Configuration(width=16, height=8,
                                    block_size=3, transform='DFT')
        ))
        self.assertTrue(np.allclose(original, restored, rtol=1))

    def test_without_subsampling(self):
        original = np.arange(6).reshape(2, 3)

        restored = decompress_band(compress_band(
            original, Configuration(width=3, height=2, block_size=1)
        ))
        self.assertTrue(np.allclose(original, restored, rtol=0.000001))

    def test_with_1pixel_blocks(self):
        original = np.arange(64).reshape(8, 8)

        restored = decompress_band(compress_band(
            original, Configuration(width=8, height=8, block_size=1, dct_size=1)
        ))
        self.assertTrue(np.allclose(original, restored, rtol=0.000001))

    def test_serializability_for_integer_valued_arrays(self):
        data = np.arange(18).reshape(2, 3, 3)

        config = Configuration(width=323, height=766, block_size=3,
                               dct_size=9, transform='DFT', quantization=None)

        original = CompressionResult(data, config)

        d = json.loads(json.dumps(original.as_dict()))
        reconstructed = CompressionResult.from_dict(d)

        self.assertTrue(np.allclose(original.data, reconstructed.data))
        self.assertEqual(original.config.block_size, reconstructed.config.block_size)
        self.assertEqual(original.config.dct_size, reconstructed.config.dct_size)
        self.assertEqual(original.config.transform, reconstructed.config.transform)

    def test_serializability_for_complex_valued_arrays(self):
        data = (np.arange(64) * 3j).reshape(4, 4, 4)

        config = Configuration(width=323, height=766, block_size=3,
                               dct_size=9, transform='DFT', quantization=None)
        original = CompressionResult(data, config)

        d = json.loads(json.dumps(original.as_dict()))
        reconstructed = CompressionResult.from_dict(d)

        self.assertTrue(np.allclose(original.data, reconstructed.data))
