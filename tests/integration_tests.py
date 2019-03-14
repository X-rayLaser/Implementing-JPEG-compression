import unittest
import sys
import numpy as np
sys.path.insert(0, '../')

from pipeline import compress_band, decompress_band, \
    Configuration, QuantizationMethod


class PipelineTests(unittest.TestCase):
    def test_compress_and_decompress_on_array(self):
        original = np.arange(128).reshape(8, 16)

        config = Configuration(width=16, height=8, block_size=3)
        restored = decompress_band(compress_band(
            original, config
        ), config)

        self.assertTrue(np.allclose(original, restored, rtol=1))

    def test_preserves_allowed_range(self):
        original = np.array([[220, 255, 123, 205],
                             [255, 255, 112, 10],
                             [15, 51, 83, 221],
                             [239, 73, 62, 22]])

        self.assertTrue(np.all(original < 256))
        self.assertTrue(np.all(original > -1))

        config = Configuration(
                width=4, height=4, block_size=1, dct_size=2,
                quantization=QuantizationMethod('divide', divisor=129)
        )
        restored = decompress_band(compress_band(
            original, config
        ), config)

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

        config = Configuration(width=3, height=2, block_size=1)
        restored = decompress_band(compress_band(
            original, config
        ), config)

        self.assertTrue(np.allclose(original, restored, rtol=0.000001))

    def test_with_1pixel_blocks(self):
        original = np.arange(64).reshape(8, 8)
        config = Configuration(width=8, height=8, block_size=1, dct_size=1)
        restored = decompress_band(compress_band(
            original, config
        ), config)
        self.assertTrue(np.allclose(original, restored, rtol=0.000001))
