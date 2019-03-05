import unittest
import sys
import json
import numpy as np
sys.path.insert(0, '../')
from util import split_into_blocks, BadArrayShapeError, EmptyArrayError
from util import pad_array, padded_size
from transforms import DCT
from pipeline import compress_band, decompress_band, SubSampling,\
    Configuration, CompressionResult


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


class SubsampleTests(unittest.TestCase):
    def test_averaging(self):
        a = np.array([[1, 2, 2, 1],
                      [3, 2, 8, 1],
                      [0, 0, 2, 2],
                      [0, 4, 2, 2]])

        config = Configuration(width=123, height=854, block_size=2,
                               dct_size=2, transform='DCT', quantization='')
        sub_sampling = SubSampling(config)
        res = sub_sampling.execute(a)
        self.assertEqual(res.shape, (2, 2))
        self.assertEqual(res[0][0], 2)
        self.assertEqual(res[0][1], 3)
        self.assertEqual(res[1][0], 1)
        self.assertEqual(res[1][1], 2)

        config = Configuration(width=123, height=854, block_size=4,
                               dct_size=2, transform='DCT', quantization='')
        sub_sampling = SubSampling(config)
        res = sub_sampling.execute(a)
        self.assertEqual(res.shape, (1, 1))
        self.assertEqual(res[0][0], 2)


class DctTests(unittest.TestCase):
    def test_dct1d(self):

        a = 255 * np.cos(np.array(range(100)) * 1)

        for i in range(a.shape[0]):
            a[i] = round(a[i])

        dct = DCT(a.shape[0])
        res = dct.transform_1d_inverse(dct.transform_1d(a))

        self.assertTrue(np.allclose(a, res, rtol=0.01))

    def test_dct2d(self):

        a = np.array([[1, 2],
                      [3, 4]])

        dct = DCT(a.shape[0])

        res = dct.transform_2d_inverse(dct.transform_2d(a))

        self.assertTrue(np.allclose(a, res, rtol=0.01))

    def test_large_dct2d(self):
        a = np.arange(64).reshape((8, 8))

        dct = DCT(a.shape[0])
        res = dct.transform_2d_inverse(dct.transform_2d(a))

        self.assertTrue(np.allclose(a, res, rtol=0.01))


class PipelineTests(unittest.TestCase):
    def test_compress_and_decompress_on_array(self):
        original = np.arange(128).reshape(8, 16)

        restored = decompress_band(compress_band(original, block_size=3))
        self.assertTrue(np.allclose(original, restored, rtol=1))

    def test_fourier_transform_option(self):
        original = np.arange(128).reshape(8, 16)

        restored = decompress_band(compress_band(original, block_size=3, transform='DFT'))
        self.assertTrue(np.allclose(original, restored, rtol=1))

    def test_without_subsampling(self):
        original = np.arange(6).reshape(2, 3)

        restored = decompress_band(compress_band(original, block_size=1))
        self.assertTrue(np.allclose(original, restored, rtol=0.000001))

    def test_with_1pixel_blocks(self):
        original = np.arange(64).reshape(8, 8)

        restored = decompress_band(compress_band(original, block_size=1, dct_size=1))
        self.assertTrue(np.allclose(original, restored, rtol=0.000001))

    def test_serializability_for_integer_valued_arrays(self):
        data = np.arange(18).reshape(6, 3)
        original = CompressionResult(data=data, block_size=3,
                                     dct_block_size=9, transform_type='DFT',
                                     width=323, height=766)

        d = json.loads(json.dumps(original.as_dict()))
        reconstructed = CompressionResult.from_dict(d)

        self.assertTrue(np.allclose(original.data, reconstructed.data))
        self.assertEqual(original.block_size, reconstructed.block_size)
        self.assertEqual(original.dct_block_size, reconstructed.dct_block_size)
        self.assertEqual(original.transform_type, reconstructed.transform_type)

    def test_serializability_for_complex_valued_arrays(self):
        data = np.array([[2+3j, 3, -10j], [0j, 2-4j, -5]])
        original = CompressionResult(data=data, block_size=3,
                                     dct_block_size=9, transform_type='DFT',
                                     width=532, height=767)

        d = json.loads(json.dumps(original.as_dict()))
        reconstructed = CompressionResult.from_dict(d)

        self.assertTrue(np.allclose(original.data, reconstructed.data))


class QuantizersTests(unittest.TestCase):
    def test_rounding_quantizer_on_real_data(self):
        a = np.array([[3.4, 8.], [0, 0.6]])

        from quantizers import RoundingQuantizer

        quantizer = RoundingQuantizer()
        expected_res = np.array([[3, 8], [0, 1]])
        res = quantizer.quantize(a)
        self.assertTrue(np.allclose(res, expected_res))

        res = quantizer.restore(res)
        self.assertTrue(np.allclose(res, expected_res))

    def test_rounding_quantizer_on_complex_data(self):
        a = np.array([[1.7j, 3j], [0j, 0.6+1j]])

        from quantizers import RoundingQuantizer

        quantizer = RoundingQuantizer()
        expected_res = np.array([[2j, 3j], [0j, 1+1j]])
        res = quantizer.quantize(a)
        self.assertTrue(np.allclose(res, expected_res))

        res = quantizer.restore(res)
        self.assertTrue(np.allclose(res, expected_res))

    def test_discarding_quantizer(self):
        from quantizers import DiscardingQuantizer
        quantizer = DiscardingQuantizer(2, 1)
        a = quantizer.quantize(np.arange(9).reshape(3, 3))

        expected_result = np.array([[0, 1, 2],
                                    [3, 4, 0],
                                    [6, 7, 0]])

        self.assertTrue(np.allclose(a, expected_result))
        res = quantizer.restore(a)
        self.assertTrue(np.allclose(a, expected_result))

    def test_modulo_quantizer(self):
        from quantizers import ModuloQuantizer
        quantizer = ModuloQuantizer(40)
        a = quantizer.quantize(np.array([80, 24, 169]))

        expected_result = np.array([[2, 1, 4]])

        self.assertTrue(np.allclose(a, expected_result))
        res = quantizer.restore(a)
        self.assertTrue(np.allclose(res, np.array([80, 40, 160])))


if __name__ == '__main__':
    unittest.main()
