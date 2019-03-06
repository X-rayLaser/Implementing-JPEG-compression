import unittest
import sys
import numpy as np
sys.path.insert(0, '../')

from quantizers import RoundingQuantizer, DivisionQuantizer, DiscardingQuantizer


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

        quantizer = RoundingQuantizer()
        expected_res = np.array([[2j, 3j], [0j, 1+1j]])
        res = quantizer.quantize(a)
        self.assertTrue(np.allclose(res, expected_res))

        res = quantizer.restore(res)
        self.assertTrue(np.allclose(res, expected_res))

    def test_discarding_quantizer(self):
        quantizer = DiscardingQuantizer(2)
        a = quantizer.quantize(np.arange(9).reshape(3, 3))

        expected_result = np.array([[0, 1, 0],
                                    [3, 4, 0],
                                    [0, 0, 0]])

        self.assertTrue(np.allclose(a, expected_result))
        res = quantizer.restore(a)
        self.assertTrue(np.allclose(res, expected_result))

    def test_modulo_quantizer(self):
        quantizer = DivisionQuantizer(40)
        a = quantizer.quantize(np.array([80, 24, 169]))

        expected_result = np.array([[2, 1, 4]])

        self.assertTrue(np.allclose(a, expected_result))
        res = quantizer.restore(a)
        self.assertTrue(np.allclose(res, np.array([80, 40, 160])))
