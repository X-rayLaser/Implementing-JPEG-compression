import numpy as np
from .base import AlgorithmStep


class Quantization(AlgorithmStep):
    step_index = 5

    def execute(self, array):
        res = np.zeros(array.shape, dtype=array.dtype)

        quantizer = self._config.quantization.quantizer

        def f(dct_block):
            return quantizer.quantize(dct_block)

        self.apply_blockwise(array, f, self._config.dct_size, res)

        return res

    def invert(self, array):
        res = np.zeros(array.shape, dtype=array.dtype)

        quantizer = self._config.quantization.quantizer

        def f(dct_block):
            return quantizer.restore(dct_block)

        self.apply_blockwise(array, f, self._config.dct_size, res)

        return res
