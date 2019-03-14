import numpy as np
from transforms import DCT
from .base import AlgorithmStep


class BasisChange(AlgorithmStep):
    step_index = 4

    # todo: fix warnings

    def execute(self, array):
        transform = self._config.transform
        dct_size = self._config.dct_size

        if transform == 'DCT':
            res = np.zeros(array.shape, dtype=np.float)
            dct = DCT(dct_size)
            self.apply_blockwise(array, dct.transform_2d, dct_size, res)
        elif transform == 'DFT':
            res = np.zeros(array.shape, dtype=np.complex)

            def dft(dct_block):
                return np.fft.fft2(dct_block)

            self.apply_blockwise(array, dft, dct_size, res)
        return res

    def invert(self, array):
        res = np.zeros(array.shape, dtype=np.float)

        transform = self._config.transform
        dct_size = self._config.dct_size

        if transform == 'DCT':
            dct = DCT(dct_size)
            self.apply_blockwise(array, dct.transform_2d_inverse, dct_size, res)
        elif transform == 'DFT':
            def idft(dct_block):
                return np.fft.ifft2(dct_block)

            self.apply_blockwise(array, idft, dct_size, res)

        return np.array(np.round(res), dtype=np.int)
