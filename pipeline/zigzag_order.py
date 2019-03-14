import numpy as np
from .base import AlgorithmStep
from util import Zigzag


class ZigzagOrder(AlgorithmStep):
    step_index = 6

    def execute(self, array):
        dct_size = self._config.dct_size

        vert_blocks = array.shape[0] // dct_size
        hor_blocks = array.shape[1] // dct_size

        shape = (vert_blocks, hor_blocks, dct_size ** 2)

        zigzag = Zigzag(dct_size)

        res = np.zeros(shape, dtype=array.dtype)
        for block, y, x in self.blocks(array, dct_size):
            res[y, x] = zigzag.zigzag_order(block)

        return res

    def invert(self, array):
        dct_size = self._config.dct_size
        vert_blocks = array.shape[0]
        hor_blocks = array.shape[1]

        zigzag = Zigzag(dct_size)

        res = np.zeros((vert_blocks * dct_size, hor_blocks * dct_size),
                       dtype=array.dtype)

        for y in range(vert_blocks):
            for x in range(hor_blocks):
                block = zigzag.restore(array[y, x])
                i = y * dct_size
                j = x * dct_size
                res[i:i + dct_size, j:j + dct_size] = block.reshape(dct_size,
                                                                    dct_size)

        return res
