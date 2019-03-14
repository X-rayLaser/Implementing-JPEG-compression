import numpy as np
from .base import AlgorithmStep
from util import RunLengthCode, padded_size


class RunLengthBlock:
    def __init__(self, block_size):
        self._size = block_size

    def non_zeros(self, a):
        for i in range(a.shape[0]):
            if a[i] != 0:
                yield a[i], i

    def encode(self, zigzag_array):
        res = []

        prev_index = -1
        for value, index in self.non_zeros(zigzag_array):
            run_length = index - prev_index - 1
            code = RunLengthCode.encode(run_length, value)
            res.extend(code)
            prev_index = index

        res.append(RunLengthCode.EOB())
        return res

    def decode(self, rle_block):
        res = []
        for code in rle_block:
            if code.is_EOB():
                nzeros = self._size - len(res)
                res.extend([0] * nzeros)
                break

            res.extend(code.decode())

        return np.array(res)


class RunLengthEncoding(AlgorithmStep):
    step_index = 7

    def execute(self, array):
        vert_blocks = array.shape[0]
        hor_blocks = array.shape[1]
        block_size = array.shape[2]

        rle_block = RunLengthBlock(block_size=block_size)

        res = []

        for i in range(vert_blocks):
            for j in range(hor_blocks):
                res.extend(
                    rle_block.encode(array[i, j])
                )

        return [c.as_tuple() for c in res]

    def invert(self, tuples_list):
        dct_size = self._config.dct_size
        vert_blocks = self._height_in_blocks()
        hor_blocks = self._width_in_blocks()

        block_size = self._config.dct_size ** 2
        rle_block = RunLengthBlock(block_size=block_size)

        decoded_blocks = []
        for block in self._rle_blocks(tuples_list):
            decoded_blocks.extend(rle_block.decode(block))

        return np.array(decoded_blocks).reshape(
            (vert_blocks, hor_blocks, dct_size ** 2)
        )

    def _height_in_blocks(self):
        padded_height = padded_size(self._config.height, self._config.block_size)
        subsampling_height = padded_height // self._config.block_size
        return padded_size(subsampling_height, self._config.dct_size) // self._config.dct_size

    def _width_in_blocks(self):
        padded_width = padded_size(self._config.width, self._config.block_size)
        subsampling_width = padded_width // self._config.block_size
        return padded_size(subsampling_width, self._config.dct_size) // self._config.dct_size

    def _rle_blocks(self, tuples_list):
        block = []
        for t in tuples_list:
            code = RunLengthCode(*t)
            block.append(code)
            if code.is_EOB():
                yield block
                block = []
