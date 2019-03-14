import numpy as np
from .base import AlgorithmStep
from util import BadArrayShapeError


class Zigzag:
    def __init__(self, block_size):
        self._size = block_size
        self._indices = None

    def zigzag_order(self, block):
        self._validate_block(block)
        results_list = [block[i, j] for i, j in self.zigzag_indices]
        return np.array(results_list)

    def restore(self, zigzag_array):
        self._validate_zigzag(zigzag_array)

        indices = self.zigzag_indices

        block = np.zeros((self._size, self._size), dtype=zigzag_array.dtype)
        for value, pos in zip(zigzag_array, indices):
            i, j = pos
            block[i, j] = value
        return block

    @property
    def zigzag_indices(self):
        if self._indices:
            return self._indices

        indices = []

        count = 0

        for d in self._diagonals():
            if count % 2 == 1:
                d.reverse()
            indices.extend(d)
            count += 1

        self._indices = indices
        return indices

    def _validate_block(self, a):
        if not (a.ndim == 2 and a.shape[0] == a.shape[1] and
                a.shape[0] == self._size):
            raise BadArrayShapeError(a.shape)

    def _validate_zigzag(self, zigzag_array):
        if not (zigzag_array.ndim == 1 and
                zigzag_array.shape[0] == self._size ** 2):
            raise BadArrayShapeError(zigzag_array.shape)

    def _left_top_diagonal(self, row):
        indices = []
        for i in range(row, -1, -1):
            j = row - i
            indices.append((i, j))
        return indices

    def _bottom_right_diagonal(self, col):
        size = self._size
        indices = []
        for j in range(col, size):
            top_row = size - 1
            delta_j = (j - col)
            i = top_row - delta_j
            indices.append((i, j))
        return indices

    def _diagonals(self):
        size = self._size

        for row in range(size):
            yield self._left_top_diagonal(row)

        for col in range(1, size):
            yield self._bottom_right_diagonal(col)


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
