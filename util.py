import math
import numpy as np


def inflate(a, factor):
    """
    Reverses a sub-sampling procedure for a given array

    :param a: instance of numpy.ndarray
    :param factor: size of sub-sampling block
    :return: new instance of numpy.ndarray
    """
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)


def pad_array(a, block_size):
    """
    Adds padding to the array so as to make it size a multiple of a block size

    :param a: 2-dimensional instance of numpy.ndarray
    :param block_size: size of block
    :return: a matrix whose width and height is multiple of block_size
    """
    height = a.shape[0]

    if a.ndim != 2:
        raise BadArrayShapeError()

    if a.shape[0] == 0 or a.shape[1] == 0:
        raise EmptyArrayError()

    while a.shape[1] % block_size != 0:
        pad = np.array(list(a[:, -1])).reshape((height, 1))
        a = np.hstack((a, pad))

    while a.shape[0] % block_size != 0:
        pad = np.array(list(a[-1, :]))
        a = np.vstack((a, pad))

    return a


def undo_pad_array(a, padding):
    new_height = a.shape[0] - padding[0]
    new_width = a.shape[1] - padding[1]
    return a[:new_height, :new_width]


def extract_nth_block(blocks_column, block_size, n):
    i = n * block_size
    return blocks_column[i:i + block_size]


def block_columns(a, block_size):
    height = a.shape[0]
    width = a.shape[1]

    tmp_height = int(width * height / block_size)
    a = a.reshape((tmp_height, block_size))
    stride = width // block_size

    for j in range(stride):
        blocks_column = a[j::stride]
        yield j, blocks_column


def split_into_blocks(a, block_size):
    """
    Takes a 2-d matrix and turns it into a matrix whose entries are blocks

    :param a: 2-dimensional instance of numpy.ndarray
    :param block_size: size of block
    :return: matrix of blocks as numpy.ndarray
    """

    a = pad_array(a, block_size)

    new_height = round(a.shape[0] // block_size)
    new_width = round(a.shape[1] // block_size)

    blocks = np.zeros((new_height, new_width, block_size, block_size),
                      dtype=a.dtype)

    for j, column in block_columns(a, block_size):
        for y in range(0, new_height):
            blocks[y, j, :] = extract_nth_block(column, block_size, y)

    return blocks.reshape((new_height, new_width, block_size, block_size))


class BadArrayShapeError(Exception):
    pass


class EmptyArrayError(Exception):
    pass


def padded_size(size, factor):
    return math.ceil(float(size) / factor) * factor


def calculate_padding(a, factor):
    padded_width = padded_size(a.shape[1], factor)
    padded_height = padded_size(a.shape[0], factor)
    return padded_height - a.shape[0], padded_width - a.shape[1]


def band_to_array(band):
    pixels = np.array(list(band.getdata()))
    return pixels.reshape((band.height, band.width))


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
