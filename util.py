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


class RunLengthCode:
    max_run_length = 15

    @staticmethod
    def EOB():
        return RunLengthCode(0, 0, 0)

    @staticmethod
    def all_zeros():
        return RunLengthCode(15, 0, 0)

    @staticmethod
    def encode(run_length, amplitude):
        res = []

        n0chains = run_length // RunLengthCode.max_run_length

        for i in range(n0chains):
            res.append(RunLengthCode.all_zeros())

        run_length = run_length % RunLengthCode.max_run_length

        bit_size = math.ceil(math.log2(abs(amplitude) + 1)) + 1
        res.append(RunLengthCode(run_length, bit_size, amplitude))

        return res

    def __init__(self, run_length, size, amplitude=0):
        self.run_length = run_length
        self.size = size
        self.amplitude = amplitude

    def decode(self):
        if self.is_zeros_chain():
            return [0] * self.max_run_length

        return [0] * self.run_length + [self.amplitude]

    def is_zeros_chain(self):
        return (self.run_length == self.max_run_length and
                self.size == 0 and self.amplitude == 0)

    def is_EOB(self):
        return self.run_length == 0 and self.size == 0

    def as_tuple(self):
        if self.is_EOB():
            return 0, 0

        return self.run_length, self.size, int(round(self.amplitude))

    def __eq__(self, other):
        return (self.run_length == other.run_length and
                self.size == other.size and
                self.amplitude == other.amplitude)

    def __repr__(self):
        return '({}, {}, {})'.format(self.run_length, self.size, self.amplitude)


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
