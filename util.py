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
                      dtype=np.int)

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
