import json
import math
import argparse
from PIL import Image
import numpy as np


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


def subsample(band_array, block_size=3):
    blocks_matrix = split_into_blocks(band_array, block_size)

    return np.mean(blocks_matrix, axis=(2, 3))


def padded_size(size, factor):
    return math.ceil(float(size) / factor) * factor


def band_to_array(band):
    pixels = np.array(list(band.getdata()))
    return pixels.reshape((band.height, band.width))


def compress(input_fname, output_fname, block_size=2):
    im = Image.open(input_fname).convert('YCbCr')

    y, cb, cr = im.split()

    padded_y = pad_array(band_to_array(y), block_size)
    padded_cb = pad_array(band_to_array(cb), block_size)
    padded_cr = pad_array(band_to_array(cr), block_size)

    mean_cb = subsample(padded_cb, block_size=block_size)
    mean_cr = subsample(padded_cr, block_size=block_size)

    width = padded_size(im.width, block_size)
    height = padded_size(im.height, block_size)

    d = {
        'width': width,
        'height': height,
        'block_size': block_size,
        'y': padded_y.tolist(),
        'mean_cb': mean_cb.tolist(),
        'mean_cr': mean_cr.tolist()
    }

    s = json.dumps(d)
    with open(output_fname, 'w') as f:
        f.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )
    parser.add_argument('infile', type=str,
                        help='a path to the file to compress')

    parser.add_argument('outfile', type=str,
                        help='a destination path')

    parser.add_argument('--block_size', action='store', type=int, default=2,
                        help='size of sub-sampling block')

    args = parser.parse_args()

    compress(args.infile, args.outfile, block_size=args.block_size)
