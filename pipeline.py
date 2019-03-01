import numpy as np
from util import split_into_blocks, pad_array, calculate_padding, inflate
from transforms import DCT


def undo_pad_array(a, padding):
    new_height = a.shape[0] - padding[0]
    new_width = a.shape[1] - padding[1]
    return a[:new_height, :new_width]


def subsample(band_array, block_size=3):
    blocks_matrix = split_into_blocks(band_array, block_size)

    return np.mean(blocks_matrix, axis=(2, 3))


def undo_subsample(a, block_size):
    return inflate(a, block_size)


def apply_blockwise(a, transformation, block_size):
    blocks = split_into_blocks(a, block_size)

    h = a.shape[0] // block_size
    w = a.shape[1] // block_size

    res = np.zeros(a.shape, dtype=np.int)

    for y in range(0, h):
        for x in range(w):
            i = y * block_size
            j = x * block_size
            block = np.round(transformation(blocks[y, x]))
            res[i:i + block_size, j: j + block_size] = block

    return res


def change_basis(a, dct_size):
    dct = DCT(dct_size)
    return apply_blockwise(a, dct.transform_2d, dct_size)


def undo_change_basis(a, dct_size):
    dct = DCT(dct_size)

    return apply_blockwise(a, dct.transform_2d_inverse, dct_size)


def compress_band(a, block_size=2, dct_size=8):
    padding = calculate_padding(a, block_size)
    a = pad_array(a, block_size)
    a = subsample(a, block_size)

    dct_padding = calculate_padding(a, dct_size)
    a = pad_array(a, dct_size)
    a = change_basis(a, dct_size)

    return CompressionResult(a, block_size, dct_size, padding, dct_padding)


def decompress_band(compression_result):
    a = compression_result.data

    a = undo_change_basis(a, compression_result.dct_block_size)
    a = undo_pad_array(a, compression_result.dct_padding)
    a = undo_subsample(a, compression_result.block_size)
    a = undo_pad_array(a, compression_result.subsampling_padding)
    return a


class CompressionResult:
    def __init__(self, data, block_size, dct_block_size,
                 subsampling_padding, dct_padding):
        self.data = data
        self.block_size = block_size
        self.dct_block_size = dct_block_size
        self.subsampling_padding = subsampling_padding
        self.dct_padding = dct_padding

    def as_dict(self):
        return {
            'data': self.data.tolist(),
            'block_size': self.block_size,
            'dct_block_size': self.dct_block_size,
            'subsampling_padding': self.subsampling_padding,
            'dct_padding': self.dct_padding
        }

    @staticmethod
    def from_dict(d):
        data = np.array(d['data'])
        block_size = d['block_size']
        dct_block_size = d['dct_block_size']
        dct_padding = d['dct_padding']
        subsampling_padding = d['subsampling_padding']
        return CompressionResult(data, block_size,
                                 dct_block_size,
                                 subsampling_padding,
                                 dct_padding)


