import numpy as np
from util import split_into_blocks, pad_array, padded_size, inflate

def undo_pad_array(a, padding):
    new_height = a.shape[0] - padding[0]
    new_width = a.shape[1] - padding[1]
    return a[:new_height, :new_width]


def subsample(band_array, block_size=3):
    blocks_matrix = split_into_blocks(band_array, block_size)

    return np.mean(blocks_matrix, axis=(2, 3))


def undo_subsample(a, block_size):
    return inflate(a, block_size)


def change_basis(a, dct_size):
    blocks = split_into_blocks(a, dct_size)

    h = a.shape[0] // dct_size
    w = a.shape[1] // dct_size

    res = np.zeros((h, w), dtype=np.int)
    for y in range(0, h):
        for x in range(w):
            i = y * dct_size
            j = x * dct_size
            res[i:i + dct_size, j: j + dct_size] = np.round(dct2d(blocks[y, x]))

    return res


def compress_band(a, block_size=2, dct_size=8):
    padded_width = padded_size(a.shape[1], block_size)
    padded_height = padded_size(a.shape[0], block_size)
    padding = (padded_height - a.shape[0], padded_width - a.shape[1])

    a = pad_array(a, block_size)
    a = subsample(a, block_size)

    return CompressionResult(a, block_size, padding)


def decompress_band(compression_result):
    a = compression_result.data
    a = undo_subsample(a, compression_result.block_size)
    a = undo_pad_array(a, compression_result.subsampling_padding)
    return a


class CompressionResult:
    def __init__(self, data, block_size, subsampling_padding):
        self.data = data
        self.block_size = block_size
        self.subsampling_padding = subsampling_padding

    def as_dict(self):
        return {
            'data': self.data.tolist(),
            'block_size': self.block_size,
            'subsampling_padding': self.subsampling_padding
        }

    @staticmethod
    def from_dict(d):
        data = np.array(d['data'])
        block_size = d['block_size']
        subsampling_padding = d['subsampling_padding']
        return CompressionResult(data, block_size,
                                 subsampling_padding)

