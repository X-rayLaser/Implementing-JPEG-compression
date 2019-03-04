import numpy as np
from util import split_into_blocks, pad_array, calculate_padding, inflate
from transforms import DCT
from quantizers import RoundingQuantizer, DiscardingQuantizer,\
    ModuloQuantizer


def undo_pad_array(a, padding):
    new_height = a.shape[0] - padding[0]
    new_width = a.shape[1] - padding[1]
    return a[:new_height, :new_width]


def subsample(band_array, block_size=3):
    blocks_matrix = split_into_blocks(band_array, block_size)

    return np.mean(blocks_matrix, axis=(2, 3))


def undo_subsample(a, block_size):
    return inflate(a, block_size)


def apply_blockwise(a, transformation, block_size, res):
    blocks = split_into_blocks(a, block_size)

    h = a.shape[0] // block_size
    w = a.shape[1] // block_size

    for y in range(0, h):
        for x in range(w):
            i = y * block_size
            j = x * block_size
            block = transformation(blocks[y, x])
            res[i:i + block_size, j: j + block_size] = block


def change_basis(a, dct_size, transform='DCT'):
    if transform == 'DCT':
        res = np.zeros(a.shape, dtype=np.float)
        dct = DCT(dct_size)
        apply_blockwise(a, dct.transform_2d, dct_size, res)
    elif transform == 'DFT':
        res = np.zeros(a.shape, dtype=np.complex)

        def dft(dct_block):
            return np.fft.fft2(dct_block)

        apply_blockwise(a, dft, dct_size, res)
    return res


def undo_change_basis(a, dct_size, transform='DCT'):
    res = np.zeros(a.shape, dtype=np.float)

    if transform == 'DCT':
        dct = DCT(dct_size)
        apply_blockwise(a, dct.transform_2d_inverse, dct_size, res)
    elif transform == 'DFT':
        def idft(dct_block):
            return np.fft.ifft2(dct_block)

        apply_blockwise(a, idft, dct_size, res)
    return res


def quantize(a, dct_size, quantizer=RoundingQuantizer()):
    res = np.zeros(a.shape, dtype=np.float)

    def f(dct_block):
        return quantizer.quantize(dct_block)

    apply_blockwise(a, f, dct_size, res)

    return res


def invert_quantize(a, dct_size, quantizer=RoundingQuantizer()):
    res = np.zeros(a.shape, dtype=np.float)

    def f(dct_block):
        return quantizer.restore(dct_block)

    apply_blockwise(a, f, dct_size, res)

    return res


def compress_band(a, block_size=2, dct_size=8, transform='DCT'):
    padding = calculate_padding(a, block_size)
    a = pad_array(a, block_size)
    a = subsample(a, block_size)

    dct_padding = calculate_padding(a, dct_size)
    a = pad_array(a, dct_size)
    a = change_basis(a, dct_size, transform)
    a = quantize(a, dct_size)

    return CompressionResult(a, block_size, dct_size, padding, dct_padding,
                             transform)


def decompress_band(compression_result):
    a = compression_result.data

    a = invert_quantize(a, compression_result.dct_block_size)
    a = undo_change_basis(a, compression_result.dct_block_size,
                          compression_result.transform_type)

    a = np.array(np.round(a), dtype=np.int)
    a = undo_pad_array(a, compression_result.dct_padding)
    a = undo_subsample(a, compression_result.block_size)
    a = undo_pad_array(a, compression_result.subsampling_padding)
    return a


class CompressionResult:
    def __init__(self, data, block_size, dct_block_size,
                 subsampling_padding, dct_padding, transform_type):
        self.data = data
        self.block_size = block_size
        self.dct_block_size = dct_block_size
        self.subsampling_padding = subsampling_padding
        self.dct_padding = dct_padding
        self.transform_type = transform_type

    def _serialize_complex(self, a):
        res = []
        for i in range(a.shape[0]):
            res.append([])
            for j in range(a.shape[1]):
                c = a[i, j]
                res[i].append({
                    'real': c.real,
                    'imag': c.imag
                })

        return res

    @staticmethod
    def _reconstruct_complex(list_of_lists):
        assert len(list_of_lists) > 0
        res = []
        nrows = len(list_of_lists)
        ncols = len(list_of_lists[0])

        for i in range(nrows):
            res.append([])
            for j in range(ncols):
                d = list_of_lists[i][j]
                complex_val = np.complex(d['real'], d['imag'])
                res[i].append(complex_val)

        return np.array(res)

    def as_dict(self):
        if self.data.dtype == np.complex:
            data = {
                'dtype': 'complex',
                'values': self._serialize_complex(self.data)
            }
        else:
            data = {
                'dtype': 'int',
                'values': self.data.tolist()
            }

        return {
            'data': data,
            'block_size': self.block_size,
            'dct_block_size': self.dct_block_size,
            'subsampling_padding': self.subsampling_padding,
            'dct_padding': self.dct_padding,
            'transform_type': self.transform_type
        }

    @staticmethod
    def from_dict(d):
        if d['data']['dtype'] == 'complex':
            data = CompressionResult._reconstruct_complex(d['data']['values'])
        else:
            data = np.array(d['data']['values'])

        block_size = d['block_size']
        dct_block_size = d['dct_block_size']
        dct_padding = d['dct_padding']
        subsampling_padding = d['subsampling_padding']
        transform_type = d['transform_type']
        return CompressionResult(data, block_size,
                                 dct_block_size,
                                 subsampling_padding,
                                 dct_padding,
                                 transform_type)
