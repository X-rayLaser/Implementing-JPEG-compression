import json
import numpy as np
from bitarray import bitarray
from PIL import Image
from util import split_into_blocks, pad_array, undo_pad_array,\
    padded_size, inflate, Zigzag, RunLengthCode, RunLengthBlock, band_to_array
from transforms import DCT
from quantizers import RoundingQuantizer, DiscardingQuantizer,\
    DivisionQuantizer, JpegQuantizationTable
import file_format

step_classes = []


class Meta(type):
    @staticmethod
    def validate_index(cls, name, class_dict):
        attr_name = 'step_index'
        if attr_name not in class_dict:
            raise MissingStepIndexError(
                'Class {} has not defined "{}" class attribute'.format(
                    name, attr_name
                )
            )

        expected_index = len(step_classes)
        step_index = cls.step_index

        if step_index != expected_index:
            raise IndexOutOfOrderError(
                '{}-th algorithm step "{}" got index {}'.format(
                    expected_index, name, step_index
                )
            )

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)

        if name != 'AlgorithmStep':
            Meta.validate_index(cls, name, class_dict)
            step_classes.append(cls)
        return cls


class IndexOutOfOrderError(Exception):
    pass


class MissingStepIndexError(Exception):
    pass


class QuantizationMethod:
    name_to_quantizer = {
        'none': RoundingQuantizer,
        'discard': DiscardingQuantizer,
        'divide': DivisionQuantizer,
        'qtable': JpegQuantizationTable
    }

    def __init__(self, name, **kwargs):
        self.name = name
        self.params = kwargs
        self.quantizer = self._get_quantizer()

    def _get_quantizer(self):
        error_msg = 'name {}, params {}'.format(self.name, self.params)
        if self.name not in self.name_to_quantizer.keys():
            raise BadQuantizationError(error_msg)

        try:
            return self.name_to_quantizer[self.name](**self.params)
        except Exception:
            raise BadQuantizationError(error_msg)

    def to_json(self):
        d = dict(self.params)
        d['quantization_scheme_name'] = self.name
        return json.dumps(d)

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        name = d['quantization_scheme_name']
        params = dict(d)
        del params['quantization_scheme_name']
        return QuantizationMethod(name, **params)


class Configuration:
    def __init__(self, width, height, block_size=2, dct_size=8,
                 transform='DCT', quantization=None):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.dct_size = dct_size
        self.transform = transform

        if quantization is None:
            self.quantization = QuantizationMethod('none')
        else:
            if quantization.name == 'qtable' and dct_size != 8:
                raise BadQuantizationError()
            self.quantization = quantization


class BadQuantizationError(Exception):
    pass


class AlgorithmStep(metaclass=Meta):
    def __init__(self, config):
        self._config = config

    def execute(self, array):
        raise NotImplementedError

    def invert(self, array):
        raise NotImplementedError

    def calculate_padding(self, factor):
        w, h = self._config.width, self._config.height
        padded_width = padded_size(w, factor)
        padded_height = padded_size(h, factor)
        return padded_height - h, padded_width - w

    def blocks(self, a, block_size):
        blocks = split_into_blocks(a, block_size)

        h = a.shape[0] // block_size
        w = a.shape[1] // block_size

        for y in range(0, h):
            for x in range(w):
                yield blocks[y, x], y, x

    def apply_blockwise(self, a, transformation, block_size, res):
        for block, y, x in self.blocks(a, block_size):
            i = y * block_size
            j = x * block_size
            res[i:i + block_size, j: j + block_size] = transformation(block)


class Padding(AlgorithmStep):
    step_index = 0

    def execute(self, array):
        if self._config.block_size == 1:
            return array

        return pad_array(array, self._config.block_size)

    def invert(self, array):
        padding = self.calculate_padding(self._config.block_size)
        return undo_pad_array(array, padding)


class SubSampling(AlgorithmStep):
    step_index = 1

    def execute(self, array):
        blocks_matrix = split_into_blocks(array, self._config.block_size)
        return np.mean(blocks_matrix, axis=(2, 3))

    def invert(self, array):
        return inflate(array, self._config.block_size)


class DCTPadding(AlgorithmStep):
    step_index = 2

    def execute(self, array):
        return pad_array(array, self._config.dct_size)

    def invert(self, array):
        w, h = self._config.width, self._config.height

        tmp_w = padded_size(w, self._config.block_size) // self._config.block_size
        tmp_h = padded_size(h, self._config.block_size) // self._config.block_size
        padded_width = padded_size(tmp_w, self._config.dct_size)
        padded_height = padded_size(tmp_h, self._config.dct_size)

        padding = padded_height - tmp_h, padded_width - tmp_w

        return undo_pad_array(array, padding)


class Normalization(AlgorithmStep):
    step_index = 3

    def execute(self, array):
        return array

    def invert(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i][j] = max(0, min(255, array[i][j]))
        return array


class BasisChange(AlgorithmStep):
    step_index = 4

    def execute(self, array):
        transform = self._config.transform
        dct_size = self._config.dct_size

        if transform == 'DCT':
            res = np.zeros(array.shape, dtype=np.float)
            dct = DCT(dct_size)
            self.apply_blockwise(array, dct.transform_2d, dct_size, res)
        elif transform == 'DFT':
            res = np.zeros(array.shape, dtype=np.complex)

            def dft(dct_block):
                return np.fft.fft2(dct_block)

            self.apply_blockwise(array, dft, dct_size, res)
        return res

    def invert(self, array):
        res = np.zeros(array.shape, dtype=np.float)

        transform = self._config.transform
        dct_size = self._config.dct_size

        if transform == 'DCT':
            dct = DCT(dct_size)
            self.apply_blockwise(array, dct.transform_2d_inverse, dct_size, res)
        elif transform == 'DFT':
            def idft(dct_block):
                return np.fft.ifft2(dct_block)

            self.apply_blockwise(array, idft, dct_size, res)

        return np.array(np.round(res), dtype=np.int)


class Quantization(AlgorithmStep):
    step_index = 5

    def execute(self, array):
        res = np.zeros(array.shape, dtype=array.dtype)

        quantizer = self._config.quantization.quantizer

        def f(dct_block):
            return quantizer.quantize(dct_block)

        self.apply_blockwise(array, f, self._config.dct_size, res)

        return res

    def invert(self, array):
        res = np.zeros(array.shape, dtype=array.dtype)

        quantizer = self._config.quantization.quantizer

        def f(dct_block):
            return quantizer.restore(dct_block)

        self.apply_blockwise(array, f, self._config.dct_size, res)

        return res


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


class BitDecoder:
    def __init__(self, array):
        self._array = array
        self._pos = 0

    def decode_unsigned(self, n):
        return self._decode_unsigned(self.read(n))

    def decode_signed(self, n):
        return self._decode_signed(self.read(n))

    def read_quad(self):
        return self.read(4)

    def read(self, n):
        index = self._pos
        res = self._array[index:index + n]
        self._pos += n
        return res

    def skip_padding(self):
        while self._pos % 8 > 0:
            self.read(1)

    def is_end(self):
        return self._pos >= len(self._array)

    def _decode_unsigned(self, bits):
        return int(bits.to01(), base=2)

    def _decode_signed(self, bits):
        res = int(bits.to01()[1:], base=2)

        negative = bits.to01()[0] == '0'
        if negative:
            res = -res
        return res


class RleBytestream(AlgorithmStep):
    step_index = 8

    def execute(self, tuples_list):
        res = bitarray()

        for t in tuples_list:
            code = RunLengthCode(*t)
            res.extend(code.as_bitsring())

            if code.is_EOB():
                self._pad_bitarray(res)

        return res.tobytes()

    def invert(self, bytestream):
        a = bitarray()
        a.frombytes(bytestream)
        tup_list = []

        for code in self._codes(a):
            tup_list.append(code.as_tuple())

        return tup_list

    def _pad_bitarray(self, a):
        while len(a) % 8 > 0:
            a.append(False)

    def _codes(self, bits):
        decoder = BitDecoder(bits)
        while not decoder.is_end():
            run_len = decoder.decode_unsigned(4)
            size = decoder.decode_unsigned(4)

            if run_len == 0 and size == 0:
                decoder.skip_padding()
                code = RunLengthCode.EOB()
            elif run_len == 15 and size == 0:
                code = RunLengthCode(15, 0, 0)
            else:
                amplitude = decoder.decode_signed(size)
                code = RunLengthCode(run_len, size, amplitude)
            yield code


def compress_band(a, config):
    for cls in step_classes:
        step = cls(config)
        a = step.execute(a)

    return a


def decompress_band(compression_result, config):
    a = compression_result

    reversed_steps = list(step_classes)
    reversed_steps.reverse()
    for cls in reversed_steps:
        step = cls(config)
        a = step.invert(a)

    return a


class CompressedData:
    def __init__(self, y, cb, cr):
        self.y = y
        self.cb = cb
        self.cr = cr


class Jpeg:
    def __init__(self, config):
        self.config = config

    def compress(self, image):
        y, cb, cr = image.split()
        res_y = compress_band(band_to_array(y), self.config)
        res_cb = compress_band(band_to_array(cb), self.config)
        res_cr = compress_band(band_to_array(cr), self.config)

        data = CompressedData(res_y, res_cb, res_cr)

        return file_format.generate_data(self.config, data)

    @staticmethod
    def decompress(bytestream):
        config, compressed_data = file_format.read_data(bytestream)
        size = (config.height, config.width)
        y = decompress_band(compressed_data.y, config)
        cb = decompress_band(compressed_data.cb, config)
        cr = decompress_band(compressed_data.cr, config)

        ycbcr = np.dstack(
            (y.reshape(size), cb.reshape(size), cr.reshape(size))
        ).astype(np.uint8)

        return Image.fromarray(np.asarray(ycbcr), mode='YCbCr')
