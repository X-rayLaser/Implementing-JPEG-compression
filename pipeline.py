import json
import numpy as np
from util import split_into_blocks, pad_array, undo_pad_array,\
    padded_size, inflate, Zigzag
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
        from util import RunLengthBlock

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

        from util import RunLengthBlock

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
        from util import RunLengthCode

        block = []
        for t in tuples_list:
            code = RunLengthCode(*t)
            block.append(code)
            if code.is_EOB():
                yield block
                block = []


class BitReader:
    def __init__(self, array):
        self._array = array
        self._pos = 0

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


class RleBytestream(AlgorithmStep):
    step_index = 8

    def execute(self, tuples_list):
        from util import RunLengthCode

        from bitarray import bitarray

        res = bitarray()

        for t in tuples_list:
            code = RunLengthCode(*t)
            if code.is_EOB():
                res.extend(bitarray('0' * 8))
                while len(res) % 8 > 0:
                    res.append(False)
            else:
                run_len_bits = self._to_bitstring(code.run_length)
                res.extend(self._pad_bitstring(run_len_bits))

                size_bits = self._to_bitstring(code.size)
                res.extend(self._pad_bitstring(size_bits))

                if not code.is_zeros_chain():
                    res.extend(self._to_bitstring(code.amplitude, signed=True))

        return res.tobytes()

    def invert(self, bytestream):
        from bitarray import bitarray
        a = bitarray()
        a.frombytes(bytestream)
        tup_list = []

        for code in self._codes(a):
            tup_list.append(code.as_tuple())

        return tup_list

    def _codes(self, bits):
        from util import RunLengthCode

        reader = BitReader(bits)
        while not reader.is_end():
            run_len = self._decode(reader.read_quad())

            size = self._decode(reader.read_quad())

            if run_len == 0 and size == 0:
                reader.skip_padding()
                yield RunLengthCode.EOB()
                continue

            if run_len == 15 and size == 0:
                yield RunLengthCode(15, 0, 0)
                continue

            amplitude = self._decode(reader.read(size), signed=True)
            yield RunLengthCode(run_len, size, amplitude)

    def _decode(self, bits, signed=False):
        if signed:
            abs_val = int(bits.to01()[1:], base=2)

            negative = bits.to01()[0] == '0'
            if negative:
                return -abs_val
            else:
                return abs_val

        return int(bits.to01(), base=2)

    def _to_bitstring(self, x, signed=False):
        from bitarray import bitarray

        s = bin(abs(x))[2:]

        if signed:
            s = '1' + s if x > 0 else '0' + s

        return bitarray(s)

    def _pad_bitstring(self, bits, size=4):
        from bitarray import bitarray

        while len(bits) < size:
            bits = bitarray('0') + bits
        return bits


def compress_band(a, config):
    for cls in step_classes:
        step = cls(config)
        a = step.execute(a)

    return CompressionResult(a, config)


def decompress_band(compression_result):
    a = compression_result.data

    reversed_steps = list(step_classes)
    reversed_steps.reverse()
    for cls in reversed_steps:
        step = cls(compression_result.config)
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
        from util import band_to_array
        res_y = compress_band(band_to_array(y), self.config)
        res_cb = compress_band(band_to_array(cb), self.config)
        res_cr = compress_band(band_to_array(cr), self.config)

        data = CompressedData(res_y, res_cb, res_cr)

        return file_format.generate_data(self.config, data)

    @staticmethod
    def decompress(bytestream):
        config, compressed_data = file_format.read_data(bytestream)
        size = (config.height, config.width)
        y = decompress_band(compressed_data.y)
        cb = decompress_band(compressed_data.cb)
        cr = decompress_band(compressed_data.cr)

        ycbcr = np.dstack(
            (y.reshape(size), cb.reshape(size), cr.reshape(size))
        ).astype(np.uint8)

        from PIL import Image
        return Image.fromarray(np.asarray(ycbcr), mode='YCbCr')


class ArraySerializer:
    @staticmethod
    def serialize(a):
        return {
            'values': a
        }

    @staticmethod
    def deserialize(d):
        return d['values']


class ComplexListSerializer:
    @staticmethod
    def serialize(complex_tuples):
        values = []
        for t in complex_tuples:
            if len(t) == 3:
                run_length, size, value = t
                d = {
                    'real': np.real(value),
                    'imag': np.imag(value)
                }
                values.append((run_length, size, d))
            else:
                values.append(t)
        return values

    @staticmethod
    def deserialize(tuples_list):
        values = []

        for t in tuples_list:
            if len(t) == 3:
                run_length, size, d = t
                complex_value = np.complex(d['real'], d['imag'])
                values.append((run_length, size, complex_value))
            else:
                values.append(t)
        return values


class CompressionResult:
    def __init__(self, data, config):
        self.data = data
        self.config = config

    @staticmethod
    def get_serializer(transform):
        if transform == 'DCT':
            return ArraySerializer
        else:
            return ComplexListSerializer

    def as_dict(self):
        serializer = self.get_serializer(self.config.transform)
        data = serializer.serialize(self.data)

        return {
            'data': data,
            'block_size': self.config.block_size,
            'dct_block_size': self.config.dct_size,
            'transform': self.config.transform,
            'width': self.config.width,
            'height': self.config.height,
            'quantization_name': self.config.quantization.name,
            'quantization_params': self.config.quantization.params
        }

    @staticmethod
    def from_dict(d):
        block_size = d['block_size']
        dct_block_size = d['dct_block_size']
        transform_type = d['transform']
        width = d['width']
        height = d['height']

        serializer = CompressionResult.get_serializer(transform_type)
        data = serializer.deserialize(d['data'])

        quantization = QuantizationMethod(d['quantization_name'],
                                          **d['quantization_params'])

        config = Configuration(width=width, height=height,
                               block_size=block_size, dct_size=dct_block_size,
                               transform=transform_type,
                               quantization=quantization)
        return CompressionResult(data, config)
