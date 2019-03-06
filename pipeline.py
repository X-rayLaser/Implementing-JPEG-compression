import numpy as np
from util import split_into_blocks, pad_array, undo_pad_array,\
    padded_size, inflate
from transforms import DCT
from quantizers import RoundingQuantizer, DiscardingQuantizer,\
    DivisionQuantizer, JpegQuantizationTable


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

    def apply_blockwise(self, a, transformation, block_size, res):
        blocks = split_into_blocks(a, block_size)

        h = a.shape[0] // block_size
        w = a.shape[1] // block_size

        for y in range(0, h):
            for x in range(w):
                i = y * block_size
                j = x * block_size
                block = transformation(blocks[y, x])
                res[i:i + block_size, j: j + block_size] = block


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


class CompressionResult:
    def __init__(self, data, config):
        self.data = data
        self.config = config

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
        if d['data']['dtype'] == 'complex':
            data = CompressionResult._reconstruct_complex(d['data']['values'])
        else:
            data = np.array(d['data']['values'])

        block_size = d['block_size']
        dct_block_size = d['dct_block_size']
        transform_type = d['transform']
        width = d['width']
        height = d['height']

        quantization = QuantizationMethod(d['quantization_name'],
                                          **d['quantization_params'])

        config = Configuration(width=width, height=height,
                               block_size=block_size, dct_size=dct_block_size,
                               transform=transform_type,
                               quantization=quantization)
        return CompressionResult(data, config)
