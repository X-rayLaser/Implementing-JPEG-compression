import json
import numpy as np
from PIL import Image
from util import band_to_array
from quantizers import RoundingQuantizer, DiscardingQuantizer,\
    DivisionQuantizer, JpegQuantizationTable
import file_format
from . import padding, subsampling, dct_padding, basis_change, normalization,\
    quantization, zigzag_order, run_length_encoding, rle_byte_stream
from .base import step_classes


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
