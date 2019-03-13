import struct
import pipeline


class Reader:
    def __init__(self, seq):
        self._seq = seq
        self._index = 0

    def read_short(self):
        return self.read(2)

    def read_long(self):
        return self.read(4)

    def read(self, n):
        chunk = self._seq[self._index:self._index + n]
        self._index += n
        return chunk


def get_header(bytestream):
    reader = Reader(bytestream)

    header_length = unpack_integer(reader.read_short())
    width = unpack_integer(reader.read_short())

    height = unpack_integer(reader.read_short())
    block_size = unpack_integer(reader.read_short())
    dct_size = unpack_integer(reader.read_short())
    transform = unpack_string(reader.read(3))

    quantization_length = unpack_integer(reader.read_short())
    quant_json = unpack_string(reader.read(quantization_length))
    quantization = pipeline.QuantizationMethod.from_json(quant_json)

    return pipeline.Configuration(width=width, height=height,
                                  block_size=block_size,
                                  dct_size=dct_size, transform=transform,
                                  quantization=quantization)


def pack_integer(value):
    return struct.pack('<H', value)


def unpack_integer(bytestream):
    return struct.unpack('<H', bytestream)[0]


def pack_long(value):
    return struct.pack('<L', value)


def unpack_long(bytestream):
    return struct.unpack('<L', bytestream)[0]


def pack_string(s):
    return bytes(s, encoding='ascii')


def unpack_string(bytestream):
    return bytestream.decode()


def create_header(config):
    quantization_method = config.quantization

    quantization_json = quantization_method.to_json()
    quantization_length = len(quantization_json)

    header_length = 2 + 13 + quantization_length
    out = pack_integer(header_length) + \
          pack_integer(config.width) + \
          pack_integer(config.height) + \
          pack_integer(config.block_size) + \
          pack_integer(config.dct_size) + \
          pack_string(config.transform) + \
          pack_integer(quantization_length) + \
          pack_string(quantization_json)

    return out


def generate_data(config, compressed_data):
    y_len = pack_long(len(compressed_data.y))
    cb_len = pack_long(len(compressed_data.cb))
    cr_len = pack_long(len(compressed_data.cr))

    header_bytes = create_header(config)
    return header_bytes + y_len + compressed_data.y + \
           cb_len + compressed_data.cb + cr_len + compressed_data.cr


def read_data(bytestream):
    reader = Reader(bytestream)
    config = get_header(bytestream)
    header_length = unpack_integer(reader.read_short())
    reader.read(header_length - 2)

    y_len = unpack_long(reader.read_long())
    y = reader.read(y_len)

    cb_len = unpack_long(reader.read_long())
    cb = reader.read(cb_len)

    cr_len = unpack_long(reader.read_long())
    cr = reader.read(cr_len)

    return config, pipeline.CompressedData(y, cb, cr)
