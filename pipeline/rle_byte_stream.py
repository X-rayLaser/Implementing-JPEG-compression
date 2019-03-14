from bitarray import bitarray
from .base import AlgorithmStep
from util import RunLengthCode


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
