import numpy as np


class RoundingQuantizer:
    def quantize(self, a):
        return np.round(a)

    def restore(self, a):
        return a


class DiscardingQuantizer(RoundingQuantizer):
    def __init__(self, xkeep=2, ykeep=2):
        self.xkeep = xkeep
        self.ykeep = ykeep

    def quantize(self, a):
        res = np.round(a)
        res[self.ykeep:, self.xkeep:] = 0
        return res


class DivisionQuantizer(RoundingQuantizer):
    def __init__(self, divisor=40):
        self.divisor = divisor

    def quantize(self, a):
        return np.round(a / float(self.divisor))

    def restore(self, a):
        return a * self.divisor


class JpegQuantizationTable(RoundingQuantizer):
    table = [[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]]

    def __init__(self):
        self._qtable = np.array(self.table)

    def quantize(self, a):
        q = self._qtable
        return np.round(a * (1.0 / q))

    def restore(self, a):
        q = self._qtable
        return np.round(a * q)
