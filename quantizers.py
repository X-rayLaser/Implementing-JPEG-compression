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


class ModuloQuantizer(RoundingQuantizer):
    def __init__(self, divisor=40):
        self.divisor = divisor

    def quantize(self, a):
        return np.round(a / float(self.divisor))

    def restore(self, a):
        return a * self.divisor
