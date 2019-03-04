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
