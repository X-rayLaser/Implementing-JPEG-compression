import unittest
import sys
import numpy as np
sys.path.insert(0, '../')

from pipeline import Configuration
from pipeline.subsampling import SubSampling


class SubsampleTests(unittest.TestCase):
    def test_averaging(self):
        a = np.array([[1, 2, 2, 1],
                      [3, 2, 8, 1],
                      [0, 0, 2, 2],
                      [0, 4, 2, 2]])

        config = Configuration(width=123, height=854, block_size=2,
                               dct_size=2, transform='DCT', quantization=None)
        sub_sampling = SubSampling(config)
        res = sub_sampling.execute(a)
        self.assertEqual(res.shape, (2, 2))
        self.assertEqual(res[0][0], 2)
        self.assertEqual(res[0][1], 3)
        self.assertEqual(res[1][0], 1)
        self.assertEqual(res[1][1], 2)

        config = Configuration(width=123, height=854, block_size=4,
                               dct_size=2, transform='DCT', quantization=None)
        sub_sampling = SubSampling(config)
        res = sub_sampling.execute(a)
        self.assertEqual(res.shape, (1, 1))
        self.assertEqual(res[0][0], 2)
