import unittest
import sys
sys.path.insert(0, '../')

from util_tests import SplitIntoBlocksTests
from padding_tests import PaddingTests
from subsample_tests import SubsampleTests
from basis_change_tests import DctTests
from quantization_tests import QuantizersTests
from integration_tests import PipelineTests
from zigzag_tests import ZigzagOrderTests
from RLE_tests import RunLengthBlockTests, RunLengthEncodingTests


if __name__ == '__main__':
    unittest.main()
