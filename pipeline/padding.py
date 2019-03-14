from util import pad_array, undo_pad_array
from .base import AlgorithmStep


class Padding(AlgorithmStep):
    step_index = 0

    def execute(self, array):
        if self._config.block_size == 1:
            return array

        return pad_array(array, self._config.block_size)

    def invert(self, array):
        padding = self.calculate_padding(self._config.block_size)
        return undo_pad_array(array, padding)
