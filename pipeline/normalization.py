from .base import AlgorithmStep


class Normalization(AlgorithmStep):
    step_index = 3

    def execute(self, array):
        return array

    def invert(self, array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i][j] = max(0, min(255, array[i][j]))
        return array
