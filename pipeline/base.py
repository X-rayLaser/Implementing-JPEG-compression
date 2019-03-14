from util import split_into_blocks, padded_size


step_classes = []


class Meta(type):
    @staticmethod
    def validate_index(cls, name, class_dict):
        attr_name = 'step_index'
        if attr_name not in class_dict:
            raise MissingStepIndexError(
                'Class {} has not defined "{}" class attribute'.format(
                    name, attr_name
                )
            )
        return

    @staticmethod
    def sort_classes():
        step_classes.sort(key=lambda cls: cls.step_index)

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)

        if name != 'AlgorithmStep':
            Meta.validate_index(cls, name, class_dict)
            step_classes.append(cls)
            Meta.sort_classes()

        return cls


class IndexOutOfOrderError(Exception):
    pass


class MissingStepIndexError(Exception):
    pass


class AlgorithmStep(metaclass=Meta):
    def __init__(self, config):
        self._config = config

    def execute(self, array):
        raise NotImplementedError

    def invert(self, array):
        raise NotImplementedError

    def calculate_padding(self, factor):
        w, h = self._config.width, self._config.height
        padded_width = padded_size(w, factor)
        padded_height = padded_size(h, factor)
        return padded_height - h, padded_width - w

    def blocks(self, a, block_size):
        blocks = split_into_blocks(a, block_size)

        h = a.shape[0] // block_size
        w = a.shape[1] // block_size

        for y in range(0, h):
            for x in range(w):
                yield blocks[y, x], y, x

    def apply_blockwise(self, a, transformation, block_size, res):
        for block, y, x in self.blocks(a, block_size):
            i = y * block_size
            j = x * block_size
            res[i:i + block_size, j: j + block_size] = transformation(block)
