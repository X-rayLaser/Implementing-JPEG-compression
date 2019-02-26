import json
import argparse
import numpy as np
from PIL import Image


def inflate(a, factor):
    """
    Reverses a sub-sampling procedure for a given array

    :param a: instance of numpy.ndarray
    :param factor: size of sub-sampling block
    :return: new instance of numpy.ndarray
    """
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)


def decompress(input_path, output_path):
    with open(input_path, 'r') as f:
        s = f.read()

    d = json.loads(s)

    block_size = d['block_size']

    y = np.array(d['y'])

    cb = inflate(
        np.array(d['mean_cb']), block_size
    )
    cr = inflate(
        np.array(d['mean_cr']), block_size
    )

    size = (d['height'], d['width'])

    ycbcr = np.dstack(
        (y.reshape(size),cb.reshape(size), cr.reshape(size))
    ).astype(np.uint8)

    reconstructed = Image.fromarray(np.asarray(ycbcr), mode='YCbCr')
    reconstructed.save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )
    parser.add_argument('infile', type=str,
                        help='a path to the file to compress')

    parser.add_argument('outfile', type=str,
                        help='a destination path')

    args = parser.parse_args()

    decompress(args.infile, args.outfile)
