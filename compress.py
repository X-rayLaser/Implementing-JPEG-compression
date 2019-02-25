from PIL import Image
import numpy as np


def block_view(a, block_size):
    height = a.shape[0]
    width = a.shape[1]
    assert width % block_size == 0
    assert height % block_size == 0

    new_height = round(a.shape[0] // block_size)
    new_width = round(a.shape[1] // block_size)

    blocks = np.zeros((new_height, new_width, block_size, block_size), dtype=np.int)

    tmp_height = int(width * height / block_size)
    a = a.reshape((tmp_height, block_size))
    stride = width // block_size

    for j in range(stride):
        blocks_column = a[j::stride]

        for y in range(0, new_height):
            i = y * block_size
            blocks[y, j, :] = blocks_column[i:i + block_size]

    return blocks.reshape((new_height, new_width, block_size, block_size))


def average_band(band, block_size=3):
    pixels = np.array(list(band.getdata()))

    pixels = pixels.reshape((band.height, band.width))

    blocks_matrix = block_view(pixels, block_size)

    return np.mean(blocks_matrix, axis=(2, 3))


def compress(input_fname, output_fname, block_size=2):
    im = Image.open(input_fname).convert('YCbCr')

    y, cb, cr = im.split()

    mean_cb = average_band(cb, block_size=block_size)
    mean_cr = average_band(cr, block_size=block_size)

    import json

    d = {
        'width': im.width,
        'height': im.height,
        'block_size': block_size,
        'y': list(y.getdata()),
        'mean_cb': mean_cb.tolist(),
        'mean_cr': mean_cr.tolist()
    }

    s = json.dumps(d)
    with open(output_fname, 'w') as f:
        f.write(s)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )
    parser.add_argument('infile', type=str,
                        help='a path to the file to compress')

    parser.add_argument('outfile', type=str,
                        help='a destination path')

    parser.add_argument('--block_size', action='store', type=int, default=2,
                        help='size of sub-sampling block')

    args = parser.parse_args()

    compress(args.infile, args.outfile, block_size=args.block_size)
