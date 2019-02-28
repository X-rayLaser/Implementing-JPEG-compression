import json
import argparse
import numpy as np
from PIL import Image
from pipeline import decompress_band, CompressionResult


def decompress(input_path, output_path):
    with open(input_path, 'r') as f:
        s = f.read()

    d = json.loads(s)
    size = (d['height'], d['width'])

    y = decompress_band(
        CompressionResult.from_dict(d['Y'])
    )
    cb = decompress_band(
        CompressionResult.from_dict(d['Cb'])
    )
    cr = decompress_band(
        CompressionResult.from_dict(d['Cr'])
    )

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
