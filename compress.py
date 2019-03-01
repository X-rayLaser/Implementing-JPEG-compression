import json
import argparse
from PIL import Image
from util import band_to_array
from pipeline import compress_band


def compress(input_fname, output_fname, block_size=2, dct_size=8):
    im = Image.open(input_fname).convert('YCbCr')

    y, cb, cr = im.split()

    res_y = compress_band(band_to_array(y), block_size=1, dct_size=dct_size)
    res_cb = compress_band(band_to_array(cb), block_size, dct_size)
    res_cr = compress_band(band_to_array(cr), block_size, dct_size)

    d = {
        'width': im.width,
        'height': im.height,
        'Y': res_y.as_dict(),
        'Cb': res_cb.as_dict(),
        'Cr': res_cr.as_dict()
    }

    s = json.dumps(d)
    with open(output_fname, 'w') as f:
        f.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )
    parser.add_argument('infile', type=str,
                        help='a path to the file to compress')

    parser.add_argument('outfile', type=str,
                        help='a destination path')

    parser.add_argument('--block_size', action='store', type=int, default=2,
                        help='size of sub-sampling block')

    parser.add_argument('--dct_size', action='store', type=int, default=8,
                        help='size of block for DCT transform')

    args = parser.parse_args()

    compress(args.infile, args.outfile, block_size=args.block_size)
