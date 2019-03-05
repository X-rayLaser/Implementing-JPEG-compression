import json
import argparse
from PIL import Image
from util import band_to_array
from pipeline import compress_band, Configuration


def compress(input_fname, output_fname, block_size=2, dct_size=8, transform='DCT'):
    im = Image.open(input_fname).convert('YCbCr')

    y, cb, cr = im.split()

    config = Configuration(width=im.width, height=im.height,
                           block_size=block_size, dct_size=dct_size,
                           transform=transform, quantization='')

    res_y = compress_band(band_to_array(y), config)
    res_cb = compress_band(band_to_array(cb), config)
    res_cr = compress_band(band_to_array(cr), config)

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

    parser.add_argument('--transform', action='store', type=str, default='DCT',
                        help='type of discrete transform (DCT vs DFT)')

    args = parser.parse_args()

    compress(args.infile, args.outfile, block_size=args.block_size,
             dct_size=args.dct_size, transform=args.transform)
