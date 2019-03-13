import argparse
from PIL import Image
from pipeline import Configuration, QuantizationMethod, Jpeg


def compress(input_fname, output_fname, block_size=2, dct_size=8,
             transform='DCT', quantization=None):

    im = Image.open(input_fname).convert('YCbCr')

    config = Configuration(width=im.width, height=im.height,
                           block_size=block_size, dct_size=dct_size,
                           transform=transform, quantization=quantization
                           )

    jpeg = Jpeg(config)
    compressed_bytes = jpeg.compress(im)

    with open(output_fname, 'wb') as f:
        f.write(compressed_bytes)


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

    parser.add_argument('--quantization', action='store', type=str, default='none',
                        help='type of quantization to use: on of none, discard, divide, qtable ')

    parser.add_argument('--qkeep', action='store', type=int, default=2,
                        help='specifies how many coefficients to keep along both axes,'
                             'applied only if quantization is set to "discard"')

    parser.add_argument('--qdivisor', action='store', type=int, default=40,
                        help='specifies an integer used to divide coefficients by,'
                             'applied only if quantization is set to "divide"')

    args = parser.parse_args()

    if args.quantization == 'discard':
        quant_method = QuantizationMethod('discard', keep=args.qkeep)
    elif args.quantization == 'divide':
        quant_method = QuantizationMethod('divide', divisor=args.qdivisor)
    elif args.quantization == 'qtable':
        quant_method = QuantizationMethod('qtable')
    else:
        quant_method = None

    compress(args.infile, args.outfile, block_size=args.block_size,
             dct_size=args.dct_size, transform=args.transform,
             quantization=quant_method)
