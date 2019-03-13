import argparse
from pipeline import Jpeg


def decompress(input_path, output_path):
    with open(input_path, 'rb') as f:
        bytestream = f.read()

    reconstructed = Jpeg.decompress(bytestream)
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
