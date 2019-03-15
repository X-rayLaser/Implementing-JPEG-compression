# Intro

The main purpose of this repository was to better understand JPEG
compression algorithm. The implementation of the
compression and decompression presented here is inefficient and far from
official JPEG standard. Therefore, it is not designed to be used for production
environment.

As of the time of this writing, the following compression steps are implemented:
subsampling, Discrete Fourier Transform/Discrete Cosine Transform, 
Quantization and Run Length Encoding.

Feel free to use it for experimenting, inspiration or fun!

# Requirements

The program was tested only for Python 3.6.7

# Installation and launch


Clone the repository or download it as a zip archive.
Open a terminal.

Go inside the repository git directory (the one containing a .git folder and 
requirements.txt file)
```
    cd /path/to/repo
```
Go inside the repository git directory and install project's dependencies.
```
    pip install -r requirements.txt
```
Compress the file with default options:
```
    python compress.py in.png out
```

Decompress the file:
```
    python decompress.py out reconstructed.png
```

Compress the file with subsampling factor = 5, size of DCT block = 8,
and use Discrete Cosine Transform:
```
    python compress.py in.png out --block_size 5 --dct_size 8 --transform DCT
```

Compress the file with subsampling factor = 5, size of DCT block = 24,
divide each DCT quotient by 1000 (Huge compression rate):
```
    python compress.py in.png out --block_size 5 --dct_size 24 --quantization divide --qdivisor 1000
```

# License
This software is licensed under GPL v3 license (see LICENSE).

## Third party libraries licenses
The software uses third party libraries that are distributed under 
their own terms (see LICENSE-3RD-PARTY).
