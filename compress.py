from PIL import Image
import numpy as np


def slow_average_band(band, block_size=3):
    pixels = np.array(list(band.getdata()))

    pixels = pixels.reshape((band.height, band.width))

    new_height = round(band.height / block_size)
    new_width = round(band.width / block_size)

    res = np.zeros((new_height, new_width))

    for y in range(new_height):
        for x in range(new_width):
            i = y * block_size
            j = x * block_size

            res[y, x] = np.mean(pixels[i:i + block_size, j:j + block_size])

    return res


def compress(input_fname, output_fname):
    im = Image.open(input_fname).convert('YCbCr')

    y, cb, cr = im.split()

    block_size = 3
    mean_cb = slow_average_band(cb, block_size=block_size)
    mean_cr = slow_average_band(cr, block_size=block_size)

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


compress('1.jpg', 'out.json')
