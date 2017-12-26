""" functions for pre-processing
"""

import os
import sys
import numpy as np
import Image

def find_with_ext(dir, ext):
    """ return all file names (without extension) in a dir with ext
    :param dir: str, path
    :param ext: str, extension to be searched for
    :return: [str], file names with extension
    """

    items = os.listdir(dir)
    file_names = []
    for name in items:
        if name.endswith(ext):
            file_names.append(name)

    return file_names


def resize(img, side_length, by_long_side=True):
    """ keep aspect ratio and resize the side specified to the given length
    :param img: one image or a list of images (loaded by Image.open())
    :param side_length: int
    :param by_long_side: bool, resize which side
    :return: resized image. in the same form as img
    """

    if isinstance(img, list):
        res = []
        for im in img:
            w, h = im.size
            h2w = float(h) / float(w)

            if by_long_side:
                if h > w:
                    curr_res = im.resize((int(side_length / h2w), int(side_length)))
                else:
                    curr_res = im.resize((int(side_length), int(side_length * h2w)))
            else:
                print "resize by short side not supported"
                sys.exit(-1)

            res.append(curr_res)
    else:
        w, h = img.size
        h2w = float(h) / float(w)

        if by_long_side:
            if h > w:
                res = img.resize((int(side_length / h2w), int(side_length)))
            else:
                res = img.resize((int(side_length), int(side_length * h2w)))
        else:
            print "resize by short side not supported"
            sys.exit(-1)

    return res


def pad(img, side_length, mean_pixel):
    """ pad input image with mean pixel, make the shorter side the same as the the longer side
    should be called after resize() !!!
    :param img: format is same as returned by Image.open()
    :param mean_pixel: tuple, mean pixel value in RGB order
    :return: padded image. format is same as returned by Image.open()
    """

    assert isinstance(mean_pixel, tuple), "type(mean_pixel) must be tuple"

    # check image mode
    mode = img.mode
    assert mode == "RGB", "image mode is not RGB"

    # check image shape
    w, h = img.size
    assert h <= side_length and w <= side_length, "h({}) or w({}) exceeds side_length({}) !".format(h, w, side_length)
    assert h == side_length or w == side_length, "call resize() before pad() !"

    # compute padding of the shorter side
    if h == side_length:
        total_pad = side_length - w
    else:
        total_pad = side_length - h

    # if nothing to pad, return
    if total_pad == 0:
        return img
    else:
        img = np.array(img)

        # create new image, fill with mean_pixel
        # TODO : use interger mean_pixel now, change to float ?
        res = np.zeros((side_length, side_length, len(mode)), dtype=np.uint8)
        res[:, :, 0] = mean_pixel[0]  # R
        res[:, :, 1] = mean_pixel[1]  # G
        res[:, :, 2] = mean_pixel[2]  # B

        # put the original image in the middle
        half_pad = total_pad / 2
        if w < side_length:
            res[0 : h, half_pad : half_pad + w, :] = img
        else:
            res[half_pad : half_pad + h, 0 : w, :] = img

        return Image.fromarray(res)


def subtract_mean(img, mean_pixel):
    """ subtract mean pixel
    :param img: one image or a list of images (loaded by Image.open())
    :param mean_pixel: tuple, mean pixel value in RGB order
    :return: mean-subtracted image (numpy.float32)
    """

    assert isinstance(mean_pixel, tuple), "type(mean_pixel) must be tuple"

    if isinstance(img, list):
        res = []
        for im in img:
            # check image mode
            mode = im.mode
            assert mode == "RGB", "image mode is not RGB"

            # convert to numpy.ndarray
            im = np.array(im)
            # convert data type
            im = im.astype(np.float32)

            for ci in range(len(mode)):
                im[:, :, ci] -= mean_pixel[ci]

            res.append(im)
    else:
        # check image mode
        mode = img.mode
        assert mode == "RGB", "image mode is not RGB"

        # convert to numpy.ndarray
        img = np.array(img)
        # convert data type
        img = img.astype(np.float32)

        for ci in range(len(mode)):
            img[:, :, ci] -= mean_pixel[ci]

        res = img
    return res


def gen_windows(img, window_sizes, window_strides, crop_central=True):
    """ crop windows from an image
    :param img: format is same as returned by Image.open()
    :param window_sizes: tuple of int, lengths of side (large windows first)
    :param window_strides: tuple of int, strides along x and y axes (in the same order as window_sizes)
    :param crop_central: bool, keep central crop of each window size
    :return: [format is same as returned by Image.open()]
    """

    assert len(window_sizes) == len(window_strides), "number of window_sizes({}) != number of window_strides({}) !".format(len(window_sizes), len(window_strides))

    img = np.array(img)
    h, w, c = img.shape

    res = []
    for size, stride in zip(window_sizes, window_strides):
        # for each size, generate the central crop first (if needed)
        if crop_central:
            y_start = (h - size) / 2
            x_start = (w - size) / 2
            crop = img[y_start : y_start + size, x_start : x_start + size, :]
            res.append(Image.fromarray(crop))

        # slide in 'Z'
        for yi in range((h - size) / stride + 1):
            y_start = stride * yi
            for xi in range((w - size) / stride + 1):
                x_start = stride * xi
                crop = img[y_start : y_start + size, x_start : x_start + size, :]
                res.append(Image.fromarray(crop))

    return res