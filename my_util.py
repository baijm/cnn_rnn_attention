""" functions for pre-processing
"""

import os
import sys
import numpy as np
from PIL import Image

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


def get_last_epoch(model_dir, prefix):
    """ get the last epoch with the specified prefix
    Parameters
    ----------------------------------
    model_dir : str
        where models (checkpoints) are saved
    prefix : str
        distinguish between models of different runs

    Return
    ----------------------------------
    int specifying epoch of the latest model, or -1 if there are no matching checkpoints
    """

    all_files = os.listdir(model_dir)
    model_epochs = []
    for f in all_files:
        if f.endswith(".params") and f.startswith(prefix):
            f = f.replace(".params", "")
            num = f.split("-")[-1]
            model_epochs.append(int(num))

    if len(model_epochs) == 0:
        return -1
    else:
        return max(model_epochs)


"""
called in prepare_data, get Image from each step
"""
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


"""
called by MyIter, generate windows by image file name
"""
def preprocess(img_dir, img_name, pre_crop_resize_length, mean_pixel, window_sizes, window_strides, after_crop_resize_length):
    """
    load an pre-process image(s) given by img_name, return pre-processed windows

    Parameters
    -----------------------------
    img_dir : str
        directory of cropped bbox images (RGB)
    img_name: str
    pre_crop_resize_length : int
        size of the long side after resizing
    mean_pixel : (int)
        in RGB order, used to pad resized image
    window_sizes : (int)
        large windows first
    window_strides : (int)
        in the same order as window_sizes
    after_crop_resize_length : int
        size of each output window

    Return:
        numpy.ndarray of size (num_windows, height, width, channel)
    -----------------------------
    """

    img_file = os.path.join(img_dir, img_name + ".jpg")
    assert os.path.exists(img_file), "image file {} not exist".img_file

    # load image
    img_orig = Image.open(img_file)

    # step 1. resize (numpy.ndarray, np.uint8)
    w_orig, h_orig = img_orig.size
    h2w = float(h_orig) / float(w_orig)
    if h_orig > w_orig:
        img_resized = img_orig.resize((int(pre_crop_resize_length / h2w), int(pre_crop_resize_length)))
    else:
        img_resized = img_orig.resize((int(pre_crop_resize_length), int(pre_crop_resize_length * h2w)))
    img_resized = np.array(img_resized)

    # step 2. pad (numpy.ndarray, np.uint8)
    assert isinstance(mean_pixel, tuple), "type(mean_pixel) must be tuple"
    assert img_resized.shape[2] == len(mean_pixel), "number of channels mismatch"

    h_resized, w_resized, c_resized = img_resized.shape
    if h_resized == pre_crop_resize_length:
        total_pad = pre_crop_resize_length - w_resized
    else:
        total_pad = pre_crop_resize_length - h_resized

    if total_pad == 0:
        img_padded = img_resized
    else:
        # create new image, fill with mean_pixel
        img_padded = np.zeros(
            (pre_crop_resize_length,
             pre_crop_resize_length,
             c_resized),
            dtype=np.uint8
        )
        for ci in range(len(mean_pixel)):
            img_padded[:, :, ci] = mean_pixel[ci]

        # put the original image in the middle
        half_pad = total_pad / 2
        if w_resized < pre_crop_resize_length:
            img_padded[0 : h_resized, half_pad : half_pad + w_resized, :] = img_resized
        else:
            img_padded[half_pad : half_pad + h_resized, 0 : w_resized, :] = img_resized

    # step 3. crop & arrange windows -> resize windows -> subtract mean
    # (get numpy.ndarray of numpy.float32 in the end)
    assert len(window_sizes) == len(window_strides), \
        "number of window_sizes({}) != number of window_strides({}) !".format(len(window_sizes),
                                                                                               len(window_strides))
    h_padded, w_padded, c_padded = img_padded.shape
    windows = []

    for size, stride in zip(window_sizes, window_strides):
        # for each size, generate the central crop first
        y_start = (h_padded - size) / 2
        x_start = (w_padded - size) / 2
        crop = img_padded[y_start : y_start + size, x_start : x_start + size, :]
        crop = Image.fromarray(crop)
        # resize
        crop_resized = crop.resize((after_crop_resize_length, after_crop_resize_length))
        # subtract mean
        crop_subtracted = np.array(crop_resized).astype(np.float32)
        for ci in range(len(mean_pixel)):
            crop_subtracted[:, :, ci] -= mean_pixel[ci]
        # add to result
        windows.append(crop_subtracted)

        # slide in 'Z'
        for yi in range((h_padded - size) / stride + 1):
            y_start = stride * yi
            for xi in range((w_padded - size) / stride + 1):
                x_start = stride * xi
                crop = img_padded[y_start : y_start + size, x_start : x_start + size, :]
                crop = Image.fromarray(crop)
                # resize
                crop_resized = crop.resize((after_crop_resize_length, after_crop_resize_length))
                # subtract mean
                crop_subtracted = np.array(crop_resized).astype(np.float32)
                for ci in range(len(mean_pixel)):
                    crop_subtracted[:, :, ci] -= mean_pixel[ci]
                # add to result
                windows.append(crop_subtracted)

    windows = np.array(windows)

    return windows


