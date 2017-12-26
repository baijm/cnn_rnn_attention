""" preprocess all images in a subset of a COCO-formed dataset, save generated windows
for each bbox RGB image (by Mask R-CNN/crop_by_bbox.py), do all the following preprocessing:
    1. resizing (long side, keep aspect ratio)
    2. padding (short side, with mean pixel)
    3. cropping & arranging windows
        (for each window size slide along x and y axes + one central crop)
        (large windows first, central windows first)
    4. resizing windows
    5. mean-subtracting
the generated windows will be arranged as a []
windows and their labels will be dumped in a .mat
"""

import os
import json
import scipy.io
import Image
import my_constant
import my_util

if __name__ == "__main__":
    """
    config (directory and file)
    """
    # root dir of COCO-formed dataset
    dataset_path = "/media/bjm/Data/datasets/fine_grained/CUB-200-2011-aug_coco"
    assert os.path.exists(dataset_path), "dataset_path {} not exist".format(dataset_path)
    print "dataset_path = {}".format(dataset_path)

    # bbox image dir
    bbox_img_path = os.path.join(dataset_path,
                                 "gt_bbox_img")
    assert os.path.exists(bbox_img_path), "bbox_img_path {} not exist".format(bbox_img_path)
    print "bbox_img_path = {}".format(bbox_img_path)

    # windows for bbox dir
    bbox_windows_path = os.path.join(dataset_path,
                                     "gt_bbox_windows")
    print "bbox_windows_path = {}".format(bbox_windows_path)
    if not os.path.exists(bbox_windows_path):
        os.makedirs(bbox_windows_path)

    # extension of files
    img_ext = ".jpg"

    """
    image info
    """
    # iid ->iname file
    iinfo_file = os.path.join("/media/bjm/Data/datasets/fine_grained/CUB-200-2011-aug",
                              "images.txt")
    assert os.path.exists(iinfo_file), "{} not exist".format(iinfo_file)
    print "reading {} ...".format(iinfo_file),
    iname2iid = {}
    # simplify image name (keep substr after '/')
    with open(iinfo_file, 'r') as file:
        for line in file:
            parts = line.split()
            assert len(parts) == 2, "error when reading images.txt : the number of parts != 2 after splitting by ' '"

            iid = int(parts[0])
            iname = parts[1].split('/')[1]
            iname2iid.update({iname : iid})
    print "done"

    # iid -> cid file
    # TODO : cid still start from 1 here. maybe -1 in DataIter will be better ?
    i2c_file = os.path.join("/media/bjm/Data/datasets/fine_grained/CUB-200-2011-aug",
                            "image_class_labels.txt")
    assert os.path.exists(i2c_file), "{} not exist".format(i2c_file)
    print "reading {} ...".format(i2c_file),
    iid2cid = {}
    with open(i2c_file, 'r') as file:
        for line in file:
            parts = line.split()
            assert len(parts) == 2, "error when reading image_class_labels.txt : the number of parts != 2 after splitting by ' '"

            iid = int(parts[0])
            cid = int(parts[1])
            iid2cid.update({iid : cid})
    print "done"

    """
    config (preprocessing)
    """
    # long side after resizing, before generating windows
    pre_crop_resize_side = my_constant.RESIZE_SIDE
    # size of sliding windows
    window_sizes = my_constant.WINDOW_SIZES
    # stride of sliding windows
    window_strides = my_constant.WINDOW_STRIDES
    # mean pixel value
    pix_mean = my_constant.MEAN_PIXEL_INT
    # side length of each window
    after_crop_resize_side = my_constant.INPUT_SIDE

    # write config file
    config_json_file = os.path.join(bbox_windows_path,
                                    "config.json")
    with open(config_json_file, 'w') as jf:
        json.dump(
            {
                "dataset_path" : dataset_path,
                "bbox_img_path" : bbox_img_path,
                "bbox_windows_path" : bbox_windows_path,
                "iinfo_file" : iinfo_file,
                "i2c_file" : i2c_file,
                "pre_crop_resize_side": pre_crop_resize_side,
                "window_sizes": window_sizes,
                "window_strides": window_strides,
                "pix_mean": pix_mean,
                "after_crop_resize_side": after_crop_resize_side
            },
            jf,
            sort_keys=True,
            indent=4
        )

    """
    load and preprocess each image
    """
    bbox_img_list = my_util.find_with_ext(bbox_img_path, img_ext)
    for i, iname in enumerate(bbox_img_list):
        print "process {} ({}/{})".format(iname, i, len(bbox_img_list))

        # load image
        img = Image.open(os.path.join(bbox_img_path, iname))
        img.save(os.path.join(bbox_windows_path, "after_load.jpg"), "jpeg")
        # class id
        cid = iid2cid[iname2iid[iname]]

        # step 1. resize
        img = my_util.resize(
            img=img,
            side_length=pre_crop_resize_side,
            by_long_side=True
        )

        # step 2. pad
        img = my_util.pad(
            img=img,
            side_length=pre_crop_resize_side,
            mean_pixel=pix_mean
        )

        # step 3. crop & arrange windows
        windows = my_util.gen_windows(
            img=img,
            window_sizes=window_sizes,
            window_strides=window_strides
        )

        # step 4. resize windows
        windows = my_util.resize(
            img=windows,
            side_length=after_crop_resize_side,
            by_long_side=True
        )

        # step 5. subtract mean (numpy.float32)
        windows = my_util.subtract_mean(
            img=windows,
            mean_pixel=pix_mean
        )

        # save to .mat
        res_mat_file = os.path.join(bbox_windows_path,
                                    iname.replace(img_ext, ".mat"))
        res_dict = {
            "preprocessed_windows" : windows, # make sure field name is shorter than 31 !!!
            "cid": cid
        }
        scipy.io.savemat(res_mat_file, res_dict, do_compression=True)