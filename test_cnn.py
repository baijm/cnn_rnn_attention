"""
script for testing CNN models
"""

import mxnet as mx
import numpy as np
import logging
import time
import os
import sys
import datetime
import argparse
import csv

import my_util
import my_constant


if __name__ == "__main__":

    """
    parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="test CNN"
    )
    # root dir of COCO-formed dataset
    parser.add_argument("--dataset_root_dir", required=True,
                        type=str,
                        help="Directory of the MS-COCO format dataset")
    # list of image names for testing (under dataset_root_dir)
    parser.add_argument("--test_iname2cid_file", required=True,
                        type=str,
                        help="list file of image_name -> cid used in testing")
    # whether to decrement ground truth cid
    parser.add_argument("--decrement_cid", required=False,
                        type=int,
                        default=1,
                        help="whether the ground truth cid should -1")

    # where to save results
    # results include :
    #   one .csv with each line (test_iname_wid, predicted_cid, score_for_cls0, score_for_cls1, ...)
    #   one .csv with each line (test_iname, predicted_cid, avg_score_for_cls0, avg_score_for_cls1, ...)
    parser.add_argument("--res_dir", required=True,
                        type=str,
                        help="Directory where summary files will be saved")
    # file prefix
    parser.add_argument("--res_prefix", required=True,
                        type=str,
                        help="prefix of summary files")

    # which model to load
    parser.add_argument("--model_prefix", required=True,
                        type=str,
                        help="prefix of the model used in testing")
    # which epoch to load
    parser.add_argument("--model_epoch", required=True,
                        type=str,
                        help="epoch of the model used in testing")

    # select gpu
    parser.add_argument("--gpu_id", required=False,
                        type=int,
                        default=0,
                        help="select gpu")

    args = parser.parse_args()

    dataset_root_dir = args.dataset_root_dir
    test_iname2cid_file = args.test_iname2cid_file
    decrement_cid = bool(args.decrement_cid)
    res_dir = args.res_dir
    res_prefix = args.res_prefix
    model_prefix = args.model_prefix
    model_epoch = int(args.model_epoch)
    gpu_id = args.gpu_id

    dataset_img_dir = os.path.join(dataset_root_dir, "gt_bbox", "rgb")


    """
    load iname2cid_file
    """
    print "####################\n# CONFIG\n####################"
    print "dataset_root_dir = {} ".format(dataset_root_dir),
    assert os.path.exists(dataset_root_dir), "directory not exist"
    print ""

    print "test_iname2cid_file = {} ".format(test_iname2cid_file),
    assert os.path.exists(os.path.join(dataset_root_dir, test_iname2cid_file)), "file not exist"
    print ""

    print "decrement_cid = {}".format(decrement_cid)

    print "res_dir = {}".format(res_dir),
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        print "\tcreated"
    else:
        print "\talready exists"
    print "res_prefix = {}".format(res_prefix)

    print "load {} ... ".format(test_iname2cid_file),
    iname2cid = {}
    with open(os.path.join(dataset_root_dir, test_iname2cid_file), 'r') as f:
        for line in f:
            parts = line.split()

            iname = parts[0]
            cid = int(parts[1])
            if decrement_cid:
                cid -= 1

            iname2cid.update({iname: cid})
    print "\tdone\n"


    """
    load model
    """
    print "####################\n# LOAD MODEL\n####################"
    print "gpu_id = {}".format(gpu_id)
    ctx = mx.gpu(gpu_id)

    print "model_prefix = {}".format(model_prefix)
    print "model_epoch = {}".format(model_epoch)

    print "load model ... \t",
    model = mx.model.FeedForward.load(
        prefix=model_prefix,
        epoch=model_epoch,
        ctx=ctx
    )

    print "done\n"


    """
    predict
    """
    print "####################\n# PREDICT\n####################"
    cid_list = sorted(list(set(iname2cid.values())))
    cid_list = [str(c) for c in cid_list]

    # result of each bbox
    bbox_res_file = os.path.join(res_dir, res_prefix + "_bbox.csv")
    br_f = open(bbox_res_file, 'w')
    br_fields = ['iname', 'gt_cid', 'pred_cid'] + cid_list
    br_writer = csv.DictWriter(br_f, fieldnames=br_fields)
    br_writer.writeheader()

    # for each test image
    for iname, cid in iname2cid.items():
        start_time = time.time()

        # load and preprocess image
        windows = my_util.preprocess(
            img_dir=dataset_img_dir,
            img_name=iname,
            pre_crop_resize_length=my_constant.RESIZE_SIDE,
            mean_pixel=my_constant.MEAN_PIXEL_INT,
            window_sizes=my_constant.WINDOW_SIZES,
            window_strides=my_constant.WINDOW_STRIDES,
            after_crop_resize_length=my_constant.INPUT_SIDE
        )

        # convert NHWC to NCHW
        windows = np.swapaxes(windows, 1, 3)
        windows = np.swapaxes(windows, 2, 3)

        # predict
        input = windows
        output = model.predict(input) # (num_windows, num_cls)

        num_windows = output.shape[0]
        avg_output = output.sum(axis=0) / num_windows
        pred_cid = np.argmax(avg_output)
        prob_dict = {}
        for c in cid_list:
            prob_dict.update({c : avg_output[int(c)]})

        print "{} : gt_cid = {}, pred_cid = {}".format(iname, cid, pred_cid),
        if pred_cid == cid:
            print "right",
        else:
            print "WRONG !",

        res_dict = {'iname' : iname, 'gt_cid' : cid, 'pred_cid' : pred_cid}
        res_dict.update(prob_dict)
        br_writer.writerow(res_dict)

        end_time = time.time()
        print "\t", end_time - start_time, "s"

    br_f.close()