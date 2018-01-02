"""
script for testing RNN models
"""

import mxnet as mx
import numpy as np
import scipy.io
import time
import os
import sys
import datetime
import argparse
import csv

import my_util
import my_constant
import my_symbol
import my_iter


if __name__ == "__main__":

    """
    parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="test RNN"
    )
    # root dir of COCO-formed dataset
    parser.add_argument("--bbox_dir", required=True,
                        type=str,
                        help="Directory of cropped bounding boxes, including 'rgb' and 'mask' subdirs")
    # list of image names for testing (under bbox_dir)
    parser.add_argument("--iname2cid_file", required=True,
                        type=str,
                        help="list file of image_name -> cid used in testing")
    # whether to decrement ground truth cid
    parser.add_argument("--decrement_cid", required=False,
                        type=int,
                        default=1,
                        help="whether the ground truth cid should -1")

    # where to save results
    # results include :
    #   one .csv with each line (test_iname, predicted_cid, avg_score_for_cls0, avg_score_for_cls1, ...)
    #   one .mat with confusion matrix
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

    bbox_dir = args.bbox_dir
    iname2cid_file = args.iname2cid_file
    decrement_cid = bool(args.decrement_cid)
    model_prefix = args.model_prefix
    model_epoch = int(args.model_epoch)
    res_dir = args.res_dir
    res_prefix = args.res_prefix + '_' + str(model_epoch)
    gpu_id = args.gpu_id


    """
    load iname2cid_file, check directories, set up context
    """
    print "####################\n# CONFIG\n####################"
    print "bbox = {} ".format(bbox_dir),
    assert os.path.exists(bbox_dir), "directory not exist"
    print ""

    bbox_img_dir = os.path.join(bbox_dir, "rgb")
    assert os.path.exists(bbox_img_dir), "'rgb' subdirectory not exist"

    print "iname2cid_file = {} ".format(iname2cid_file),
    assert os.path.exists(iname2cid_file), "file not exist"
    print ""

    # bbox_dir and iname2cid_file should be in the same dir
    assert os.path.dirname(bbox_dir) == os.path.dirname(iname2cid_file), "bbox_dir and iname2cid_file should be in the same directory"

    num_cls = 200 if "CUB-200-2011" in bbox_dir else 37
    print "{} classes".format(num_cls)

    print "decrement_cid = {}".format(decrement_cid)

    # load iname2cid
    iname2cid = {}
    with open(iname2cid_file, 'r') as f:
        for line in f:
            parts = line.split()

            iname = parts[0]
            cid = int(parts[1])

            if decrement_cid:
                cid -= 1

            iname2cid.update({iname : cid})

    print "res_dir = {}".format(res_dir),
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        print "\tcreated"
    else:
        print "\talready exists"
    print "res_prefix = {}".format(res_prefix)

    print "gpu_id = {}".format(gpu_id)
    ctx = mx.gpu(gpu_id)


    """
    load symbol
    """
    print "####################\n# LOAD SYMBOL\n####################"
    symbol = my_symbol.get_cnn_rnn_attention(
        num_cls=num_cls,
        for_training=False,
        rnn_dropout=my_constant.RNN_DROPOUT,
        rnn_hidden=my_constant.NUM_RNN_HIDDEN,
        rnn_window=my_constant.NUM_RNN_WINDOW
    )
    print "done\n"
    print symbol.list_arguments()
    print symbol.list_outputs()


    """
    load model
    """
    print "####################\n# LOAD MODEL\n####################"
    print "model_prefix = {}".format(model_prefix)
    print "model_epoch = {}".format(model_epoch)

    print "load model ... \t",
    _, arg_params, aux_params = mx.model.load_checkpoint(prefix=model_prefix, epoch=model_epoch)


    print "done\n"


    """
    bind
    """
    print "####################\n# CREATE MODULE\n####################"
    data_shapes = []
    init_state_arrays = []
    data_shapes.append(('data',
                        (1,
                         my_constant.NUM_RNN_WINDOW * my_constant.INPUT_CHANNEL,
                         my_constant.INPUT_SIDE,
                         my_constant.INPUT_SIDE)))
    for i in range(my_constant.NUM_RNN_LAYER):
        data_shapes.append(('rnn_l%d_init_c' % i,
                            (1, my_constant.NUM_RNN_HIDDEN)))
        init_state_arrays.append(mx.nd.zeros((1, my_constant.NUM_RNN_HIDDEN)))
        data_shapes.append(('rnn_l%d_init_h' % i,
                            (1, my_constant.NUM_RNN_HIDDEN)))
        init_state_arrays.append(mx.nd.zeros((1, my_constant.NUM_RNN_HIDDEN)))

    mod = mx.mod.Module(symbol=symbol,
                        data_names=[d[0] for d in data_shapes],
                        label_names=None,
                        context=ctx)
    mod.bind(for_training=False,
             data_shapes=data_shapes)
    mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)
    print "done\n"


    """
    predict
    """
    print "####################\n# PREDICT\n####################"
    cid_list = range(num_cls)
    cid_list = [str(c) for c in cid_list]

    # result of each bbox
    br_file = os.path.join(res_dir, res_prefix + "_bbox.csv")
    br_f = open(br_file, 'w')
    br_fields = ['iname', 'gt_cid', 'pred_cid'] + cid_list
    br_writer = csv.DictWriter(br_f, fieldnames=br_fields)
    br_writer.writeheader()

    # confusion matrix
    conf_mat = np.zeros((num_cls, num_cls))

    # for each test image
    for iname, gt_cid in iname2cid.items():
        start_time = time.time()

        # load and preprocess image
        windows = my_util.preprocess(
            img_dir=bbox_img_dir,
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
        windows = windows.reshape(1, -1, windows.shape[2], windows.shape[3])
        windows = [mx.nd.array(windows)]
        input = windows + init_state_arrays

        mod.forward(mx.io.DataBatch(data=input))

        output = mod.get_outputs()[0].asnumpy()

        pred_cid = np.argmax(output)
        prob_dict = {}
        for c in cid_list:
            prob_dict.update({c : output[0, int(c)]})

        print "{} :\tgt_cid = {}\tpred_cid = {}\t".format(iname, gt_cid, pred_cid),
        if pred_cid == gt_cid:
            print "right",
        else:
            print "WRONG !",

        # write csv file
        res_dict = {'iname' : iname, 'gt_cid' : gt_cid, 'pred_cid' : pred_cid}
        res_dict.update(prob_dict)
        br_writer.writerow(res_dict)

        # add to confusion matrix
        conf_mat[gt_cid][pred_cid] += 1

        end_time = time.time()
        print "\t", end_time - start_time, "s"

    br_f.close()

    # save confusion matrix
    cm_file = os.path.join(res_dir, res_prefix + "_conf_matrix.mat")
    scipy.io.savemat(cm_file, {'confusion_matrix' : conf_mat})