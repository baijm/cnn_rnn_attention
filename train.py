"""
script for training on fine-grained datasets
"""

import mxnet as mx
import logging
import time
import os
import sys
import datetime
import argparse

import my_symbol
import my_iter
import my_constant
import my_util


if __name__ == "__main__":

    """
    parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="train CNN/CNN_RNN_attention"
    )
    # root dir of COCO-formed dataset
    parser.add_argument("--dataset_root_dir", required=True,
                        type=str,
                        help="Directory of the MS-COCO format dataset")
    # list of image names for training (under dataset_root_dir)
    parser.add_argument("--train_img_list", required=True,
                        type=str,
                        help="list file of image names used in training")
    # list of image names for testing (under dataset_root_dir)
    parser.add_argument("--test_img_list", required=True,
                        type=str,
                        help="list file of image names used in testing")

    # where to save logs
    parser.add_argument("--log_dir", required=True,
                        type=str,
                        help="Directory to save .log files")
    # where to save checkpoints
    parser.add_argument("--ckpt_dir", required=True,
                        type=str,
                        help="Directory to save checkpoints")

    # resume training or fine-tune public model
    parser.add_argument("--resume_training", required=True,
                        type=int,
                        help="resume traing from specified epoch (1) or fine-tune public model (0)")
    # public model (needed if resume_training == 0)
    parser.add_argument("--public_model_prefix", required=False,
                        type=str,
                        default="",
                        help="prefix of the public model (needed if resume_training == 0)")
    parser.add_argument("--public_model_epoch", required=False,
                        type=int,
                        default=0,
                        help="epoch of the public model (needed if resume_training == 0)")
    # which checkpoint to continue from (needed if resume_training == 1)
    parser.add_argument("--from_epoch", required=False,
                        type=int,
                        default=-1,
                        help="which checkpoint to continue from (needed if resume_training == 1, default=-1(latest))")

    # number of different images in a batch
    parser.add_argument("--train_imgs_per_batch", required=False,
                        type=int,
                        default=1,
                        help="number of different images in a batch for training")
    # what to train
    parser.add_argument("--train_rnn", required=True,
                        type=int,
                        help="train CNN_RNN_attention (1) or CNN only (0)")
    # number of epochs to train
    parser.add_argument("--num_epochs", required=False,
                        type=int,
                        default=100,
                        help="number of epochs to train")
    # learning_rate
    parser.add_argument("--learning_rate", required=False,
                        type=float,
                        default=0.0001,
                        help="learning rate in training")
    args = parser.parse_args()

    dataset_root_dir = args.dataset_root_dir
    train_img_list = args.train_img_list
    test_img_list = args.test_img_list

    log_dir = args.log_dir
    ckpt_dir = args.ckpt_dir

    resume_training = bool(args.resume_training)
    public_model_prefix = args.public_model_prefix
    public_model_epoch = args.public_model_epoch
    from_epoch = args.from_epoch

    train_imgs_per_batch = args.train_imgs_per_batch
    train_rnn = bool(args.train_rnn)
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate


    """
    set up logging
    """
    print "####################\n# SET UP LOGGING\n####################"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print "dir = {}".format(log_dir)

    # timestamp
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-15s %(message)s',
                        filename=os.path.join(log_dir, timestamp_str + '.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-15s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    print "done\n"

    """
    config
    """
    print "####################\n# CONFIG\n####################"
    # dataset
    print "dataset :"
    dataset_windows_path = os.path.join(dataset_root_dir, "gt_bbox_windows")
    print "\twindow_path = {}".format(dataset_windows_path)
    dataset_name = dataset_root_dir.split('/')[-1]
    print "\tname = {}".format(dataset_name)
    dataset_num_cls = 200 if "CUB-200-2011" in dataset_name else 37
    print "\tnum_cls = {}".format(dataset_num_cls)
    # device
    ctx = mx.gpu(0)

    # training config
    print "training :"
    train_config = {
        "windows_dir" : dataset_windows_path,
        "imgs_per_batch" : train_imgs_per_batch,
        "data_name" : "data",
        "label_name" : "softmax_label",
        "img_name_file" : os.path.join(dataset_root_dir, train_img_list),
        "shuffle_inside_image" : True,
        "decrement_cid" : True
    }
    for k, v in train_config.items():
        print "\t", k, " = ", v

    # testing config
    print "testing :"
    test_config = {
        "windows_dir": dataset_windows_path,
        "imgs_per_batch": 1,
        "data_name": "data",
        "label_name": "softmax_label",
        "img_name_file": os.path.join(dataset_root_dir, test_img_list),
        "shuffle_inside_image": False,
        "decrement_cid": True
    }
    for k, v in test_config.items():
        print "\t", k, " = ", v

    # checkpoint
    print "checkpoint :"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print "\tdir = {}".format(ckpt_dir)

    ckpt_prefix = dataset_name
    if train_rnn:
        ckpt_prefix += "_full"
    else:
        ckpt_prefix += "_cnn"
    print "\tprefix = {}".format(ckpt_prefix)

    print "done\n"

    """
    symbol
    """
    print "####################\n# LOAD SYMBOL\n####################"
    if train_rnn:
        print "load CNN + RNN + attention symbol"

        # TODO : CNN + RNN + attention symbol
        symbol, loss = my_symbol.get_cnn_rnn_attention(
            num_cls=dataset_num_cls,
            for_training=True,
            rnn_dropout=my_constant.RNN_DROPOUT,
            rnn_hidden=my_constant.NUM_RNN_HIDDEN,
            rnn_window= 32 # TODO
        )
    else:
        print "load CNN symbol"

        symbol = my_symbol.get_cnn(num_cls=dataset_num_cls)

    print "done\n"


    """
    data iter
    """
    print "####################\n# CREATE DATAITERS\n####################"
    print "train iter ... "
    train_iter = my_iter.MyIter(
        windows_dir=train_config["windows_dir"],
        imgs_per_batch=train_config["imgs_per_batch"],
        data_name=train_config["data_name"],
        label_name=train_config["label_name"],
        img_name_file=train_config["img_name_file"],
        shuffle_inside_image=train_config["shuffle_inside_image"],
        decrement_cid=train_config["decrement_cid"]
    )

    print "test iter ... "
    test_iter = my_iter.MyIter(
        windows_dir=test_config["windows_dir"],
        imgs_per_batch=test_config["imgs_per_batch"],
        data_name=test_config["data_name"],
        label_name=test_config["label_name"],
        img_name_file=test_config["img_name_file"],
        shuffle_inside_image=test_config["shuffle_inside_image"],
        decrement_cid=test_config["decrement_cid"]
    )

    print "done\n"


    """
    create model
    """
    print "####################\n# CREATE MODEL\n####################"
    # start from pre-trained model or resume training
    if resume_training:
        if from_epoch < 0:
            # get epoch number of the last model
            from_epoch = my_util.get_last_epoch(ckpt_dir + '/', ckpt_prefix)

        assert from_epoch >= 0

        # from specified epoch
        ckpt_model = mx.model.FeedForward.load(ckpt_dir + '/' + ckpt_prefix, from_epoch)
        model = mx.model.FeedForward(
            ctx=ctx,
            symbol=ckpt_model.symbol,
            num_epoch=num_epochs,
            optimizer='sgd',
            learning_rate=learning_rate,
            momentum=0.9,
            wd=0.0001,
            arg_params=ckpt_model.arg_params,
            aux_params=ckpt_model.aux_params,
            begin_epoch=from_epoch
        )
    else:
        # load public model
        assert len(public_model_prefix) != 0

        _, pretrained_arg_params, _ = mx.model.load_checkpoint(prefix=public_model_prefix,
                                                               epoch=public_model_epoch)
        model = mx.model.FeedForward(
            ctx=ctx,
            symbol=symbol,
            num_epoch=num_epochs,
            optimizer='sgd',
            learning_rate=learning_rate,
            momentum=0.9,
            wd=0.0001,
            initializer=mx.init.Load(
                param=pretrained_arg_params,
                default_init=mx.init.Xavier(factor_type="in", magnitude=2),
                verbose=True
            )
        )

    print "done\n"


    """
    train
    """
    print "####################\n# TRAIN\n####################"
    model.fit(
        X=train_iter,
        # eval_data=test_iter,
        eval_metric=['acc', 'ce'],
        batch_end_callback=mx.callback.log_train_metric(50),
        epoch_end_callback=mx.callback.do_checkpoint(ckpt_dir + '/' + ckpt_prefix)
    )

    print "done\n"