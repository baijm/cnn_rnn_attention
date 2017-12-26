"""
script for training on fine-grained datasets
"""

import mxnet as mx
import logging
import time
import os
import sys
import datetime
import my_symbol
import my_iter
import my_constant


if __name__ == "__main__":

    """
    set up logging
    """
    # timestamp
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-15s %(message)s',
                        filename=timestamp_str + '.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-15s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    """
    config
    """
    # dataset
    dataset_name = "CUB-200-2011-aug_coco"
    dataset_num_cls = 200 if "CUB-200-2011" in dataset_name else 37
    dataset_root_path = "/media/bjm/Data/datasets/fine_grained/CUB-200-2011-aug_coco"
    dataset_windows_path = os.path.join(dataset_root_path, "gt_bbox_windows")

    # device
    ctx = mx.gpu(0)

    # what to train
    train_rnn = False

    # training config
    train_config = {
        "windows_dir" : dataset_windows_path,
        "imgs_per_batch" : 1,
        "data_name" : "data",
        "label_name" : "softmax_label",
        "img_name_file" : os.path.join(dataset_root_path, "train_orig.txt"),
        "shuffle_inside_image" : True,
        "decrement_cid" : True
    }

    # testing config
    test_config = {
        "windows_dir": dataset_windows_path,
        "imgs_per_batch": 1,
        "data_name": "data",
        "label_name": "softmax_label",
        "img_name_file": os.path.join(dataset_root_path, "test.txt"),
        "shuffle_inside_image": False,
        "decrement_cid": True
    }

    # checkpoint
    ckpt_path = "/home/bjm/projects/cnn_rnn_attention/checkpoints"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt_prefix = dataset_name
    if train_rnn:
        ckpt_prefix += "_full"
    else:
        ckpt_prefix += "_cnn"


    """
    symbol
    """
    if train_rnn:
        # TODO : CNN + RNN + attention symbol
        symbol = my_symbol.get_cnn_rnn_attention(
            num_cls=dataset_num_cls,
            for_training=True,
            rnn_dropout=my_constant.RNN_DROPOUT,
            rnn_hidden=my_constant.NUM_RNN_HIDDEN,
            rnn_window= 32 # TODO
        )
    else:
        symbol = my_symbol.get_cnn(num_cls=dataset_num_cls)


    """
    data iter
    """
    train_iter = my_iter.MyIter(
        windows_dir=train_config["windows_dir"],
        imgs_per_batch=train_config["imgs_per_batch"],
        data_name=train_config["data_name"],
        label_name=train_config["label_name"],
        img_name_file=train_config["img_name_file"],
        shuffle_inside_image=train_config["shuffle_inside_image"],
        decrement_cid=train_config["decrement_cid"]
    )

    test_iter = my_iter.MyIter(
        windows_dir=test_config["windows_dir"],
        imgs_per_batch=test_config["imgs_per_batch"],
        data_name=test_config["data_name"],
        label_name=test_config["label_name"],
        img_name_file=test_config["img_name_file"],
        shuffle_inside_image=test_config["shuffle_inside_image"],
        decrement_cid=test_config["decrement_cid"]
    )


    """
    pre-trained model
    """
    public_model_prefix = "/home/bjm/pre-trained_models/vgg16"
    public_model_epoch = 0

    # load public model
    _, pretrained_arg_params, _ = mx.model.load_checkpoint(prefix=public_model_prefix,
                                                           epoch=public_model_epoch)


    """
    create model
    """
    model = mx.model.FeedForward(
        ctx=ctx,
        symbol=symbol,
        num_epoch=100,
        optimizer='sgd',
        learning_rate=0.001,
        initializer=mx.init.Load(
            param=pretrained_arg_params,
            default_init=mx.init.Xavier(factor_type="in", magnitude=2),
            verbose=True
        )
    )


    """
    train
    """
    model.fit(
        X=train_iter,
        eval_data=test_iter,
        eval_metric=['acc', 'ce'],
        batch_end_callback=mx.callback.log_train_metric(10),
        epoch_end_callback=mx.callback.do_checkpoint(ckpt_path + ckpt_prefix)
    )