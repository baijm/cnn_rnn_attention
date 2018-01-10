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
import my_metric


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
    parser.add_argument("--train_iname2cid_file", required=True,
                        type=str,
                        help="list file of image_name -> cid used in training")
    # list of image names for testing (under dataset_root_dir)
    parser.add_argument("--test_iname2cid_file", required=True,
                        type=str,
                        help="list file of image_name -> cid used in testing")

    # where to save logs
    parser.add_argument("--log_dir", required=True,
                        type=str,
                        help="Directory to save .log files")
    # where to save checkpoints
    parser.add_argument("--ckpt_dir", required=True,
                        type=str,
                        help="Directory to save checkpoints")

    # whether cnn should be trained
    # if train_rnn == True, cnn will be trained together if train_cnn == True, otherwise cnn part will be fixed
    parser.add_argument("--train_cnn", required=True,
                        type=int,
                        help="train RNN+attention")
    # whether rnn should be trained
    parser.add_argument("--train_rnn", required=True,
                        type=int,
                        help="train CNN")

    # resume training or fine-tune public model
    parser.add_argument("--resume_training", required=True,
                        type=int,
                        help="resume training from specified epoch (1) or fine-tune public model (0)")
    # which checkpoint to continue from (needed if resume_training == 1)
    parser.add_argument("--from_epoch", required=False,
                        type=int,
                        default=-1,
                        help="which checkpoint to continue from (needed if resume_training == 1, default=-1(latest))")

    # public model (needed if resume_training == 0)
    parser.add_argument("--public_model_prefix", required=False,
                        type=str,
                        default="",
                        help="prefix of the public model (needed if resume_training == 0)")
    parser.add_argument("--public_model_epoch", required=False,
                        type=int,
                        default=0,
                        help="epoch of the public model (needed if resume_training == 0)")

    # number of different images in a batch
    parser.add_argument("--train_imgs_per_batch", required=False,
                        type=int,
                        default=my_constant.IMG_PER_BATCH,
                        help="number of different images in a batch for training")

    # number of epochs to train
    parser.add_argument("--num_epochs", required=False,
                        type=int,
                        default=100,
                        help="number of epochs to train")
    # learning_rate
    parser.add_argument("--learning_rate", required=False,
                        type=float,
                        default=my_constant.CNN_LEARNING_RATE,
                        help="learning rate in training")

    # select gpu
    parser.add_argument("--gpu_id", required=False,
                        type=int,
                        default=0,
                        help="select gpu")
    args = parser.parse_args()


    dataset_root_dir = args.dataset_root_dir
    train_iname2cid_file = args.train_iname2cid_file
    test_iname2cid_file = args.test_iname2cid_file

    log_dir = args.log_dir
    ckpt_dir = args.ckpt_dir

    resume_training = bool(args.resume_training)
    public_model_prefix = args.public_model_prefix
    public_model_epoch = args.public_model_epoch
    from_epoch = args.from_epoch

    train_imgs_per_batch = args.train_imgs_per_batch
    train_cnn = bool(args.train_cnn)
    train_rnn = bool(args.train_rnn)
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    gpu_id = args.gpu_id

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
    dataset_img_dir = os.path.join(dataset_root_dir, "gt", "by_bbox", "rgb")
    print "\twindow_path = {}".format(dataset_img_dir)
    dataset_name = dataset_root_dir.split('/')[-1]
    print "\tname = {}".format(dataset_name)
    dataset_num_cls = 200 if "CUB-200-2011" in dataset_name else 37
    print "\tnum_cls = {}".format(dataset_num_cls)

    # device
    ctx = mx.gpu(gpu_id)

    # training config
    print "training :"
    train_config = {
        "img_dir" : dataset_img_dir,
        "iname2cid_file": os.path.join(dataset_root_dir, train_iname2cid_file),
        "imgs_per_batch" : train_imgs_per_batch,
        "shuffle_inside_image" : True,
        "decrement_cid" : True
    }
    for k, v in train_config.items():
        print "\t", k, " = ", v

    # testing config
    print "testing :"
    test_config = {
        "img_dir": dataset_img_dir,
        "iname2cid_file": os.path.join(dataset_root_dir, test_iname2cid_file),
        "imgs_per_batch": train_imgs_per_batch, # 1,
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
    # at least one of CNN and RNN+attention should be trained
    assert train_cnn or train_rnn, "at least one of CNN and RNN+attention should be trained"
    if train_rnn:
        if train_cnn:
            ckpt_prefix += "_full"
        else:
            ckpt_prefix += "_rnn"
    else:
        ckpt_prefix += "_cnn"

    print "\ttrain_cnn = {}".format(train_cnn)
    print "\ttrain_rnn = {}".format(train_rnn)
    print "\tprefix = {}".format(ckpt_prefix)

    print "done\n"

    """
    symbol
    """
    print "####################\n# LOAD SYMBOL\n####################"
    if train_rnn:
        print "load CNN + RNN + attention symbol (train_cnn = {})".format(train_cnn)

        symbol = my_symbol.get_cnn_rnn_attention(
            num_cls=dataset_num_cls,
            for_training=True,
            rnn_dropout=my_constant.RNN_DROPOUT,
            rnn_hidden=my_constant.NUM_RNN_HIDDEN,
            rnn_window= my_constant.NUM_RNN_WINDOW,
            fix_till_relu7=not train_cnn
        )

        # code below runs
        #arg_shape, out_shape, aux_shape = symbol.infer_shape(
        #    data= (train_imgs_per_batch,
        #           my_constant.NUM_RNN_WINDOW * my_constant.INPUT_CHANNEL,
        #           my_constant.INPUT_SIDE, my_constant.INPUT_SIDE),
        #    rnn_l0_init_c=(train_imgs_per_batch, my_constant.NUM_RNN_HIDDEN),
        #    rnn_l0_init_h=(train_imgs_per_batch, my_constant.NUM_RNN_HIDDEN),
        #    att_gesture_softmax_label=(train_imgs_per_batch,),
        #    gesture_softmax_label=(train_imgs_per_batch,))
        #
        #arg_name =symbol.list_arguments()
        #
        #print 'output shape = ', out_shape, '\n'
        #
        #for name, shape in zip(arg_name, arg_shape):
        #    print name, ' : ', shape

    else:
        print "load CNN symbol"

        symbol = my_symbol.get_cnn(num_cls=dataset_num_cls,
                                   fix_till_relu7=not train_cnn)

    print "done\n"

    """
    data iter
    """
    print "####################\n# CREATE DATAITERS\n####################"
    print "train iter ... "
    train_iter = my_iter.MyIter(
        for_rnn=train_rnn,
        img_dir=train_config["img_dir"],
        iname2cid_file=train_config["iname2cid_file"],
        imgs_per_batch=train_config["imgs_per_batch"],
        shuffle_inside_image=train_config["shuffle_inside_image"],
        decrement_cid=train_config["decrement_cid"]
    )

    print "test iter ... "
    test_iter = my_iter.MyIter(
        for_rnn=train_rnn,
        img_dir=test_config["img_dir"],
        iname2cid_file=test_config["iname2cid_file"],
        imgs_per_batch=test_config["imgs_per_batch"],
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
            momentum=my_constant.MOMENTUM,
            wd=my_constant.WD,
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
            momentum=my_constant.MOMENTUM,
            wd=my_constant.WD,
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
        eval_data=test_iter,
        eval_metric=my_metric.Accuracy(0, 'g'),
        batch_end_callback=mx.callback.log_train_metric(50),
        epoch_end_callback=mx.callback.do_checkpoint(ckpt_dir + '/' + ckpt_prefix)
    )

    print "done\n"
