import mxnet as mx
import numpy as np
import os
import sys
import random
import scipy.io
import json
import my_constant
import my_util


class MyIter(mx.io.DataIter):
    """ take a COCO-formed dataset, get data iterator for training CNN and RNN
    Parameters
    ----------------
    for_rnn : bool
        if this iterator will be used to train / test CNN+RNN+attention model
    img_dir : str
        dir of RGB images.
    iname2cid_file : str
        .txt file with each line 'image_name cid'
    img_per_batch : int
        number of different images per batch
    shuffle_inside_image : bool
        shuffle windows of the same scale within each image (for training)
    decrement_cid : bool
        make label start from 0 instead of 1
    """

    def __init__(self,
                 for_rnn,
                 img_dir,
                 iname2cid_file,
                 imgs_per_batch,
                 shuffle_inside_image=True,
                 decrement_cid=True,
                 ):
        super(MyIter, self).__init__()

        # save internal states
        self.for_rnn = for_rnn

        self.img_dir = img_dir
        assert os.path.exists(self.img_dir), "img_dir={} not exist".format(self.ing_dir)
        self.iname2cid_file = iname2cid_file

        self.img_per_batch = imgs_per_batch
        self.shuffle_inside_image = shuffle_inside_image
        self.decrement_cid = decrement_cid

        # load configurations in pre-processing
        # TODO : these are fixed for now
        self.window_sizes = my_constant.WINDOW_SIZES
        self.window_strides = my_constant.WINDOW_STRIDES
        self.pre_crop_resize_side = my_constant.RESIZE_SIDE
        self.mean_pixel = my_constant.MEAN_PIXEL_INT
        self.after_crop_resize_length = my_constant.INPUT_SIDE

        # compute number of windows per image
        self.windows_per_scale = []
        self.windows_per_img = 0
        for p in zip(self.window_sizes, self.window_strides):
            curr_num = ((self.pre_crop_resize_side - p[0]) / p[1] + 1) ** 2 + 1
            self.windows_per_scale.append(curr_num)
            self.windows_per_img += curr_num
        self.windows_per_scale = tuple(self.windows_per_scale)

        # batch size
        self.batch_size = self.img_per_batch * self.windows_per_img

        # status
        self.cursor = 0 # index of the next image, point to self.iname_list
        self.iname_list, self.iname2cid = self._load_iname2cid_file() # list of image names

        # data shapes
        self.data_shapes = []
        self.data_shapes.append(('data', (self.batch_size, 3, my_constant.INPUT_SIDE, my_constant.INPUT_SIDE)))

        # RNN init states
        self.init_state_shapes = []
        if self.for_rnn:
            for i in range(my_constant.NUM_RNN_LAYER):
                self.init_state_shapes.append(('rnn_l%d_init_c' % i, (1, my_constant.NUM_RNN_HIDDEN)))
                self.init_state_shapes.append(('rnn_l%d_init_h' % i, (1, my_constant.NUM_RNN_HIDDEN)))

        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in self.init_state_shapes]

        # label shapes
        self.label_shapes = []
        if self.for_rnn:
            self.label_shapes.append(('gesture_softmax_label', (1, )))
            self.label_shapes.append(('att_gesture_softmax_label', (self.batch_size, )))
        else:
            self.label_shapes.append(('softmax_label', (self.batch_size, )))

        # provide_data and provide_label
        self.provide_data = self.data_shapes + self.init_state_shapes
        self.provide_label = self.label_shapes


    def _load_iname2cid_file(self):
        """ load image names in training set, remove extension
        """

        # check if the file exists
        assert os.path.exists(self.iname2cid_file), "self.iname2cid_file={} not exist".format(self.iname2cid_file)

        iname_list = []
        iname2cid = {}

        with open(self.iname2cid_file, 'r') as f:
            for line in f:
                parts = line.split()

                iname = parts[0]
                cid = int(parts[1])

                iname_list.append(iname)
                iname2cid.update({iname : cid})

        return iname_list, iname2cid


    def reset(self):
        """
        restart sampling
        """

        self.cursor = 0
        random.shuffle(self.iname_list)


    def next(self):
        """
        get the data of next batch (DataBatch)
        """

        if self.iter_next():
            data = []
            label = []

            # for each image
            for ii in range(self.img_per_batch):
                curr_iname = self.iname_list[self.cursor]

                # generate windows
                curr_windows = my_util.preprocess(
                    img_dir=self.img_dir,
                    img_name=curr_iname,
                    pre_crop_resize_length=self.pre_crop_resize_side,
                    mean_pixel= self.mean_pixel,
                    window_sizes=self.window_sizes,
                    window_strides=self.window_strides,
                    after_crop_resize_length=self.after_crop_resize_length
                )

                # convert NHWC to NCHW
                curr_windows = np.swapaxes(curr_windows, 1, 3)
                curr_windows = np.swapaxes(curr_windows, 2, 3)

                # decrement cid
                curr_cid = self.iname2cid[curr_iname]
                if self.decrement_cid:
                    curr_cid -= 1

                # move to next image
                self.cursor += 1

                # shuffle windows with the same scale
                reorder_ids = []
                offset = 0
                for count in self.windows_per_scale:
                    scale_ids = range(offset, offset + count)
                    random.shuffle(scale_ids)

                    reorder_ids.extend(scale_ids)
                    offset += count

                curr_windows = curr_windows[reorder_ids, :, :, :]

                # append to list
                data.append(curr_windows)
                label.append([curr_cid for i in range(self.windows_per_img)])

            # make batch
            data = np.concatenate(data, axis=0)
            label = np.concatenate(label, axis=0)

            data = [mx.nd.array(data)]

            if self.for_rnn:
                assert len(np.unique(label)) == 1
                single_label = np.array(np.unique(label))

                # ('gesture_softmax_label', (1, ))
                # ('att_gesture_softmax_label', (self.batch_size, ))
                return mx.io.DataBatch(
                    data=data + self.init_state_arrays,
                    label=[mx.nd.array(single_label), mx.nd.array(label)])

            else:
                label = [mx.nd.array(label)]

                return mx.io.DataBatch(data=data, label=label)

        else:
            raise StopIteration


    def iter_next(self):
        """
        tell whether moving to the next batch is possible
        """

        if self.cursor + self.img_per_batch > len(self.iname_list):
            return False
        else:
            return True


