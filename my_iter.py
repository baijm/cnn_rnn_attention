import mxnet as mx
import numpy as np
import os
import sys
import random
import scipy.io
import json
import my_constant


class MyIter(mx.io.DataIter):
    """ take a COCO-formed dataset, get data iterator for training CNN and RNN
    Parameters
    ----------------
    windows_dir : str
        dir of preprocessed windows.
            - each image has a .mat file, with keys 'cid' and 'preprocessed_windows'
            - one .json with information about pre-processing
    img_per_batch : int
        number of different images per batch
    data_name : str
    label_name : str
    img_name_file : str
        .txt file with each line as the name of a training image
    shuffle_inside_image : bool
        shuffle windows of the same scale within each image (for training)
    decrement_cid : bool
        make label start from 0 instead of 1
    """

    def __init__(self,
                 windows_dir,
                 img_name_file,
                 data_name,
                 label_name,
                 imgs_per_batch,
                 shuffle_inside_image=True,
                 decrement_cid=True,
                 ):
        super(MyIter, self).__init__()

        # save internal states
        self.windows_dir = windows_dir
        assert os.path.exists(self.windows_dir), "windows_dir={} not exist".format(self.windows_dir)
        self.img_name_file = img_name_file

        self.data_name = data_name
        self.label_name = label_name

        self.img_per_batch = imgs_per_batch
        self.shuffle_inside_image = shuffle_inside_image
        self.decrement_cid = decrement_cid

        # load configurations in pre-processing
        self.preprocess_config = self._load_preprocessing_config()

        # compute number of windows per image
        self.window_sizes = self.preprocess_config["window_sizes"]
        self.window_strides = self.preprocess_config["window_strides"]
        self.pre_crop_resize_side = self.preprocess_config["pre_crop_resize_side"]

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
        self.cursor = 0 # index of the next image, point to self.image_name_list
        self.image_name_list = self._load_img_name_file() # list of image names


    def _load_preprocessing_config(self):
        """ load configurations in pre-processing
        """

        # check if config.json is in self.windows_dir
        assert os.path.exists(os.path.join(self.windows_dir, "config.json")), "config.json not in windows_dir"
        config_info = json.load(open(os.path.join(self.windows_dir, "config.json"), 'r'))
        return config_info


    def _load_img_name_file(self):
        """ load image names in training set, remove extension
        """

        # check if the file exists
        assert os.path.exists(self.img_name_file), "self.img_name_file={} not exist".format(self.img_name_file)
        res = []

        with open(self.img_name_file, 'r') as f:
            for line in f:
                iname = line.split('.')[0]
                res.append(iname)

        return res


    @property
    def provide_data(self):
        """
        the name and shape of data provided by this iterator
        """
        return [mx.io.DataDesc(
            name=self.data_name,
            shape=(self.batch_size, 3, my_constant.INPUT_SIDE, my_constant.INPUT_SIDE),
            # layout='NCHW'
        )]


    @property
    def provide_label(self):
        """
        the name and shape of label provided by this iterator
        """
        return [mx.io.DataDesc(
            name=self.label_name,
            shape=(self.batch_size, )
        )]


    def reset(self):
        """
        restart sampling
        """

        self.cursor = 0
        random.shuffle(self.image_name_list)


    def next(self):
        """
        get the data of next batch (DataBatch)
        """

        if self.iter_next():
            data = []
            label = []

            # for each image
            for ii in range(self.img_per_batch):
                # load windows
                winfo_file = self.image_name_list[self.cursor] + ".mat"
                assert os.path.exists(os.path.join(self.windows_dir, winfo_file)), \
                    "{} not exist".format(os.path.join(self.windows_dir, winfo_file))
                winfo = scipy.io.loadmat(os.path.join(self.windows_dir, winfo_file))
                curr_windows = winfo["preprocessed_windows"]

                # convert NHWC to NCHW
                curr_windows = np.swapaxes(curr_windows, 1, 3)
                curr_windows = np.swapaxes(curr_windows, 2, 3)

                curr_cid = np.squeeze(winfo["cid"])
                # decrement cid
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
            label = [mx.nd.array(label)]
            return mx.io.DataBatch(
                data=data,
                label=label)
        else:
            raise StopIteration


    def iter_next(self):
        """
        tell whether moving to the next batch is possible
        """

        if self.cursor + self.img_per_batch > len(self.image_name_list):
            return False
        else:
            return True


"""
DataDesc

For sequential data, by default layout is set to NTC, where N is number of examples in the batch, T the temporal axis representing time and C is the number of channels.
"""


