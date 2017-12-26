# from constant.py
NUM_FILTER = 16 # TODO : change ?
NUM_HIDDEN = 512 # TODO : change ?
NUM_BOTTLENECK = 128 # TODO : change ?
DROPOUT = 0.5
GAMMA = 10

# sliding window config
RESIZE_SIDE = 256
WINDOW_SIZES = (224, 168, 112)
WINDOW_STRIDES = (32, 44, 48)
WINDOW_NUMS = tuple([((RESIZE_SIDE - p[0])/p[1] + 1) ** 2 + 1
                     for p in zip(WINDOW_SIZES, WINDOW_STRIDES)])


# CNN config
INPUT_SIDE = 224
FEATURE_DIM = 4096
MEAN_PIXEL = (123.68, 116.779, 103.939) # RGB
MEAN_PIXEL_INT = (123, 117, 104) # RGB


# RNN config
NUM_RNN_HIDDEN = 512 # as in paper
NUM_RNN_LAYER = 1
RNN_DROPOUT = 0.5


# training config
IMG_PER_BATCH = 1