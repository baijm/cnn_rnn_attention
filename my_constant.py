# from constant.py
DROPOUT = 0.5
GAMMA = 10




# sliding window config
RESIZE_SIDE = 256
WINDOW_SIZES = (224, 168) #(224, 168, 112)
WINDOW_STRIDES = (32, 44) #(32, 44, 48)
#WINDOW_NUMS = tuple([((RESIZE_SIDE - p[0])/p[1] + 1) ** 2 + 1
#                     for p in zip(WINDOW_SIZES, WINDOW_STRIDES)])


# CNN config
INPUT_SIDE = 224
FEATURE_DIM = 4096
MEAN_PIXEL = (123.68, 116.779, 103.939) # RGB
MEAN_PIXEL_INT = (123, 117, 104) # RGB


# RNN config
NUM_RNN_HIDDEN = 512 # as in paper
NUM_RNN_LAYER = 1
NUM_RNN_WINDOW = 32
RNN_DROPOUT = 0.5


# training config
IMG_PER_BATCH = 1
CNN_LEARNING_RATE = 0.0001
MOMENTUM=0.9
WD=0.0001