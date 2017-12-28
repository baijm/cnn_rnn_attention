import os
import numpy as np
from PIL import Image
import scipy.io

windows_dir = "/media/bjm/Data/datasets/fine_grained/CUB-200-2011-aug_coco/gt_bbox_windows"
res_dir = "/home/bjm/tmp"

MEAN_PIXEL_INT = (123, 117, 104)

im_name = "American_Crow_0004_25819"
mat = scipy.io.loadmat(os.path.join(windows_dir, im_name + ".mat"))
windows = mat["preprocessed_windows"]

for wi in range(windows.shape[0]):
	w = windows[wi, :, :, :]
	w[:, :, 0] += MEAN_PIXEL_INT[0]
	w[:, :, 1] += MEAN_PIXEL_INT[1]
	w[:, :, 2] += MEAN_PIXEL_INT[2]
	w = w.astype(np.uint8)
	wim = Image.fromarray(w)
	wim.save(os.path.join(res_dir, "{}_{}.jpg".format(im_name, wi)))
    
