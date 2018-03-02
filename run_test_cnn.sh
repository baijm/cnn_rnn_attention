#!/bin/bash

# test all models given by a list
python_path=/home/bjm/environments/mxnet_python2/bin/python

# which gpu to use
gpu_id=1

################## DATASET CONFIG ####################
# dataset root dir
dataset_root=/home/bjm/datasets/Flower-13-cub-aug-coco
dataset_name=`basename ${dataset_root}`

# test image list
iname2cid_file=${dataset_root}/test_iname2cid.txt

################## MODEL CONFIG ######################
# cid--
decrement_cid=1

# model root dir
model_root=/home/bjm/projects/cnn_rnn_attention/for_thesis/flower_cnn

# model suffix
model_suffix=cnn
# model epoch
model_epoch=4

# do test
# 1. gt_bbox
dataset_sub=gt/by_bbox/rgb
res_name=`echo $dataset_sub | sed 's/\//_/g'`
$python_path test_cnn.py \
--bbox_dir=${dataset_root}/${dataset_sub} \
--iname2cid_file=${iname2cid_file} \
--decrement_cid=${decrement_cid} \
--model_prefix=${model_root}/${dataset_name}_${model_suffix} \
--model_epoch=${model_epoch} \
--res_dir=${dataset_root} \
--res_prefix=${model_suffix}_${model_epoch}_${res_name} \
--gpu_id=${gpu_id}

# 2. det_segm
dataset_sub=det/by_segm/rgb
res_name=`echo $dataset_sub | sed 's/\//_/g'`
$python_path test_cnn.py \
--bbox_dir=${dataset_root}/${dataset_sub} \
--iname2cid_file=${iname2cid_file} \
--decrement_cid=${decrement_cid} \
--model_prefix=${model_root}/${dataset_name}_${model_suffix} \
--model_epoch=${model_epoch} \
--res_dir=${dataset_root} \
--res_prefix=${model_suffix}_${model_epoch}_${res_name} \
--gpu_id=${gpu_id}
