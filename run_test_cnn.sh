#!/bin/bash

# test all models given by a list
python_path=/home/bjm/environments/mxnet_python2/bin/python

# which gpu to use
gpu_id=2

################## DATASET CONFIG ####################
# dataset root dir
dataset_root=/home/bjm/datasets/Oxford-IIIT-Pet_cub-aug-coco
dataset_name=`basename ${dataset_root}`

# test image list
iname2cid_file=${dataset_root}/test_iname2cid.txt

################## MODEL CONFIG ######################
# cid--
decrement_cid=1

# model root dir
model_root=/home/bjm/projects/cnn_rnn_attention/for_thesis/pet_cnn_erode_7

# model suffix
model_suffix=cnn
# model epoch
model_epoch=92

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

dataset_sub=gt/by_bbox_erode_5/rgb
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

dataset_sub=gt/by_bbox_erode_7/rgb
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

# 3. det_bbox
dataset_sub=det/by_bbox/rgb
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

dataset_sub=det/by_bbox_erode_5/rgb
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

dataset_sub=det/by_bbox_erode_7/rgb
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

# 4. det_segm
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

dataset_sub=det/by_segm_erode_5/rgb
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

dataset_sub=det/by_segm_erode_7/rgb
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


