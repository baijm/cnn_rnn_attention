# train CNN from public model
/home/bjm/environments/mxnet_python2/bin/python test_rnn.py \
--bbox_dir=/home/bjm/datasets/Oxford-IIIT-Pet_cub-aug-coco/gt_bbox_test/rgb \
--iname2cid_file=/home/bjm/datasets/Oxford-IIIT-Pet_cub-aug-coco/test_iname2cid.txt \
--decrement_cid=1 \
--res_dir=/home/bjm/datasets/Oxford-IIIT-Pet_cub-aug-coco \
--res_prefix=full_12_gt_bbox \
--model_prefix=/home/bjm/projects/cnn_rnn_attention/checkpoints/Oxford-IIIT-Pet_cub-aug-coco_full \
--model_epoch=12 \
--gpu_id=1 > test_pet_full_12_gt_bbox.log 2>&1
