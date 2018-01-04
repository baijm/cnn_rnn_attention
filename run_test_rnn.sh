# train CNN from public model
/home/bjm/environments/mxnet_python2/bin/python test_rnn.py \
--bbox_dir=/media/bjm/Data/datasets/fine_grained/CUB-200-2011-aug_coco/det_result/bbox/rgb_masked \
--iname2cid_file=/media/bjm/Data/datasets/fine_grained/CUB-200-2011-aug_coco/test_iname2cid.txt \
--decrement_cid=1 \
--res_dir=/media/bjm/Data/datasets/fine_grained/CUB-200-2011-aug_coco \
--res_prefix=rnn_6_det_segm \
--model_prefix=/home/bjm/projects/cnn_rnn_attention/for_thesis/CUB-200-2011-aug_coco_rnn \
--model_epoch=6 \
--gpu_id=0
