# train CNN from public model
/home/bjm/environments/mxnet_python2/bin/python train.py \
--dataset_root_dir=/media/bjm/Data/datasets/fine_grained/CUB-200-2011-aug_coco \
--train_img_list=train_orig.txt \
--test_img_list=test.txt \
--log_dir=/home/bjm/projects/cnn_rnn_attention/logs \
--ckpt_dir=/home/bjm/projects/cnn_rnn_attention/checkpoints \
--resume_training=0 \
--public_model_prefix=/home/bjm/pre-trained_models/vgg16 \
--public_model_epoch=0 \
--train_imgs_per_batch=1 \
--train_rnn=0 \
--num_epochs=100 \
--learning_rate=0.0001 # 0.0001 in caffe
# use step lr_policy in caffe