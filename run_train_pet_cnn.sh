# train CNN from public model
# batch_size = 4 for workstation
nohup /home/bjm/environments/mxnet_python2/bin/python train.py \
--dataset_root_dir=/media/bjm/Data/datasets/fine_grained/Oxford-IIIT-Pet_cub-aug-coco \
--train_iname2cid_file=train_1000_iname2cid.txt \
--test_iname2cid_file=test_iname2cid.txt \
--log_dir=/home/bjm/projects/cnn_rnn_attention/logs \
--ckpt_dir=/home/bjm/projects/cnn_rnn_attention/checkpoints \
--resume_training=0 \
--from_epoch=0 \
--public_model_prefix=/home/bjm/pre-trained_models/vgg16 \
--public_model_epoch=0 \
--train_imgs_per_batch=2 \
--train_cnn=1 \
--train_rnn=0 \
--num_epochs=100 \
--gpu_id=0 > pet_cnn_from_public.log 2>&1 &
