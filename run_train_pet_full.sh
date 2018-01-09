# train RNN and CNN, initialize by CNN
# batch_size = 4 for workstation
nohup /home/bjm/environments/mxnet_python2/bin/python train.py \
--dataset_root_dir=/home/bjm/datasets/Oxford-IIIT-Pet_cub-aug-coco \
--train_iname2cid_file=train_1000_iname2cid.txt \
--test_iname2cid_file=test_iname2cid.txt \
--log_dir=/home/bjm/projects/cnn_rnn_attention/logs \
--ckpt_dir=/home/bjm/projects/cnn_rnn_attention/checkpoints \
--train_cnn=1 \
--train_rnn=1 \
--resume_training=0 \
--from_epoch=5 \
--public_model_prefix=/home/bjm/projects/cnn_rnn_attention/for_thesis/Oxford-IIIT-Pet_cub-aug-coco_cnn \
--public_model_epoch=88 \
--train_imgs_per_batch=4 \
--num_epochs=100 \
--gpu_id=0 > pet_full_from_cnn88.log 2>&1 &
