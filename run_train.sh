# train RNN and CNN, initialize by CNN
# batch_size = 4 for workstation
nohup /home/bjm/environments/mxnet_python2/bin/python train.py \
--dataset_root_dir=/home/bjm/datasets/Flower-13-cub-aug-coco \
--train_iname2cid_file=train_1000_iname2cid.txt \
--test_iname2cid_file=test_iname2cid.txt \
--log_dir=/home/bjm/projects/cnn_rnn_attention/for_thesis/flower_full_masked \
--ckpt_dir=/home/bjm/projects/cnn_rnn_attention/for_thesis/flower_full_masked \
--train_cnn=1 \
--train_rnn=1 \
--resume_training=0 \
--public_model_prefix=/home/bjm/projects/cnn_rnn_attention/for_thesis/flower_cnn_masked/Flower-13-cub-aug-coco_cnn \
--public_model_epoch=10 \
--train_imgs_per_batch=4 \
--num_epochs=100 \
--gpu_id=1 > flower_full_masked.log 2>&1 &
