python main.py \
--gpus 1 \
--batch_size 128 \
--log_name 'train1' \
--check_val_every_n_epoch 100 \
--seed 42 \
--num_workers 4 \
--projection_dim 256 \
--device '0' \
--action 'train' \
--data_root '/home/yuliu/Dataset/Face1' \
--log_parh '/home/yuliu/Projects/Face/results/' \
--is_logger_enabled \