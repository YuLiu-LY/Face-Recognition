python train.py \
--gpus 2 \
--batch_size 256 \
--log_name res50 \
--check_val_every_n_epoch 100 \
--is_logger_enabled \
--seed 42 \
--num_workers 8 \
--projection_dim 2048 \
--prediction_dim 2048 \