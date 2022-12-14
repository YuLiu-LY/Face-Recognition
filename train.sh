python train.py \
--gpus 2 \
--batch_size 512 \
--log_name test_cosface \
--check_val_every_n_epoch 50 \
--is_logger_enabled \
--seed 42 \
--num_workers 4 \
--projection_dim 256 \
# --contras_weight 0 \
# --triplet_weight 1 \
# --predict_mode 'euclidean' \