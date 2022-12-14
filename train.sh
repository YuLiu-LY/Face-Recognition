python train.py \
--gpus 1 \
--batch_size 128 \
--log_name cosface_b128 \
--check_val_every_n_epoch 50 \
--is_logger_enabled \
--seed 42 \
--num_workers 4 \
--projection_dim 256 \
--device '1' \
# --contras_weight 0 \
# --triplet_weight 1 \
# --predict_mode 'euclidean' \