python train.py \
--gpus 1 \
--batch_size 128 \
--log_name w_aug \
--check_val_every_n_epoch 50 \
--is_logger_enabled \
--seed 42 \
--num_workers 4 \
--projection_dim 256 \
--data_root '/scratch/generalvision/SlotAttention/Face' \
--log_path '/home/liuyu/scratch/Face/results/' \
--device '0' \
# --contras_weight 0 \
# --triplet_weight 1 \
# --predict_mode 'euclidean' \