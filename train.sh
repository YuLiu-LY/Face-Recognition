python train.py \
--gpus 1 \
--batch_size 128 \
--log_name val_in \
--check_val_every_n_epoch 100 \
--is_logger_enabled \
--seed 42 \
--num_workers 4 \
--projection_dim 256 \
--device '1' \
--data_root '/home/yuliu/Dataset/Face1' \
--N_layer 64 \
--margin 0 \
--m_warmup_steps 0 \
--scale 64 \
--learn_scale \
--max_steps 40000 \
# --margin 0 \
# --log_path '/home/liuyu/scratch/Face/results/' \
# --contras_weight 0 \
# --triplet_weight 1 \
# --predict_mode 'euclidean' \