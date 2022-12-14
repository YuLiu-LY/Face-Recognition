python train.py \
--gpus 1 \
--batch_size 128 \
--log_name wo_maigin_64 \
--check_val_every_n_epoch 50 \
--is_logger_enabled \
--seed 42 \
--num_workers 4 \
--projection_dim 256 \
--device '2' \
--data_root '/home/yuliu/Dataset/Face1' \
--N_layer 64 \
--margin 0 \
--m_warmup_steps 0 \
# --log_path '/home/liuyu/scratch/Face/results/' \
# --contras_weight 0 \
# --triplet_weight 1 \
# --predict_mode 'euclidean' \