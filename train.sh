python train.py \
--gpus 1 \
--batch_size 128 \
--log_name move_all \
--check_val_every_n_epoch 100 \
--seed 42 \
--num_workers 4 \
--projection_dim 256 \
--device '3' \
--N_layer 64 \
--action 'test' \
--test_ckpt_path '/home/yuliu/Projects/Face/results/base/version_0/checkpoints/last.ckpt' \
--data_root '/home/yuliu/Dataset/Face1' \
# --is_logger_enabled \
# --use_aug \
# --use_BN \
# --margin 0.35 \
# --learn_scale \
# --relu_type 'prelu' \
# --margin 0 \
# --log_path '/home/liuyu/scratch/Face/results/' \
# --contras_weight 0 \
# --triplet_weight 1 \
# --predict_mode 'euclidean' \