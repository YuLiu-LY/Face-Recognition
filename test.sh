python train.py \
--gpus 1 \
--batch_size 128 \
--seed 42 \
--num_workers 8 \
--projection_dim 256 \
--test \
--test_ckpt_path '/home/yuliu/Projects/Face/results/d256_b512/version_1/checkpoints/last.ckpt' \
--predict_mode 'euclidean' \