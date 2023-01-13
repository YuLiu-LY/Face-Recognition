python main.py \
--gpus 1 \
--batch_size 128 \
--log_name 'test' \
--data_root '/home/yuliu/Dataset/Face1' \
--log_path '/home/yuliu/Projects/Face/results/' \
--test_ckpt_path './ckpt/train_val.ckpt' \
--action 'test' \
--test_result_name 'pred_train_val' \
# --fix_threshold \
# --threshold 0.5 \
# --is_logger_enabled \

