export WANDB_MODE=offline

# Define dataset explicitly for run_name usage
dataset=sudoku_train

# Ensure absolute path to model config to avoid cwd issues
MODEL_DIR=$(pwd)/model_config_tiny

exp=output/sudoku/Zmask-origin-temp0
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --dataset $dataset \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 2048 \
    --z_mask True \
    --z_mask_method origin \
    --z_mask_temperature 0 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 1024 \
    --per_device_eval_batch_size 512 \
    --eval_steps 100 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin \
    --eval_temperatures 0.0 0.5 1.0 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 500.0 \
    --run_name Sudoku-Zmask-origin-temp0 \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --alpha 0.25 \
    --gamma 1 \
    > $exp/train.log



exp=output/sudoku/Zmask-topk-temp0
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --dataset $dataset \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 2048 \
    --z_mask True \
    --z_mask_method topk \
    --z_mask_temperature 0 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 1024 \
    --per_device_eval_batch_size 512 \
    --eval_steps 100 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin \
    --eval_temperatures 0.0 0.5 1.0 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 500.0 \
    --run_name Sudoku-Zmask-topk-temp0 \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --alpha 0.25 \
    --gamma 1 \
    > $exp/train.log

exp=output/sudoku/Zmask-topk-margin-temp0
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --dataset $dataset \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 2048 \
    --z_mask True \
    --z_mask_method topk-margin \
    --z_mask_temperature 0 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 1024 \
    --per_device_eval_batch_size 512 \
    --eval_steps 100 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin \
    --eval_temperatures 0.0 0.5 1.0 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 500.0 \
    --run_name Sudoku-Zmask-topk-margin-temp0 \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --alpha 0.25 \
    --gamma 1 \
    > $exp/train.log



# temperature=0

# for dataset in sudoku_test
# do
# mkdir -p $exp/${dataset}-origin
# echo "运行评测: Origin denoising"
# CUDA_VISIBLE_DEVICES=1  \
# python3 -u src/train_bash.py \
#     --stage mdm --overwrite_output_dir \
#     --cache_dir /root/code/EasyMDM/Sudoku/cache \
#     --model_name_or_path model_config_tiny \
#     --do_predict \
#     --cutoff_len 164 \
#     --dataset $dataset \
#     --finetuning_type full \
#     --diffusion_steps 20 \
#     --output_dir $exp/${dataset}-origin \
#     --checkpoint_dir $exp  \
#     --remove_unused_columns False \
#     --denoise_method origin \
#     --temperature ${temperature} \
#     --run_name Sudoku-TKRW-origin-temp${temperature} \
#     > $exp/${dataset}-origin/eval-origin-temp${temperature}.log
# echo "Origin denoising 评测完成，日志保存到: $exp/${dataset}-origin/eval-origin-temp${temperature}.log"
# done

# # 第二次评测：TopK=True, linear 调度
# for dataset in sudoku_test
# do
# mkdir -p $exp/${dataset}-TopK
# echo "运行评测: TopK denoising"
# CUDA_VISIBLE_DEVICES=1  \
# python3 -u src/train_bash.py \
#     --stage mdm --overwrite_output_dir \
#     --cache_dir /root/code/EasyMDM/Sudoku/cache \
#     --model_name_or_path model_config_tiny \
#     --do_predict \
#     --cutoff_len 164 \
#     --dataset $dataset \
#     --finetuning_type full \
#     --diffusion_steps 20 \
#     --output_dir $exp/${dataset}-TopK \
#     --checkpoint_dir $exp  \
#     --remove_unused_columns False \
#     --denoise_method topk \
#     --decoding_strategy linear \
#     --temperature ${temperature} \
#     --run_name Sudoku-TKRW-TopK-temp${temperature} \
#     > $exp/${dataset}-TopK/eval-TopK-temp${temperature}.log
# echo "TopK denoising 评测完成，日志保存到: $exp/${dataset}-TopK/eval-TopK-temp${temperature}.log"
# done

# # 第三次评测：TopK-margin=True, linear 调度
# for dataset in sudoku_test
# do
# mkdir -p $exp/${dataset}-TopK-margin
# echo "运行评测: TopK-margin denoising"
# CUDA_VISIBLE_DEVICES=1  \
# python3 -u src/train_bash.py \
#     --stage mdm --overwrite_output_dir \
#     --cache_dir /root/code/EasyMDM/Sudoku/cache \
#     --model_name_or_path model_config_tiny \
#     --do_predict \
#     --cutoff_len 164 \
#     --dataset $dataset \
#     --finetuning_type full \
#     --diffusion_steps 20 \
#     --output_dir $exp/${dataset}-TopK-margin \
#     --checkpoint_dir $exp  \
#     --remove_unused_columns False \
#     --denoise_method topk-margin \
#     --decoding_strategy linear \
#     --temperature ${temperature} \
#     --run_name Sudoku-TKRW-TopK-margin-temp${temperature} \
#     > $exp/${dataset}-TopK-margin/eval-TopK-margin-temp${temperature}.log
# echo "TopK-margin 评测完成，日志保存到: $exp/${dataset}-TopK-margin/eval-TopK-margin-temp${temperature}.log"
# done

