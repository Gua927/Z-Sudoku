#  nohup bash scripts/sudoku/train-vanilla.sh >train-vanilla.log &

export WANDB_MODE=offline

# Define dataset explicitly for run_name usage
dataset=sudoku_train

# Ensure absolute path to model config to avoid cwd issues
MODEL_DIR=$(pwd)/model_config_tiny

exp=output/sudoku/Vanilla
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
    --z_mask False \
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
    --num_train_epochs 300 \
    --run_name Sudoku-Vanilla \
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


exp=output/sudoku/Zmask-topk-margin-temp0-allz
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
    --z_schedule all-z \
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
    --num_train_epochs 300 \
    --run_name Sudoku-Zmask-topk-margin-temp0-allz \
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

exp=output/sudoku/Zmask-topk-temp0-allz
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
    --z_schedule all-z \
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
    --num_train_epochs 300 \
    --run_name Sudoku-Zmask-topk-temp0-allz \
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


exp=output/sudoku/Zmask-origin-temp0-allz
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
    --z_schedule all-z \
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
    --num_train_epochs 300 \
    --run_name Sudoku-Zmask-origin-temp0-allz \
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

