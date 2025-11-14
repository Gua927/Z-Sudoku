#!/bin/bash
export WANDB_MODE=offline

# Define dataset explicitly for run_name usage
dataset=sudoku_train
temperature=1.0

exp=output/sudoku/mdm-alpha0.25-gamma1-lr1e-3-ep300-T20-`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

# Ensure absolute path to model config to avoid cwd issues
MODEL_DIR=$(pwd)/model_config_tiny

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --dataset $dataset \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 2048 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 2048 \
    --per_device_eval_batch_size 512 \
    --eval_steps 50 \
    --save_steps 100 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --plot_loss \
    --run_name Sudoku-NOTKRW-Linear \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --denoise_method origin \
    --alpha 0.25 \
    --gamma 1 \
    --temperature ${temperature} \
    > $exp/train.log

echo "训练完成！输出目录: $exp"