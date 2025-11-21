export WANDB_MODE=offline

# Define dataset explicitly for run_name usage
dataset=sudoku_train

# Ensure absolute path to model config to avoid cwd issues
MODEL_DIR=$(pwd)/model_config_tiny

exp=output/sudoku/Z-mask-RE/topk-margin-temp0-sigmoid0.3-weight1.5
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=2,3 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --dataset $dataset \
    --eval_dataset sudoku_test \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 2048 \
    --z_mask True \
    --reunmask True \
    --reunmask_correction_weight 1.5 \
    --z_mask_method topk-margin \
    --z_mask_temperature 0 \
    --z_schedule sigmoid \
    --z_sigmoid_center 0.3 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 1000 \
    --per_device_eval_batch_size 500 \
    --eval_steps 100 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin downk \
    --eval_temperatures 0.0 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 500.0 \
    --run_name topk-margin-temp0-sigmoid0.3-weight1.5 \
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



# exp=output/sudoku/Zmask-topk-margin-reunmask-temp0-sigmoid0.3-testdata-seed123
# mkdir -p $exp

# python3 -c "import swanlab; swanlab.sync_wandb()" && \
# CUDA_VISIBLE_DEVICES=0,1 \
# accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 2 --main_process_port 20099 \
# src/train_bash.py \
#     --stage mdm --overwrite_output_dir \
#     --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
#     --model_name_or_path $MODEL_DIR \
#     --seed 123 \
#     --do_train \
#     --do_eval \
#     --dataset $dataset \
#     --eval_dataset sudoku_test \
#     --finetuning_type full \
#     --cutoff_len 164 \
#     --output_dir $exp \
#     --overwrite_cache \
#     --per_device_train_batch_size 2048 \
#     --z_mask True \
#     --reunmask True \
#     --z_mask_method topk-margin \
#     --z_mask_temperature 0 \
#     --z_schedule sigmoid \
#     --z_sigmoid_center 0.3 \
#     --gradient_accumulation_steps 1 \
#     --lr_scheduler_type cosine \
#     --logging_steps 1 \
#     --val_size 1000 \
#     --per_device_eval_batch_size 500 \
#     --eval_steps 100 \
#     --eval_strategy steps \
#     --eval_denoise_methods origin topk topk-margin downk \
#     --eval_temperatures 0.0 \
#     --save_steps 500 \
#     --learning_rate 1e-3 \
#     --num_train_epochs 300.0 \
#     --run_name Sudoku-Zmask-topk-margin-reunmask-temp0-sigmoid0.3-testdata \
#     --preprocessing_num_workers 8 \
#     --fp16 \
#     --save_total_limit 1 \
#     --remove_unused_columns False \
#     --diffusion_steps 20 \
#     --save_safetensors False \
#     --token_reweighting False \
#     --time_reweighting linear \
#     --alpha 0.25 \
#     --gamma 1 \
#     > $exp/train.log
