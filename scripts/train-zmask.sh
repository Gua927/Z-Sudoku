export WANDB_MODE=offline
export WANDB_PROJECT="Z-Sudoku"
export SWANLAB_PROJ_NAME="Z-Sudoku"


# Ensure absolute path to model config to avoid cwd issues
MODEL_DIR=$(pwd)/model_config_tiny
devices=0,1,2,3,4,5,6,7

exp=output/z-mask
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20100 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --seed 42 \
    --dataset sudoku_train \
    --eval_dataset sudoku_test \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --z_mask True \
    --z_loss True \
    --z_loss_step_interval 100 \
    --reunmask False \
    --z_mask_method downk-margin \
    --downk_ratio 0.3 \
    --z_mask_temperature 2.5 \
    --z_schedule all-z \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 1000 \
    --per_device_eval_batch_size 125 \
    --eval_steps 500 \
    --eval_strategy steps \
    --eval_denoise_methods origin downk-margin \
    --eval_temperatures 0.0 \
    --save_steps 2000 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name z-mask \
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
