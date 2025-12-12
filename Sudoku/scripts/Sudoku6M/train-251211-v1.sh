# nohup bash scripts/Sudoku6M/train-251211-v1.sh > scripts/Sudoku6M/logs/train-251211-v1.log 2>&1 &
# disown

export WANDB_MODE=offline
export WANDB_PROJECT="Sudoku-6M"
export SWANLAB_PROJ_NAME="Sudoku-6M"

# Define dataset explicitly for run_name usage
dataset=sudoku_train
testdataset=sudoku_test
seed=42
devices=0,1,2,3,4,5,6,7

# Ensure absolute path to model config to avoid cwd issues
MODEL_DIR=$(pwd)/model_config_tiny

echo "============z1-mask MDM Training============"

z_schedule=all-z

exp=output/sudoku6M/z-mask/z1-allz-DKMG-temp2.5-NORM
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
    --seed $seed \
    --dataset $dataset \
    --eval_dataset $testdataset \
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
    --z_schedule $z_schedule \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 1000 \
    --per_device_eval_batch_size 125 \
    --eval_steps 500 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin downk-margin \
    --eval_temperatures 0.0 1.0 \
    --save_steps 2000 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name z1-allz-DKMG-temp2.5-NORM \
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


exp=output/sudoku6M/z-mask/z1-allz-DKMG-temp2-NORM
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
    --seed $seed \
    --dataset $dataset \
    --eval_dataset $testdataset \
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
    --z_mask_temperature 2 \
    --z_schedule $z_schedule \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 1000 \
    --per_device_eval_batch_size 125 \
    --eval_steps 500 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin downk-margin \
    --eval_temperatures 0.0 1.0 \
    --save_steps 2000 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name z1-allz-DKMG-temp2-NORM \
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


exp=output/sudoku6M/z-mask/z1-allz-DKMG-temp1.5-NORM
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
    --seed $seed \
    --dataset $dataset \
    --eval_dataset $testdataset \
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
    --z_mask_temperature 1.5 \
    --z_schedule $z_schedule \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 1000 \
    --per_device_eval_batch_size 125 \
    --eval_steps 500 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin downk-margin \
    --eval_temperatures 0.0 1.0 \
    --save_steps 2000 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name z1-allz-DKMG-temp1.5-NORM \
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
