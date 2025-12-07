# nohup bash scripts/Sudoku12M/train-251206-v3.sh > scripts/Sudoku12M/logs/train-251206-v3.log 2>&1 &
# disown

export WANDB_MODE=offline
export WANDB_PROJECT="Sudoku-12M"
export SWANLAB_PROJ_NAME="Sudoku-12M"

# Define dataset explicitly for run_name usage
dataset=sudoku_train
testdataset=sudoku_test
seed=42
devices=0,1,2,3,4,5,6,7

# Ensure absolute path to model config to avoid cwd issues
MODEL_DIR=$(pwd)/model_config_12M


z_schedule=all-z

exp=output/sudoku12M/z-mask/z1PSOY-allz-DKMG-NORM
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
    --eval_dataset sudoku_test \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --z_mask True \
    --pos_only True \
    --reunmask False \
    --z_mask_method downk-margin \
    --downk_ratio 0.3 \
    --z_mask_temperature 1 \
    --z_schedule $z_schedule \
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
    --learning_rate 6e-4 \
    --num_train_epochs 300.0 \
    --run_name z1PSOY-allz-DKMG-NORM \
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



exp=output/sudoku12M/z-mask/z-1-allz-LSDV-NORM
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20101 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --seed $seed \
    --dataset $dataset \
    --eval_dataset sudoku_test \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --z_mask True \
    --z_mask_noise_step -1 \
    --z_mask_method loss_driven \
    --z_mask_temperature 1 \
    --z_schedule $z_schedule \
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
    --learning_rate 6e-4 \
    --num_train_epochs 300.0 \
    --run_name z-1-allz-LSDV-NORM \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --alpha 0.25 \

exp=output/sudoku12M/z-mask/z-1-allz-LSDV-RM2
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20101 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --seed $seed \
    --dataset $dataset \
    --eval_dataset sudoku_test \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --z_mask True \
    --z_mask_noise_step -1 \
    --z_mask_method loss_driven \
    --z_mask_temperature 1 \
    --reunmask True \
    --reunmask_correction_weight 2.0 \
    --z_schedule $z_schedule \
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
    --learning_rate 6e-4 \
    --num_train_epochs 300.0 \
    --run_name z-1-allz-LSDV-RM2 \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --alpha 0.25 \


z_schedule=piecewise
piecewise_ratio=0.5

exp=output/sudoku12M/z-mask/z1-piecewise0.5-LSDV-RM2
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20101 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --do_eval \
    --seed $seed \
    --dataset $dataset \
    --eval_dataset sudoku_test \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --z_mask True \
    --z_mask_noise_step -1 \
    --reunmask True \
    --reunmask_correction_weight 2.0 \
    --z_mask_method loss_driven \
    --z_mask_temperature 1 \
    --z_schedule $z_schedule \
    --piecewise_ratio $piecewise_ratio \
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
    --learning_rate 6e-4 \
    --num_train_epochs 300.0 \
    --run_name z-1-allz-LSDV-RM2 \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --alpha 0.25 \


exp=output/sudoku12M/z-mask/z1PSOY-piecewise0.5-DKMG-NORM
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
    --eval_dataset sudoku_test \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --z_mask True \
    --pos_only True \
    --reunmask False \
    --z_mask_method downk-margin \
    --downk_ratio 0.3 \
    --z_mask_temperature 1 \
    --z_schedule $z_schedule \
    --piecewise_ratio $piecewise_ratio \
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
    --learning_rate 6e-4 \
    --num_train_epochs 300.0 \
    --run_name z1PSOY-piecewise0.5-DKMG-NORM \
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