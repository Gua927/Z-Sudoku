# nohup bash scripts/sudoku/train-origin-explore.sh > scripts/sudoku/logs/train-origin-explore.log 2>&1 &
# disown 

export WANDB_MODE=offline
export WANDB_PROJECT="Sudoku1204"
export SWANLAB_PROJ_NAME="Sudoku-RE1204"

# Define dataset explicitly for run_name usage
dataset=sudoku_train
seed=42
devices=0,1,2,3,4,5,6,7
z_schedule=piecewise
piecewise_ratio=0.25

# Ensure absolute path to model config to avoid cwd issues
MODEL_DIR=$(pwd)/model_config_tiny


exp=output/sudoku/Z-mask-RE/origin-$z_schedule-$piecewise_ratio-z1
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
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
    --reunmask False \
    --z_mask_method origin \
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
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name origin-$z_schedule$piecewise_ratio-z1 \
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

exp=output/sudoku/Z-mask-RE/origin-$z_schedule$piecewise_ratio-z2
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
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
    --z_mask_noise_step 2 \
    --reunmask False \
    --z_mask_method origin \
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
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name origin-$z_schedule$piecewise_ratio-z2 \
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

exp=output/sudoku/Z-mask-RE/origin-$z_schedule$piecewise_ratio-z3
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
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
    --z_mask_noise_step 3 \
    --reunmask False \
    --z_mask_method origin \
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
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name origin-$z_schedule$piecewise_ratio-z3 \
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

exp=output/sudoku/Z-mask-RE/origin-$z_schedule$piecewise_ratio-z-1
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
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
    --reunmask False \
    --z_mask_method origin \
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
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name origin-$z_schedule$piecewise_ratio-z-1 \
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


exp=output/sudoku/Z-mask-RE/origin-$z_schedule$piecewise_ratio-z1-weight2
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
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
    --reunmask True \
    --reunmask_correction_weight 2 \
    --z_mask_method origin \
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
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name origin-$z_schedule$piecewise_ratio-z1-weight2 \
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


exp=output/sudoku/Z-mask-RE/origin-$z_schedule$piecewise_ratio-z2-weight2
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
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
    --z_mask_noise_step 2 \
    --reunmask True \
    --reunmask_correction_weight 2 \
    --z_mask_method origin \
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
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name origin-$z_schedule$piecewise_ratio-z2-weight2 \
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


exp=output/sudoku/Z-mask-RE/origin-$z_schedule$piecewise_ratio-z3-weight2
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
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
    --z_mask_noise_step 3 \
    --reunmask True \
    --reunmask_correction_weight 2 \
    --z_mask_method origin \
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
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name origin-$z_schedule$piecewise_ratio-z3-weight2 \
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


exp=output/sudoku/Z-mask-RE/origin-$z_schedule$piecewise_ratio-z-1-weight2
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
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
    --reunmask_correction_weight 2 \
    --z_mask_method origin \
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
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name origin-$z_schedule$piecewise_ratio-z-1-weight2 \
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
