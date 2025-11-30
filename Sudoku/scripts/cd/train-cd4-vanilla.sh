# nohup bash scripts/cd/train-cd4-vanilla.sh > scripts/cd/logs/train-cd4-vanilla.log 2>&1 &
# disown

export WANDB_MODE=offline
export WANDB_PROJECT="Cutdown"
export SWANLAB_PROJ_NAME="Cutdown"

# Define dataset explicitly for run_name usage
dataset=cd4_train
seed=42
devices=0,1,2,3,4,5,6,7

exp=output/Cutdown/cd4-Vanilla/cd4-Vanilla-seed$seed
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path model_config \
    --do_train \
    --do_eval \
    --seed $seed \
    --dataset $dataset \
    --finetuning_type full \
    --cutoff_len 64 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --z_mask False \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 56 \
    --eval_steps 1000 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin downk-margin \
    --eval_temperatures 0.0 1.0 \
    --save_steps 2000 \
    --learning_rate 3e-4 \
    --num_train_epochs 600.0 \
    --run_name cd4-Vanilla-seed$seed \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --denoise_method topk \
    --alpha 0.25 \
    --gamma 2 \
    > $exp/train.log

seed=114

exp=output/Cutdown/cd4-Vanilla/cd4-Vanilla-seed$seed
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path model_config \
    --do_train \
    --do_eval \
    --seed $seed \
    --dataset $dataset \
    --finetuning_type full \
    --cutoff_len 64 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --z_mask False \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 8 \
    --eval_steps 1000 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin downk-margin \
    --eval_temperatures 0.0 1.0 \
    --save_steps 2000 \
    --learning_rate 3e-4 \
    --num_train_epochs 600.0 \
    --run_name cd4-Vanilla-seed$seed \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --denoise_method topk \
    --alpha 0.25 \
    --gamma 2 \
    > $exp/train.log

seed=514

exp=output/Cutdown/cd4-Vanilla/cd4-Vanilla-seed$seed
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=$devices \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path model_config \
    --do_train \
    --do_eval \
    --seed $seed \
    --dataset $dataset \
    --finetuning_type full \
    --cutoff_len 64 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --z_mask False \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 8 \
    --eval_steps 1000 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin downk-margin \
    --eval_temperatures 0.0 1.0 \
    --save_steps 2000 \
    --learning_rate 3e-4 \
    --num_train_epochs 600.0 \
    --run_name cd4-Vanilla-seed$seed \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --denoise_method topk \
    --alpha 0.25 \
    --gamma 2 \
    > $exp/train.log