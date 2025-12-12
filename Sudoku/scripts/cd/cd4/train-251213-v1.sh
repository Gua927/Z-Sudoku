# nohup bash scripts/cd/cd4/train-251213-v1.sh > scripts/cd/logs/train-251213-v1.log 2>&1 &
# disown

export WANDB_MODE=offline
export WANDB_PROJECT="cd4-85M"
export SWANLAB_PROJ_NAME="cd4-85M"

# Define dataset explicitly for run_name usage
dataset=cd4_train
devices=0,1,2,3,4,5,6,7
MODEL_DIR=$(pwd)/model_config

seed=42

z_schedule=all-z

exp=output/Cutdown/cd4-85M/z1-$z_schedule-DKMG-temp2.5-NORM
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
    --finetuning_type full \
    --cutoff_len 64 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 512 \
    --z_mask True \
    --z_loss True \
    --z_loss_step_interval 100 \
    --z_mask_method downk-margin \
    --downk_ratio 0.3 \
    --z_mask_temperature 2.5 \
    --z_schedule $z_schedule \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \
    --logging_steps 1 \
    --val_size 1000 \
    --per_device_eval_batch_size 125 \
    --eval_steps 2000 \
    --eval_strategy steps \
    --eval_denoise_methods origin topk topk-margin downk-margin \
    --eval_temperatures 0.0 1.0 \
    --save_steps 5000 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --run_name z1-$z_schedule-DKMG-temp2.5-NORM \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting False \
    --time_reweighting linear \
    --alpha 0.25 \
    --gamma 2 \
    > $exp/train.log

