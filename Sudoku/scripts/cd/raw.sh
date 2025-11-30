# nohup bash scripts/cd/raw.sh > scripts/cd/logs/raw.log 2>&1 &
# disown


export WANDB_MODE=offline
export WANDB_PROJECT="Cutdown"
export SWANLAB_PROJ_NAME="Cutdown"


exp=output/Cutdown/cd4-raw
mkdir -p $exp

python3 -c "import swanlab; swanlab.sync_wandb()" && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /GenSIvePFS/users/zlzhang/trz/DLM-A100-2/cache \
    --model_name_or_path model_config \
    --do_train \
    --dataset cd4_train \
    --finetuning_type full \
    --cutoff_len 64 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 56 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --save_steps 2000 \
    --learning_rate 3e-4 \
    --num_train_epochs 600.0 \
    --plot_loss \
    --run_name cd4-raw \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting True \
    --time_reweighting linear \
    --topk_decoding True \
    --alpha 0.25 \
    --gamma 2 \
    > $exp/train.log

for dataset in cd4_tot24 cd4_test
do
topk_decoding=True
mkdir $exp/$dataset
CUDA_VISIBLE_DEVICES=1  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config \
    --do_predict \
    --cutoff_len 64 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding $topk_decoding \
    > $exp/${dataset}/eval-TopK$topk_decoding.log
done