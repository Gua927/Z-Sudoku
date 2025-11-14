export WANDB_MODE=offline

python3 -c "import swanlab; swanlab.sync_wandb()" && \

exp=output/sudoku/mdm-alpha0.25-gamma1-lr1e-3-ep300-T20-20251113-183444

temperature=1.0


echo "开始评测，使用训练输出目录: $exp"

# 第一次评测：TopK=False
for dataset in sudoku_test
do
mkdir -p $exp/${dataset}-origin
echo "运行评测: Origin denoising"
CUDA_VISIBLE_DEVICES=1  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /root/code/EasyMDM/Sudoku/cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 164 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/${dataset}-origin \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --denoise_method origin \
    --temperature ${temperature} \
    --run_name Sudoku-NOTKRW-origin-temp1.0
    > $exp/${dataset}-origin/eval-origin-temp1.0.log
echo "Origin denoising 评测完成，日志保存到: $exp/${dataset}-origin/eval-origin-temp1.0.log"
done

# 第二次评测：TopK=True, linear 调度
for dataset in sudoku_test
do
mkdir -p $exp/${dataset}-TopK
echo "运行评测: TopK denoising"
CUDA_VISIBLE_DEVICES=1  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /root/code/EasyMDM/Sudoku/cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 164 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/${dataset}-TopK \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --denoise_method topk \
    --decoding_strategy linear \
    --temperature ${temperature} \
    --run_name Sudoku-NOTKRW-TopK-temp1.0
    > $exp/${dataset}-TopK/eval-TopK-temp1.0.log
echo "TopK denoising 评测完成，日志保存到: $exp/${dataset}-TopK/eval-TopK-temp1.0.log"
done

# 第三次评测：TopK-margin=True, linear 调度
for dataset in sudoku_test
do
mkdir -p $exp/${dataset}-TopK-margin
echo "运行评测: TopK-margin denoising"
CUDA_VISIBLE_DEVICES=1  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir /root/code/EasyMDM/Sudoku/cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 164 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/${dataset}-TopK-margin \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --denoise_method topk-margin \
    --decoding_strategy linear \
    --temperature ${temperature} \
    --run_name Sudoku-NOTKRW-TopK-margin-temp1.0
    > $exp/${dataset}-TopK-margin/eval-TopK-margin-temp1.0.log
echo "TopK-margin 评测完成，日志保存到: $exp/${dataset}-TopK-margin/eval-TopK-margin-temp1.0.log"
done

echo "所有评测完成！"