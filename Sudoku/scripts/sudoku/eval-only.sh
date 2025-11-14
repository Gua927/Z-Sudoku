export WANDB_MODE=offline

python3 -c "import swanlab; swanlab.sync_wandb()" && \

exp=output/sudoku/mdm-alpha0.25-gamma1-lr1e-3-ep300-T20-20251113-183444



echo "开始评测，使用训练输出目录: $exp"

# 第一次评测：TopK=False
for dataset in sudoku_test
do
mkdir -p $exp/${dataset}-TopK-False
echo "运行评测: TopK=False"
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
    --output_dir $exp/${dataset}-TopK-False \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --topk_decoding False \
    --run_name Sudoku-NOTKRW-Linear-TopK-False
    > $exp/${dataset}-TopK-False/eval-TopK-False.log
echo "TopK=False 评测完成，日志保存到: $exp/${dataset}-TopK-False/eval-TopK-False.log"
done

# 第二次评测：TopK=True, stochastic0.5-linear
for dataset in sudoku_test
do
mkdir -p $exp/${dataset}-TopK-True-stochastic
echo "运行评测: TopK=True, stochastic0.5-linear"
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
    --output_dir $exp/${dataset}-TopK-True-stochastic \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --topk_decoding True \
    --decoding_strategy stochastic0.5-linear \
    --run_name Sudoku-NOTKRW-Linear-TopK-True-stochastic
    > $exp/${dataset}-TopK-True-stochastic/eval-TopK-True-stochastic0.5-linear.log
echo "TopK=True 评测完成，日志保存到: $exp/${dataset}-TopK-True-stochastic/eval-TopK-True-stochastic0.5-linear.log"
done

echo "所有评测完成！"