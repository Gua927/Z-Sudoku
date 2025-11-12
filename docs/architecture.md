# EasyMDM 代码库概览与上手指南（聚焦 ℓ=1 模型）

本文旨在帮助你快速理解本仓库（`EasyMDM/`）的整体实现方式，尤其聚焦于 MDLM/MDLM-Prime 在 target_length=1（下文简写为“ℓ=1”或“l1 模型”）时的前向/反向过程、关键接口与采样路径。

—— 如果你只想快速跑通 l1 模型训练/评估/采样，请先看“如何运行”与“关键超参”两节；想深入原理与代码流转，请继续阅读“架构与主要模块”和“训练与损失”、“采样与时间调度”等章节。


## 目录

- 目录结构速览
- 架构与主要模块
- 训练与损失（前向/反向链路）
- 采样与时间调度（DDPM / Analytic / Cache）
- 评测：PPL、GenPPL、Mauve 与可选指标
- 关键超参与常见配置（聚焦 l1）
- 快速上手：命令示例
- 重要类/函数速查表
- 进阶：ℓ>1 的 Prime 模式有何不同


## 目录结构速览

仓库的核心逻辑集中在以下文件夹（只列最相关部分）：

- `main.py`：Hydra 启动入口。模式（训练/评估/采样）切换、Trainer 构建、数据管线与模型实例化。
- `diffusion.py`：核心 LightningModule（类 `Diffusion`）。定义前向/损失、采样器、噪声调度与评测。
- `models/`
  - `dit.py`：DiT/Perceiver-DiT 骨干，适用于双向掩码式建模；l1 场景通常走 `DIT`。
  - `autoregressive.py`：自回归（AR）骨干，因果注意力；当 `parameterization=ar` 或 `backbone=ar` 时使用。
  - `dimamba.py`：基于 BiMamba 的双向混合器骨干，支持带时间嵌入的 Masked LM。
- `dataloader.py`：数据集下载、分词与拼接（wrap 模式）、DataLoader 与容错采样器。
- `utils/`
  - `noise_schedule.py`：噪声调度（loglinear/cosine/linear/…）。
  - `utils.py`：Prime 模式的子词编码/解码器、marginalization 等。
  - `eval_utils.py`：Mauve、GenPPL、熵等评测。
- `configs/`：Hydra 配置，`config.yaml` 为默认总入口；子配置包含 `model/`、`noise/`、`data/` 等。


## 架构与主要模块

### 启动与模式（`main.py`）

- 使用 Hydra 组织配置；运行时根据 `mode` 分支：
  - `train`：训练（Lightning Trainer + 回调 + EMA）；
  - `ppl_eval`：只做困惑度评估（零样本 PPL 或其他数据集）；
  - `sample_and_save`：离线采样并保存到 JSON；
  - `sample_quality_eval`：对已生成文本做质量评测（Mauve/自 BLEU/重复率/熵/GenPPL）。
- 构建 Tokenizer 与 DataLoaders：`dataloader.get_tokenizer`、`dataloader.get_dataloaders`。
- 模型载入：
  - 训练新模型：`diffusion.Diffusion(config, tokenizer)`。
  - 从 checkpoint 载入：`Diffusion.load_from_checkpoint(...)` 或 HuggingFace Hub。

### Diffusion LightningModule（`diffusion.py`）

类 `Diffusion` 统一封装了 Masked Diffusion 的训练、评估与采样逻辑，支持三类“参数化”方式：
- `parameterization=subs`（默认）：“SUBS”参数化，适配文本离散扩散；
- `parameterization=d3pm`：离散时间（T>0）版本；
- `parameterization=sedd`：SEDD 损失（score-entropy）。

并支持多种骨干（`backbone`）：
- `dit`（默认，l1 常用）：见 `models/dit.DIT`；
- `dimamba`：见 `models/dimamba.DiMamba`；
- `ar`：自回归骨干，见 `models/autoregressive.AR`（当 `parameterization=ar` 时）。

ℓ=1 时，不涉及 `wrapped_encoder/decoder` 与 carry-over；词表大小 `vocab_size` 等于 tokenizer 词表（若 tokenizer 无 mask_token，会在训练内部分配新 mask id）。


## 训练与损失（前向/反向链路）

以默认 `parameterization=subs`、`backbone=dit`、ℓ=1 为例。

### 数据与输入形状

- 训练 batch：`batch['input_ids']` 与 `attention_mask`；长度 `config.model.length`。
- l1 情况下，Diffusion 的“扩散输入长度”= `model.length`，与分词后的序列长度一致。

### 前向核心链路

1) 采样时间 t 与噪声（`_forward_pass_diffusion`）
- 连续时间（`T=0`）：从均匀分布采样 t∈(sampling_eps, 1]，经 `noise_schedule.get_noise`（默认 loglinear）得到：
  - 总噪声 σ(t);
  - 对应的移动概率 `move_chance = 1 - exp(-σ)`（或按 `change_of_variables` 分支计算）。

2) 构造带噪输入 x_t（`q_xt`）
- 对输入 token x₀ 的每个位置独立以 `move_chance(t)` 的概率置为 mask，否则保留原 token，得到 x_t。

3) Backbone 前向（`forward`）
- `self.backbone(x_t, sigma)` 输出 logits；随后根据参数化（SUBS/D3PM/SEDD）映射为 log 概率（或 log-score）。
  - SUBS（ℓ=1）：`_subs_parameterization` 会将 mask 通道 log 概率置为 −∞，未被 mask 的位置只允许“保持原词”这一类别（其余类别 −∞）。这等价于对“masked/unmasked”状态施加结构化约束。

4) 损失计算
- 连续时间 SUBS（`T=0`）：取模型对真值 x₀ 的 log 概率，并乘上来自噪声微分项的权重，形成 token 级损失，最后与 `attention_mask` 相乘后做平均得到 `Loss.loss`。
- D3PM（`T>0`）：离散时间版本，包含 variational bound 项与（对 D3PM）重构项；
- SEDD：使用 `_score_entropy` 实现的 score-entropy loss。

5) 反向传播
- Lightning 常规优化步骤；若开启 EMA（`training.ema>0`），使用 `models/ema.ExponentialMovingAverage` 跟踪影子权重。


### 简化契约（l1/SUBS）

- 输入：
  - x₀: LongTensor [B, L]
  - attention_mask: FloatTensor [B, L]
- 中间：
  - t ~ U(eps, 1]；σ = total_noise(t)，move_chance = 1-exp(-σ)
  - x_t = q_xt(x₀, move_chance)
  - log_pθ = forward(x_t, σ)
- 输出：
  - token_nll = 平均到每个有效 token 的负对数似然
  - 训练日志：train/val/test ppl、bpd、nll


## 采样与时间调度（`_sample`）

采样阶段从全 mask 的先验开始，沿时间从 t=1 → eps 逐步“去掩码”。支持三条主干：

- `predictor=ddpm`（`_ddpm_update`）：严格用 σ(t), σ(t-dt) 计算 `move_chance_t` 与 `move_chance_s`，本步释放的概率质量约为 `(move_chance_t - move_chance_s)`，保留为 mask 的概率约 `move_chance_s`。
- `predictor=ddpm_cache`（`_ddpm_caching_update`，默认）：采用近似 `move_chance ≈ t`，简化计算并支持 p(x₀) 缓存（若输入不变且非时间条件模型）。
- `predictor=analytic`：解析步（`_analytic_update` / `_denoiser_update`），基于 score 与转移矩阵；用于理论对比或可选的最后一步去噪。

采样策略：
- `sampling.sample_method`：`origin`（所有 mask 位置都可更新）、`top-k`、`top-k-margin`。
  - 后两者会额外“选择本步允许更新的位置子集”，k 随时间（基于 `decode_prob=1-move_chance_s`）隐式调度；并在词表层面（或按 margin）做截断。若你只希望截断词表而不限制位置，建议改造 `_compute_update_mask` 逻辑。
- `sampling.nucleus_p`：先对 p(x₀) 做 nucleus 截断（p<1 时生效）。
- `sampling.temperature`：top-k* 路径下的温度缩放（1.0 为默认）。
- `sampling.noise_removal`：在最后一步再做一次“去掩码/重采样”（常配合 argmax）。

ℓ=1 时，不涉及编码/解码器；ℓ>1 时改为在“编码后的子词空间”做同样的扩散/采样，再用 `wrapped_decoder` 还原。


## 评测

- 零样本 PPL：`main.py` 的 `ppl_eval` 模式使用 `trainer.validate` 在验证集上计算 token 级 NLL/Perplexity。
- 生成困惑度（GenPPL）：将生成文本送入“外部 AR 模型”（默认从 `eval.gen_ppl_eval_model_name_or_path` 懒加载，例如 Qwen/LLaMA/GPT2），按其分词重新计算 NLL。
- 质量指标：`utils/eval_utils.py` 提供 Mauve、自 BLEU、重复率、熵等；`mode=sample_quality_eval` 会读入 JSON 文本并计算。


## 关键超参与常见配置（聚焦 l1）

- `parameterization=subs`：l1 默认；连续时间建议 `T=0`。
- `backbone=dit`：l1 首选骨干；可选 `dimamba`/`ar`。
- `noise=loglinear`：默认噪声调度；`training.sampling_eps` 为最小时间。
- `prime.target_length=1`：即 l1；不会启用 `wrapped_encoder/decoder`。
- `sampling.predictor=ddpm_cache`：采样主干，速度/效果折中；
- `sampling.steps`：扩散步数，常见 256–2048；
- `sampling.sample_method`：`origin`（推荐基准）、或 `top-k`/`top-k-margin`（若使用，建议仅截断词表，不限制位置，见“进阶建议”）。


## 快速上手：命令示例

训练（OpenWebText 示例，关闭所有额外评测）：

```bash
python -u -m main \
  data=openwebtext-split data.cache_dir=/app/huggingface_cache \
  loader.batch_size=16 loader.eval_batch_size=16 \
  model=small backbone=dit parameterization=subs \
  prime.target_length=1 prime.carry_over=True \
  eval.compute_zero_shot=False eval.conditional_sampling=False \
  eval.compute_generative_perplexity=False \
  sampling.predictor=ddpm_cache sampling.steps=512
```

零样本 PPL 评估已有 checkpoint：

```bash
python -u -m main mode=ppl_eval \
  eval.compute_zero_shot=True eval.generate_samples=False \
  eval.checkpoint_path=/path/to/checkpoint.ckpt \
  data=wikitext2 data.cache_dir=/your/hf/cache \
  loader.batch_size=16 loader.eval_batch_size=16 \
  model=small prime.target_length=1
```

生成 + GenPPL（注意需要能加载外部 AR 模型）：

```bash
python -u -m main mode=ppl_eval \
  eval.checkpoint_path=/path/to/checkpoint.ckpt \
  eval.compute_generative_perplexity=True \
  eval.gen_ppl_eval_model_name_or_path=Qwen/Qwen2.5-7b \
  eval.perplexity_batch_size=20 \
  loader.eval_batch_size=20 sampling.num_sample_batches=2 \
  sampling.steps=512 sampling.sample_method=origin
```


## 重要类/函数速查表

- 训练/评估入口：`main.main()`
  - 训练：`_train()`
  - PPL 评估：`_ppl_eval()`
  - 采样与保存：`_sample_and_save()`
- LightningModule：`diffusion.Diffusion`
  - 损失：`_forward_pass_diffusion()`（采样 t → q(x_t|x_0) → forward → loss）
  - 前向：`forward(x, sigma)`（按 parameterization 映射 logits）
  - 采样：`_sample()`→`_ddpm_update()` / `_ddpm_caching_update()` / `_analytic_update()`
  - 噪声：`utils.noise_schedule.get_noise()`（默认 LogLinear）
  - 生成困惑度：`compute_generative_perplexity()`（懒加载外部 AR Tokenizer+Model）
- 数据管线：`dataloader.get_dataloaders()` / `get_dataset()` / `get_tokenizer()`
- 骨干：
  - l1/双向 Masked LM：`models/dit.DIT`
  - 自回归：`models/autoregressive.AR`
  - BiMamba：`models/dimamba.DiMamba`


## 进阶：ℓ>1 的 Prime 模式有何不同

当 `prime.target_length>1` 时：
- 使用 `utils.encoder/decoder` 将每个 token 拆为 ℓ 个“子 token”（进制为 `prime.base`，满足 `base^ℓ ≥ vocab_size`）。
- 扩散过程在“子 token 序列”上进行，mask 索引改为 `mask_index_enc = base`；
- 训练与采样仍复用同一条链路，只是 `forward`/`loss` 在子词空间运作，最后通过 `wrapped_decoder` 还原；
- 可选 carry-over/marginalization（`prime.carry_over=True`）以对齐类别边界。


## 常见问题与建议（与 GenPPL、top-k 相关）

- 如你观察到 `top-k`/`top-k-margin` 在 GenPPL 上高于 `origin`：
  - 当前实现里同时“截断词表”和“限制本步允许更新的位置”，容易导致早期上下文建立不足、后期一次性补齐，进而被外部 AR 模型判定似然较低；
  - 建议首先只做“词表截断”不限制位置（修改 `_compute_update_mask`），或为位置更新设计更平滑的 k 调度，并在末几步强制所有残余 mask 参与 2–3 次 refinement 再去噪。


---

若你需要：
- 我可以在 `diffusion._sample` 中加入轻量日志，统计每步释放的 token 数与位置分布；
- 或提供只截断词表的 `top-k` 改造示例，便于与 `origin` 做公平对比。
