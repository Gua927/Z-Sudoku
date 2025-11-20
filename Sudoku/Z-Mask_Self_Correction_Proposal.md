# Z-Mask 进阶：利用自生成偏置样本训练自修正能力

## 1. 核心洞察 (Key Insight)

**Z-Mask 不仅仅是一个 Mask 策略，它本质上是一个完美的“On-Policy 噪声构造器”。**

*   **传统噪声 (Random Mask)**：人工随机生成，与模型无关。模型学会修复人工错误，但无法应对推理时产生的自身逻辑错误。
*   **Z-Mask 噪声**：由模型当前的参数生成。它保留了那些**模型置信度高但实际上错误**的 Token。
    *   这精确模拟了推理时的**“盲目自信”陷阱**。
    *   这构成了最完美的**有偏样本 (Biased Samples)**：看起来通顺、模型也很确信、但实际上是错的。

## 2. 范式转变 (Paradigm Shift)

目前的训练逻辑限制了 Z-Mask 的潜力，我们需要从 **In-painting** 转向 **Rewriting**。

| 维度 | 当前模式 (In-painting) | 提议模式 (Rewriting) |
| :--- | :--- | :--- |
| **Loss 范围** | 仅计算 `[MASK]` 位置 | 计算 `[MASK]` + **错误 Token** 位置 |
| **模型心态** | “填空。如果上下文有错，就顺着错或忽略它。” | “填空。同时**质疑上下文**，把之前的错别字改过来！” |
| **获得能力** | 完形填空 (Completion) | **自修正 (Self-Correction)** |
| **推理行为** | 依赖上下文，误差累积 | **迭代细化**，每一步都在自我检查和修正 |

## 3. 实施方案 (Implementation)

只需要对 `trainer.py` 中的 `inner_forward` 进行微小的改造，扩大 Loss 的计算范围。

**原逻辑：**
```python
# 只对被 Mask 掉的位置计算 Loss
loss_mask = mask_t 
```

**新逻辑：**
```python
# 对 [MASK] 位置 以及 模型预测错误的位置 计算 Loss
# x_t: Z-Mask 构造的输入 (含模型生成的错误)
# x: Ground Truth
if is_z_mask_mode:
    # 找出所有与目标不一致的位置（排除 padding）
    rewriting_mask = (x_t != x) & (~src_mask)
    loss_mask = rewriting_mask
```

## 4. 预期收益 (Expected Benefits)

1.  **极致的训推一致性 (Train-Inference Consistency)**：
    模型在训练时就在处理“由自己生成的错误上下文”，这与推理时的场景完全一致，彻底解决 Exposure Bias。

2.  **获得“后悔药”能力**：
    在多步生成/扩散过程中，模型不再是一条道走到黑。如果第 $t$ 步生成了错误内容，第 $t-1$ 步模型有能力识别并将其“重写”为正确内容。

3.  **自动课程学习 (Auto-Curriculum)**：
    *   **初期**：模型弱 -> 错误多且明显 -> 学习基础纠错。
    *   **后期**：模型强 -> 错误少且隐蔽 -> 学习精细语义修正。
    *   训练难度自动跟随模型能力进化。

## 5. 建议与注意事项

*   **Warmup 策略**：训练初期模型生成的错误太多，全量纠错可能导致梯度不稳定。建议在训练初期降低 Z-Mask Loss 的权重，或混合使用 Standard Mask。
*   **权重平衡**：纠错（改错）通常比填空（生成）更难。可以结合现有的 `token_reweighting` (Focal Loss) 机制，自动给予纠错部分更高的 Loss 权重。
