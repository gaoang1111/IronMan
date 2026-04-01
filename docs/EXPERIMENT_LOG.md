# 实验日志

> 本文档记录所有重要实验，包括成功和失败的。
> 格式：假设 → 配置 → 结果 → 结论 → 后续行动
>
> **约定**：
> - `[TODO]` 标记需要补充的信息（具体超参、wandb 链接、数据等）
> - 每个实验有唯一 ID（EXP-xxx），方便其他文档交叉引用
> - wandb 项目中有 15 个 run 记录可供回溯

---

## EXP-001: Mark-1 Linear 压缩基线

**版本**: Mark-1 | **日期**: [TODO]

**假设**: 简单的 Linear 投影足以从 Compressor hidden states 中提取有用的压缩信息。

**配置**:
- 压缩模块：单层 Linear，无 attention
- 注入方式：仅 embedding 层注入
- 冻结策略：Compressor 冻结，Generator backbone 冻结
- [TODO] 具体超参数（lr, batch_size, steps 等）

**结果**:
- Loss 前期下降正常
- Loss 后期卡住不降
- 未做生成测试

**结论**:
- Linear 投影能力不足，很可能只是透传了最后一个 token 的信息
- 没有机制保证提取的多样性

**后续**: → EXP-002（引入 CrossAttention）

---

## EXP-002: Mark-33 CrossAttention + Query 正交性

**版本**: Mark-33 | **日期**: [TODO]

**假设**: CrossAttention + 多个可训练 Query 能提取更丰富的压缩信息；正交性约束能防止 Query 退化。

**配置**:
- 压缩模块：Javis CrossAttention
- 可训练 Query 数量：[TODO]
- 正交性约束：cosine similarity penalty
- 注意力散度约束：鼓励不同 Query 关注不同位置
- 注入方式：仅 embedding 层注入
- [TODO] 具体超参数

**结果**:
- Needle-in-Haystack 表现差
- 梯度回传弱——从 Generator 到 Compressor 的梯度很低
- 模型到一定程度后不再改善 Compressor
- 未做生成验证

**结论**:
- CrossAttention 比 Linear 好，但仅在 embedding 层注入不够
- 32 层 Transformer 会把底层注入的信息"掩埋"
- 梯度路径太长，Compressor 无法得到有效的优化信号

**后续**: → EXP-003（Deep KV Injection）

---

## EXP-003: Mark-42 Deep KV Injection（全 32 层）

**版本**: Mark-42 (初始版本) | **日期**: [TODO]

**假设**: 在 Generator 的所有 32 层都注入压缩 KV，能解决信息掩埋和梯度回传问题。

**配置**:
- 注入层数：32 层全部
- 注入方式：残差加法 `orig_k + k_javis`
- Layer Gates：每层一个可学习标量，初始值 0.07
- [TODO] 具体超参数

**结果**:
- 训练过程中，各层的 Layer Gate 系数被模型自己逐渐**降低**
- 模型在尝试"关闭"不需要的注入层

**结论**:
- 并非所有层都需要压缩信息注入
- 模型通过降低 gate 值来自动选择
- 全层注入引入了不必要的参数和计算

**后续**: → EXP-004（3 层注入）

---

## EXP-004: Mark-42 Deep KV Injection（3 层）

**版本**: Mark-42 (最终版本) | **日期**: [TODO]

**假设**: 只在前/中/后三个代表性层注入，既能提供多层信息，又避免冗余。

**配置**:
- 注入层：约第 15 层、第 23 层、第 31 层
- 注入方式：残差加法 `orig_k + k_javis`
- Layer Gates：每层可学习标量，初始值 0.07
- [TODO] 具体超参数

**结果**:
- 训练 loss 降到约 **2.3** 后卡住
- **首次流式生成验证成功**——可以生成连贯文本
- Layer Gate 系数分布：后层 > 中层 > 前层
- 后层系数在训练中**自发上涨**（从 0.07 往上）

**结论**:
- 3 层注入足够，且训练更稳定
- 后面层更需要压缩信息（与 Llama 层间分工吻合）
- 模型自发上涨后层 gate，说明它"想要"未经前置层过滤的压缩信息
- 但 loss 卡在 2.3，生成能力来源不确定（可能是模型自身能力）

**关键指标**:
- [TODO] wandb 中的 loss 曲线截图/链接
- [TODO] Layer Gate 值的演变记录
- [TODO] 生成样例

**后续**: → EXP-005（生成验证）, EXP-006（Needle 测试）, EXP-007（Dynamic Q）

---

## EXP-005: 流式生成验证

**版本**: Mark-42 | **日期**: [TODO]

**假设**: 如果压缩信息有效，模型应该能在 Staircase Mask 下仅凭压缩向量 + `<eoc>` 生成连贯内容。

**配置**:
- 使用 Staircase Mask 模式生成
- 每个 chunk 的第一个 token 完全由前置压缩信息 + `<eoc>` 预测
- 用 `<eoc>` 而非 raw token 作为预测起点（防止压缩信息坍缩向 next token prediction）

**结果**:
- 可以生成连贯文本
- 但**无法确定**这是压缩信息的贡献还是模型本身的续写能力

**结论**:
- Staircase Mask 下能生成，在一定程度上说明压缩信息被学到了
  - 如果压缩向量是随机噪声，`<eoc>` 后不应能生成合理内容
- 但需要更严格的 ablation（例如：替换压缩向量为随机向量，对比生成质量）

**待做的 Ablation** (尚未执行):
- [ ] 随机压缩向量 vs 真实压缩向量的生成对比
- [ ] 去掉压缩向量（只保留 `<eoc>`）的生成质量
- [ ] 不同 buffer_size 下的生成质量对比

**后续**: 需要设计更严格的 ablation 来确认压缩信息的贡献

---

## EXP-006: Needle-in-Haystack 测试

**版本**: Mark-42 | **日期**: [TODO]

**假设**: 压缩记忆应该能在长上下文中定位和回忆特定信息（needle）。

**配置**:
- 在长文本中插入 needle，测试不同 offset 位置的 recall
- 对比有/无 Deep KV injection
- 数据存放：`_data/needle_scale*/`

**结果**:

| 条件 | 结果 |
|------|------|
| Needle 在 chunk 中间 (offset=8) | Recall 差 |
| Needle 在 chunk 末尾 (offset=9) | Recall 好很多 |
| 有 Deep KV injection | [TODO] 具体数据 |
| **无 Deep KV injection** | **反而更好** |

**关键发现**:
1. **Tail Bias 显著**：needle 位于 chunk 末尾时 recall 大幅提升
2. **KV injection 在 needle 场景下有负面效果**：训推不一致反而更好
3. Compressor 注意力分布：token 15 ≈ 0.17，均匀应该 ≈ 0.0625

**结论**:
- Tail Bias 是自回归模型的天然特性，不应硬性对抗，而应利用
- Deep KV 的直接残差注入可能对高层信息有"冲击"
- 需要重新思考注入方式（→ Sparse Attention 解耦方案）

**后续**: → EXP-008（Adaptive Chunking），→ Sparse Attention 方案（未实现）

---

## EXP-007: Tail Bias 诊断实验

**版本**: Mark-42 | **日期**: [TODO]

**假设**: Compressor 过度依赖 chunk 最后的 token，可以通过干预来验证。

### 子实验 A: Mask 最后一个 token

**配置**: 在 Compressor 处理 chunk 时，mask 掉最后一个 token
**结果**: Recall 改善
**结论**: 确认了 Compressor 过度依赖 tail

### 子实验 B: Delta-Q（自适应 query 偏置）

**配置**: 在 query 上加内容相关的偏置 `q = q_base + delta_q(chunk_mean)`
**结果**: Loss 下降更快，但注意力仍然 skewed
**结论**: 改善了优化过程，但没有从根本上解决 tail bias

### 子实验 C: 真实文本 vs 空格填充

**配置**: 对比 chunk 中填充真实文本 vs 空格的 recall
**结果**: 真实文本 recall 更差
**结论**: 背景文本会与 needle 竞争 Compressor 的注意力（background competition effect）

**后续**: → 转向利用 Tail Bias（Adaptive Chunking）

---

## EXP-008: Dynamic Q (Delta-Q) 效果验证

**版本**: Mark-42+ | **日期**: [TODO]

**假设**: 内容自适应的 query（delta-q）能让模型根据 chunk 内容调整信息提取策略。

**配置**:
- `q = q_base + delta_q(Linear(mean(chunk_hidden_states)))`
- 对比 delta-q 开启 vs 置零的注意力分布

**结果**:

| 条件 | 注意力分布 | Loss |
|------|-----------|------|
| Delta-Q 开启 | 后位 token 注意力更高（利用 tail bias） | 下降更顺滑 |
| Delta-Q = 0 | 注意力比较平均 | 下降较慢 |

**结论**:
- Delta-Q 有效：模型确实学会了根据内容自适应地提取信息
- Tail Bias 在 delta-q 下被"利用"而非"放大"——模型选择性地关注信息密度最高的位置
- 每次架构迭代都让训练 loss 下降更顺滑

**后续**: → EXP-009（Adaptive Chunking）

---

## EXP-009: Adaptive Chunking 基础验证

**版本**: adaptive-seg | **日期**: [TODO]

**假设**: 按语义边界切分（而非固定 16 token）能更好地利用 Tail Bias，提高压缩质量。

### 子实验 A: Pipeline 正确性验证 (fixed16_as_adaptive)

**配置**: 用固定 16-token 的切分结果喂入 adaptive pipeline
**结果**: 与 fixed pipeline 结果一致
**结论**: Adaptive pipeline 实现无 bug

### 子实验 B: Adaptive 分布的 Loss

**配置**: 使用真实的 adaptive 切分数据
**结果**: Loss 从 2.3 → 2.6
**结论**: 是分布偏移导致的（chunk 长度分布变了），非实现错误。需要 fine-tuning。

### 子实验 C: Surprisal 切分尝试

**配置**: 纯 surprisal 驱动的切分
**结果**: 切分逻辑很难控制
- Needle 插入位置：surprisal 很高
- Needle 内部：surprisal 很低
- Needle 结束：surprisal 又很高
**结论**: 纯 surprisal 不适合作为切分的主要依据，改为天然分隔符为主 + surprisal 辅助

**后续**: → Adaptive fine-tuning [TODO]

---

## EXP-010: Buffer Size 对比 + 压缩信息有效性验证

**版本**: Mark-42 | **日期**: ~2024-03

**假设**: 不同的 buffer size（允许看到多少前置 chunk 的 raw text）会影响性能。压缩信息应该在所有 buffer size 下都提供增益。

**配置**:
- **Oracle**: 原生 Llama 3.1 8B，完整上下文（不压缩，不截断），性能上界
- **Amnesiac**: 原生 Llama 3.1 8B，截断窗口上下文（无压缩记忆），性能下界
- **Mark-42**: IronCell 压缩记忆 + 不同大小的 raw text 窗口
- buffer=0: 只看压缩向量，不看前置 raw text
- buffer=1/2/4: 滑动窗口，允许看前 1/2/4 个 chunk 的 raw text
- 评测数据：511 samples（_data/eval_results/），233 samples（data/eval_results/）

**结果**:

| Setting | Avg Loss | PPL | Samples |
|---------|----------|-----|---------|
| **Oracle** (完整上下文) | **1.39** | **4.02** | 511 |
| Mark-42 + buffer=4 | 1.80 | 6.03 | 511 |
| Mark-42 + buffer=2 | 1.89 | 6.61 | 511 |
| Mark-42 + buffer=1 | 1.99 | 7.33 | 511 |
| Amnesiac + buffer=4 | 2.05 | 7.79 | 511 |
| Mark-42 + buffer=0 | 2.27 | 9.66 | 511 |
| Amnesiac + buffer=2 | 2.36 | 10.59 | 511 |
| Mark-42 + buffer=0 (新测) | 2.50 | 12.21 | 233 |
| **Amnesiac** (无压缩无窗口) | **2.82** | **16.76** | 511 |

**关键对比与结论**:

1. **压缩信息有效性（核心结论）**:
   - mark42-buf0 (2.27) vs amnesiac (2.82): **差 0.55 loss**，PPL 从 16.76 → 9.66
   - **纯压缩信息（无 raw 窗口）就能带来显著的 PPL 改善**，证明压缩信息不是噪声
   
2. **压缩信息与 raw 窗口互补**:
   - mark42-buf2 (1.89) vs amnesiac-buf2 (2.36): 同样 2-chunk 窗口，加压缩降 **0.47**
   - mark42-buf4 (1.80) vs amnesiac-buf4 (2.05): 同样 4-chunk 窗口，加压缩降 **0.25**
   - 窗口越大，压缩的边际贡献越小（合理：raw 窗口已覆盖更多上下文）

3. **压缩 gap**:
   - Oracle (1.39) vs mark42-buf4 (1.80): 最好的压缩方案与完整上下文差 **0.41**
   - 8:1 压缩确实丢失了一些信息，但差距可接受

4. **Buffer size 效果**:
   - 每增加一级 buffer，mark42 loss 降约 0.1
   - Amnesiac 的 buffer 效果更大（说明没有压缩信息时更依赖 raw 窗口）

**后续**: 这些数据已经回答了"压缩信息是否有效"的核心问题 — **答案是肯定的**。

---

## 待记录的实验

> 以下是可能存在但尚未整理的实验，需要回顾 wandb 记录来补充。

- [ ] Phase 2 (cmp) 训练实验——解冻 Compressor 的效果
- [ ] Phase 3 (full) 训练实验——全量 fine-tuning 的效果
- [ ] 不同 query 数量（num_queries=1 vs 2）的对比
- [ ] 不同 compression_rate（4:1 vs 8:1 vs 16:1）的对比
- [ ] Layer Gate 初始值的影响（0.01 vs 0.07 vs 0.1）
- [ ] Javis hidden distillation 实验（代码中有 `distill_hidden.py`）
- [ ] Javis attention distillation 实验（git history 中有 `javis proj&attn distill`）
- [ ] [TODO] 其他你记得的实验？

---

## wandb 项目信息

- 项目目录：`wandb/` (15 个 run)
- [TODO] wandb 项目 URL
- [TODO] 各 run 对应的实验 ID 映射
