# 研究路线图

> 本文档整理未来的研究方向、待验证假设、优先级排序。
> 标注 `[TODO: priority]` 的条目需要讨论确定优先级。

---

## 1. 短期目标（下一步实验）

### 1.1 Sparse Attention 解耦方案

**优先级**: [TODO: priority]

**背景**: 当前 Deep KV 的直接残差注入（`orig_k + k_javis`）在 Needle 测试中表现不佳，可能对高层 attention pattern 有"冲击"。

**方案**:
```
原始 Attention：  Q_orig × K_orig → Attn → V_orig → hidden_orig  （不修改）
压缩 Attention：  Q_orig × K_compressed → similarity → V_compressed × weight → delta_hidden
最终输出：       hidden = hidden_orig + delta_hidden
```

**具体步骤**:
1. 原始 attention 照常计算（不修改原始 KV）
2. 当前 query 与历史压缩 K 计算相似度
3. 取对应压缩 V 加权求和
4. 结果加到 hidden state（而非 KV）
5. 本质：从 KV 空间解耦到 hidden state 空间的 sparse cross-attention

**预期效果**:
- 避免直接修改 KV 对原始 attention 的污染
- 保留 Deep KV 跨层提供压缩信息的能力
- 可能需要额外的可学习参数（gate/scale）

**风险**:
- 增加了计算量（多一次 attention）
- 新引入的模块需要训练，可能影响收敛
- 需要重新设计梯度流

### 1.2 Adaptive Chunking Fine-Tuning

**优先级**: [TODO: priority]

**背景**: Adaptive 切分后 loss 从 2.3 → 2.6，是分布偏移导致的，需要在新分布上 fine-tuning。

**步骤**:
1. 在 adaptive 数据上从 `phase-full_step_480` checkpoint 继续训练
2. 监控 loss 是否回降到 2.3 或更低
3. 对比 fixed vs adaptive 的 Needle 表现
4. 对比 fixed vs adaptive 的生成质量

**预期**:
- Loss 应该能回降到 2.3 附近
- 如果 Tail Bias 利用得当，Needle recall 应该改善

### 1.3 压缩信息有效性的严格 Ablation

**优先级**: [TODO: priority]

**背景**: 需要确认生成能力确实来自压缩信息。

**Ablation 设计**:

| 实验 | 设定 | 预期 |
|------|------|------|
| **Baseline** | 正常压缩向量 | 正常生成 |
| **Random V** | 替换压缩向量为随机向量（同分布） | 生成质量显著下降 |
| **Zero V** | 压缩向量全部置零 | 生成质量下降 |
| **No V** | 完全去掉压缩向量，只保留 `<eoc>` | 生成质量应该最差 |
| **Shuffled V** | 打乱压缩向量的顺序（chunk 顺序错乱） | 如果压缩有效，应该导致不连贯 |

---

## 2. 中期目标

### 2.1 Position-Invariant Compression

**背景**: Tail Bias 说明压缩质量与信息在 chunk 中的位置强相关。理想情况下，压缩应该对位置不敏感。

**可能方向**:
- 双向 Compressor（放弃自回归约束，但只在压缩阶段）
- 多轮提取：先粗后细
- 位置增强的 query：让 query 关注特定位置范围
- [TODO] 其他想法？

### 2.2 压缩率探索

**实验**:
- 4:1（8 tokens → 2 vectors）
- 8:1（16 tokens → 2 vectors）← 当前设定
- 16:1（16 tokens → 1 vector）
- 对比 loss、Needle recall、生成质量

### 2.3 更大规模验证

- 更大的训练数据集
- 更长的训练步数
- 更长的输入序列（测试压缩记忆在极长上下文下的表现）
- [TODO] 有特定的 benchmark 想跑吗？（如 LongBench、SCROLLS）

### 2.4 Compressor 架构探索

- 更小的 Compressor（如 1B/3B）是否足够？
- 非自回归 Compressor？
- Compressor 是否需要和 Generator 同步更新？

---

## 3. 长期愿景

### 3.1 同源分化范式的扩展

当前验证的是"压缩记忆"这一个功能分化。更长远的问题：

- 能否分化出其他功能的模块？例如：
  - 检索模块（从外部知识库检索）
  - 推理模块（专门做逻辑推理）
  - 编辑模块（专门做文本修改）
- 多个分化模块如何协同？
- 动态分化：根据任务类型选择/激活不同的分化模块
- [TODO] 你对这个范式的终极愿景是什么？

### 3.2 论文方向

**如果要写论文，核心贡献可以是**:

1. **范式贡献**：同源模型分化（Homologous Model Differentiation）作为模型能力扩张的新思路
2. **架构贡献**：Deep KV Injection + Staircase Mask + Adaptive Chunking
3. **实验发现**：
   - Tail Bias 的发现与利用
   - Layer Gate 系数的自发分化
   - Delta-Q 的有效性
   - 训推不一致的 Needle 现象

**需要补充的实验**:
- 与 Gisting/AutoCompressor/ICAE 的直接对比
- 压缩信息有效性的 ablation
- 不同压缩率的对比
- Scaling 实验（模型大小、数据量）
- [TODO] 目标投稿哪些会议/期刊？

---

## 4. 待验证假设清单

| ID | 假设 | 状态 | 优先级 |
|----|------|------|--------|
| H-01 | Sparse attention 解耦能解决 Deep KV "冲击"问题 | 未验证 | [TODO] |
| H-02 | Adaptive chunking + fine-tuning 能降低 loss 到 ≤2.3 | 未验证 | [TODO] |
| H-03 | 压缩向量确实包含有效信息（非噪声） | 部分验证（生成可 work） | [TODO] |
| H-04 | 同源模型比异构模型更容易桥接 | 未验证 | [TODO] |
| H-05 | 后层 gate 系数上涨说明模型需要 deep injection | 观察到现象 | [TODO] |
| H-06 | Adaptive chunking 能改善 Needle recall | 未验证 | [TODO] |
| H-07 | 更高/更低的压缩率有明确的 trade-off | 未验证 | [TODO] |
| H-08 | 更小的 Compressor 足以完成压缩任务 | 未验证 | [TODO] |
| [TODO] | 其他你想验证的假设？ | | |

---

## 5. 技术债务

| 项目 | 描述 | 优先级 |
|------|------|--------|
| `train.py` 简化 | 仍然较复杂，可以进一步拆分 | 低 |
| 测试覆盖 | 无正式测试套件，只有零散的验证脚本 | 中 |
| Linter 配置 | 无 flake8/ruff/black/mypy 配置 | 低 |
| wandb 实验管理 | 15 个 run 缺少系统化的命名和标注 | 中 |
| Eval 入口统一 | `eval_ppl.py` 和 `eval_with_buffer.py` 两套入口，容易混淆 | 中 |
| checkpoint 管理 | 缺少 checkpoint 的元数据记录（哪个 checkpoint 对应哪个实验配置） | 中 |
