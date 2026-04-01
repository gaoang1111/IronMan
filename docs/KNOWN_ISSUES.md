# 已知问题与技术决策

> 本文档记录：已修复的 bug、活跃的研究挑战、关键技术决策及其原因、潜在风险。
> 目的：避免重复踩坑，为新的 AI 对话提供决策上下文。

---

## 1. 已修复的问题

### 1.1 FSDP Deep KV Injection Bug

**问题**: `hack_llama_fsdp.py` 从 `kwargs` 而非 `DEEP_KV_CONTEXT` 读取 KV，导致 FSDP 模式下注入静默失败。

**症状**: FSDP 训练时 loss 不降，但没有报错。

**修复**: 统一使用 `DEEP_KV_CONTEXT` 全局字典。在重构中合并 `hack_llama_ddp.py` 和 `hack_llama_fsdp.py` 为统一的 `src/attention/patched_attention.py`。

**状态**: 已修复。

### 1.2 代码结构问题

**问题**:
- `train.py` 过大（641 行），混杂模型定义、训练逻辑、工具函数
- `hack_llama_ddp.py` 和 `hack_llama_fsdp.py` 大量重复代码
- 模型、训练、评估逻辑没有清晰分离

**修复**:
- 创建 `src/models/`（config, javis, iron_cell）
- 创建 `src/attention/`（统一的 TrainStepModule）
- 删除旧的 hack 文件
- 创建 `src/utils/`（checkpoint, data, distributed, model_loader, javis_init）

**状态**: 大部分已修复。`train.py` 仍可进一步简化。

### 1.3 Projector → Javis 重命名

**问题**: 早期版本中模块叫 `projector`，后来重命名为 `javis`，导致旧 checkpoint 的 state_dict key 不匹配。

**修复**: `IronCellModel._load_from_state_dict` 中做了 `projector.*` → `javis.*` 的自动转换。`IronCellConfig` 中 `trainable_components` 如果包含 `"projector"` 会自动替换为 `"javis"`。

**状态**: 已修复，向后兼容。

---

## 2. 活跃的研究挑战

### 2.1 Loss 卡在 ~2.3

**现象**: Fixed chunking 下训练 loss 降到约 2.3 后不再下降。

**可能原因**:
- 8:1 压缩率（16 tokens → 2 vectors）可能存在信息瓶颈
- Chunk 内部信息被模糊化——压缩向量无法精确保留所有 token 级信息
- Staircase Mask 限制了 Generator 能利用的信息量
- [TODO] 与 oracle baseline（直接看 raw text）的 loss 差距是多少？这个差距能说明压缩的信息损失程度

**相关实验**: EXP-004, EXP-010

### 2.2 Tail Bias Problem

**现象**: Compressor 退化为优先读取每个 chunk 最后 1-2 个 token。

**根因**: 自回归 LLM 的天然特性——`h_i = f(x_0...x_i)`，后面的 token hidden state 包含更完整的上下文信息。

**当前策略**: 从"修复"转向"利用"——
- Adaptive Chunking：按语义边界切分，让每个 chunk 的最后一个 token 天然包含该语义单元的摘要
- Delta-Q：让模型自适应地利用 tail bias

**未解决的问题**:
- Chunk 前部信息是否被完全丢弃？
- 是否需要额外机制来保证 chunk 内部信息的均匀提取？
- [TODO] 尝试双向 attention 的 Compressor？（但会破坏自回归特性）

### 2.3 Deep KV 直接残差注入的"冲击"

**现象**: Needle 测试中，不用 Deep KV injection 反而 recall 更好。

**推测原因**: 高层 KV 的直接残差加法（`orig_k + k_javis`）可能破坏了原有的 attention pattern。压缩信息的"风格"与该层已有的 KV 信息不匹配，导致注意力分数计算被干扰。

**拟议解决方案**: Sparse Attention 解耦（详见 `docs/ROADMAP.md`）

**相关实验**: EXP-006

### 2.4 压缩信息有效性 — 已通过 Buffer Size 实验部分验证

**结论更新**: Buffer size 对比实验（EXP-010）已经证明压缩信息有效：
- mark42-buf0 (loss 2.27) vs amnesiac (loss 2.82)：纯压缩信息降 0.55 loss，PPL 从 16.76 → 9.66
- 在所有 buffer size 下，加压缩信息都优于不加

**仍需验证的部分**:
- 随机压缩向量 ablation（排除 `<eoc>` embedding 自身贡献的可能性）
- 压缩信息在生成质量（而非 PPL）上的贡献

**相关实验**: EXP-005, EXP-010

---

## 3. 关键技术决策及其原因

### 3.1 为什么选择残差注入而非覆盖

**决策**: `key_states[:, :, pos, :] += k_javis` 而非 `= k_javis`

**原因**:
- 覆盖方式会导致原始 token 语义丢失
- 实验中覆盖方式导致梯度消失
- 残差加法保留了原始信息，压缩信息作为"增强"

**权衡**: 残差加法可能导致高层信息"冲击"（见 2.3），需要在注入强度上小心控制。

### 3.2 为什么选 3 层注入而非 32 层

**决策**: 只在约第 15/23/31 层注入。

**原因**:
- 32 层全注入时，模型自动将大部分层的 gate 降到接近 0
- 3 层注入减少参数量和计算开销
- 选择前/中/后代表不同层间分工

**观察**: 后层系数 > 中层 > 前层，且后层系数训练中自发上涨。

### 3.3 为什么用 `<eoc>` 预测而非 raw token

**决策**: 用 `<eoc>` 作为 chunk 的预测锚点。

**原因**: 如果直接预测下一个 raw token，压缩信息可能坍缩为"记住下一个 token"的捷径，而不是学习有效的压缩表示。`<eoc>` 是一个特殊 token，迫使模型从压缩信息中恢复更通用的语义。

### 3.4 为什么 Staircase Mask 限制只看前置压缩信息

**决策**: Chunk N 的 raw tokens 不能看到 Chunk N-1 的 raw tokens，只能看到其压缩信息。

**原因**: 如果允许直接看前置 raw tokens，Generator 会完全忽略压缩向量（走捷径）。Staircase Mask 强制 Generator 依赖压缩信息，推动 Compressor + Javis 学习有效的压缩。

**权衡**: 这个约束在训练时是必要的，但在推理时可能过于严格。Buffer size > 0 允许部分放宽。

### 3.5 为什么用同源模型而非异构模型

**决策**: Compressor 和 Generator 都从同一个 Llama 3.1 8B checkpoint 出发。

**原因**:
- Tokenizer 一致——不需要跨词汇表的映射
- Embedding 对齐——压缩向量和生成器的输入在同一个向量空间
- 层间语义结构共享——Javis 只需要学"如何压缩"，不需要学"如何对齐"
- 生物学动机：同源分化无排异

**权衡**: 两个完整的 8B 模型，显存和计算开销较大。未来可以探索用更小的模型做 Compressor。

### 3.6 为什么 Adaptive 用天然分隔符而非纯 Surprisal

**决策**: 以句号/感叹号/问号/换行等天然分隔符为主，surprisal 仅作辅助。

**原因**:
- 纯 surprisal 的切分边界不稳定（如 needle 场景：开头高、中间低、结尾高）
- 天然分隔符在语义上更可靠——句号确实是语义单元的自然边界
- Surprisal 用于辅助判断：边界 surprisal 低 → 可以合并；长段内 surprisal 谷值 → 可以拆分

---

## 4. 潜在风险与待验证假设

### 4.1 "同源"的优势是否真的显著？

- 假设：同源模型的前置相似性使得桥接更容易
- 风险：没有和异构模型的对比实验，无法定量证明同源优势
- 验证方案：用不同的预训练模型做 Compressor（如 Mistral），对比 Javis 的训练难度和最终效果

### 4.2 压缩信息是否真的被利用？

- 假设：Generator 利用了压缩向量中的信息
- 风险：Generator 可能主要依赖自身的 LM 能力 + `<eoc>` token 的 embedding
- 验证方案：随机向量 ablation（见 2.4）

### 4.3 8:1 压缩率是否合理？

- 假设：16 tokens → 2 vectors 是合理的压缩率
- 风险：可能丢失了太多细节信息，导致 loss 无法继续下降
- 验证方案：对比 4:1、8:1、16:1 的 loss 和生成质量

### 4.4 训练数据是否足够？

- 当前数据规模：[TODO] 训练数据量？
- 风险：如果数据量不够，模型可能没有充分学习到压缩模式
- 验证方案：在更大数据集上训练，观察 loss 是否继续下降
