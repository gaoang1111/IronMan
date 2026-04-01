# 版本演进记录

> 本文档记录 IronCell 从初始构想到当前状态的完整迭代历程。
> 每个版本记录：做了什么、为什么这么做、发现了什么、为什么要迭代到下一个版本。

---

## 演进总览

```
Mark-1 ──→ Mark-33 ──→ Mark-42 ──→ Mark-42+ (Dynamic Q) ──→ Adaptive Segmentation
 Linear     CrossAttn    Deep KV       Delta-Q                  自适应分块
 注入        压缩提取      多层注入       内容自适应                 利用 Tail Bias
```

**核心观察**：每次迭代变化后，训练 loss 下降都会更顺滑——说明每一步改进确实在降低优化难度。

---

## Mark-1：线性压缩（初始验证）

**Git ref**: `ced270a` — `feat: Mark 1`

### 做了什么
- 单层 Linear 投影作为压缩模块
- 压缩向量仅在 embedding 层注入 Generator
- 实现了基本的 Zipper Layout + Staircase Mask

### 验证情况
- 仅观察训练 loss，**未做生成测试**

### 发现
- Loss 前期下降正常，后期卡住不降
- 推测原因：Linear 投影太简单，可能只是把最后一个 token 的信息透传过去了，没有真正学会压缩提取

### 导向下一步的原因
- 需要更有效的信息提取机制，不能只是线性映射
- 需要保证提取的多样性，防止只读最后一个 token

---

## Mark-33：CrossAttn 压缩提取

**Git ref**: `origin/ga/mark-33`

### 做了什么
- 引入 **Javis CrossAttention 模块**，替代简单 Linear
- 设计了 **可训练 Query**：多个 learnable query 去 attend compressor 的 hidden states
- 添加 **正交性约束**：惩罚 query 之间的 cosine similarity，防止提取同质化
- 添加 **注意力散度**约束：鼓励不同 query 关注不同位置

### 验证情况
- 训练观察，**未做生成验证**
- 做了 Needle-in-Haystack 测试

### 发现
- Needle 表现差——压缩信息不足以支持精确检索
- 梯度回传低——从 Generator 到 Compressor 的梯度很弱
- 模型到一定程度后"不愿意"修改 Compressor（优化停滞）

### 导向下一步的原因
- 信息只在 embedding 层注入，经过 32 层 Transformer 后被逐渐"掩埋"
- 梯度回传路径太长，优化信号到不了 Compressor
- 联想到**跨模型 ResNet**：需要在更深的层直接注入压缩信息

---

## Mark-42：Deep KV Injection

**Git ref**: `0e17105` — `feat: Mark42-deep kv injection & double query javis & test to release`
**PR**: `#1` (merged to main)

### 做了什么
- 实现 **Deep KV Injection**：Javis 输出经过投影后，残差注入到 Generator 的多个 Transformer 层的 KV 中
- 初始尝试：注入到全部 32 层
- 发现问题后调整：只注入到 3 层（约第 15 层、第 23 层、第 31 层）
- 引入 **Layer Gates**：每层一个可学习标量控制注入强度（初始 0.07）
- 注入方式：**残差加法** `orig_k + k_javis`（而非覆盖替换）
- 实现了 monkey-patch 机制（`DEEP_KV_CONTEXT`）避免修改 HuggingFace 源码

### 验证情况
- **首次做流式生成验证**——可以正常生成！
- PPL 测试：loss 降到约 2.3 后卡住
- 但不确定生成能力是来自压缩信息还是模型本身的能力

### 关键发现

**32层全注入的问题**：
- 训练过程中，各层的注入系数（Layer Gates）被模型自己逐渐降低
- 说明模型在尝试"关闭"不需要的注入层

**3层注入的观察**：
- 后面层（~31层）的系数 > 中间层（~23层）> 前面层（~15层）
- 后面层的系数在训练中还会**自发上涨**（从 0.07 往上涨）
- 解读：与 Llama 架构吻合——后面层处理更融合的全局信息，更需要清晰的压缩信息注入

**Needle 测试的意外发现**：
- 不用 KV inject 反而 recall 更好
- 训练和推理不一致（训练时有 inject，推理时无 inject）反而可以 work
- 推测：高层信息的直接残差加法可能有"冲击"，破坏了原有的注意力模式

### 导向下一步的原因
- Loss 卡在 2.3，需要更多手段
- 生成能力的来源不确定，需要更多验证
- Deep KV 的直接残差注入可能不是最优方式

---

## Mark-42+：Dynamic Q（Delta-Q）

**Git ref**: `887f09d` — `feat: alpha for dynamic q`，`5e7877f` — `feat: dynamicq`
**分支**: `ga/javis_dynamic_q`

### 做了什么
- 在 Javis 的 query 上引入 **Delta-Q 机制**：`q = q_base + delta_q(mean(chunk_hidden))`
- `delta_q` 由一个 Linear 层计算，输入是 chunk hidden states 的均值
- 目的：让 query 根据 chunk 内容进行自适应调整，而非纯静态提取

### 验证情况
- 注意力分布分析
- 训练 loss 曲线对比

### 关键发现

**注意力分布变化**：
- Delta-Q 开启时：更靠后的 token 获得更高的注意力分数（利用了 tail bias）
- Delta-Q 置为 0 时：注意力分布变得比较平均
- 说明 delta-q 确实让模型学会了**内容自适应的信息提取**

**训练表现**：
- 引入 delta-q 后，训练 loss 下降更顺滑
- 每次架构迭代都让优化变得更容易

### 意义
- 证明模型可以在原有提取模式下进行适应性调整
- Tail bias 不仅仅是一个"问题"，也可以被利用——让模型自动把注意力放到信息最密集的位置

---

## Adaptive Segmentation（当前阶段）

**分支**: `ga/adaptive-seg`（当前工作分支）
**Git ref**: `47c22e6` — `chore: save all for reset`

### 做了什么
- 实现了**自适应分块**（Adaptive Chunking），替代固定 16-token 切分
- 切分策略：
  1. 按强分隔符切分（句号、感叹号、问号、换行）
  2. 过短的块：如果边界 surprisal 低，则合并
  3. 过长的块：按弱标点或 surprisal 谷值拆分

### 改进动机

**来自 Needle 测试的关键发现**：
- 当 needle 恰好落在 chunk 边界时（即成为某个 chunk 的最后几个 token），recall 会好很多
- 这是因为 **Tail Bias 是自回归模型的天然特性**——最后位置的 token 信息最丰富

**思路转变**：
- 之前把 Tail Bias 当作需要"修复"的问题
- 现在转变为**利用** Tail Bias：让同质的信息自然地落在一个 chunk 内
- 每个 chunk 的最后一个 token 天然包含了该 chunk 的"摘要"
- 天然的语义边界（句号等）正好把同质信息分到一起

### Surprisal 切分的困难
- 纯 surprisal 切分逻辑很难控制
- 例子：插入一个 needle 时，开始位置 surprisal 很高，needle 内部很低，结尾又很高
- 切分边界不稳定
- 因此选择以天然分隔符为主，surprisal 只作为合并/拆分的辅助判断

### 验证情况
- `fixed16_as_adaptive` 测试通过——pipeline 无 bug
- Loss 从 2.3 → 2.6 是分布偏移（chunk 长度分布变了），非实现错误
- 需要进一步 fine-tuning 来适应新的分块分布

### 当前状态
- 离线切分逻辑已实现
- 需要在新分布上做 fine-tuning
- [TODO] Fine-tuning 结果？

---

## 下一步方向（未实现）

### Sparse Attention 解耦方案

**动机**：
- Deep KV 的直接残差注入（`orig_k + k_javis`）可能对高层信息造成"冲击"
- Needle 测试中不用 inject 反而更好，进一步支持这个推测

**方案构想**：
- 原始 attention 照常计算（不修改原始 KV）
- 新增一个独立的 attention 模块
- 当前 query 与历史压缩 K 计算相似度
- 取对应压缩 V 加权求和
- 结果加到 hidden state 上（而非 KV 上）
- 本质：把 Deep KV 从 KV 空间解耦到 hidden state 空间

**预期效果**：
- 避免直接修改 KV 对原始 attention 模式的污染
- 保留 Deep KV 提供跨层压缩信息的能力
- 类似 sparse cross-attention 的形式

**状态**：纯想法，未写代码

---

## 版本命名说明

项目使用了漫威风格的命名：
- **Iron Cell** / **IronMan**：项目整体
- **Javis**（JARVIS 的变体）：核心 cross-attention 模块
- **Mark-1, Mark-33, Mark-42**：版本迭代代号
- **SoulBone**：项目的另一个代号（灵魂骨骼，暗喻模型的骨架扩张）
