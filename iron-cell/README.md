# Project Iron-Cell (SoulBone)

Iron-Cell 是一个自回归压缩记忆系统（Autoregressive Compressed Memory System）。核心训练范式采用 Masked Parallel Training：通过构造“拉链式输入布局（Zipper Layout）”与“魔法阶梯 Mask（Staircase Mask）”，在一个 batch 内并行训练多个阶段的重建任务，而不使用 RNN 式循环训练。

## 核心概念

- **Compressor（装甲）**：冻结的 Llama-3（或同结构的因果 LM），用于从文本 chunk 中提取语义 hidden states。
- **Projector（贾维斯）**：线性层，将 compressor hidden 映射为压缩向量 **V**，支持 Identity+Noise 初始化。
- **Generator（托尼）**：Llama-3（因果 LM），接收拼接后的 `inputs_embeds` 与自定义 `attention_mask` 进行训练/生成。

## Zipper Layout（拉链式布局）

单条样本在 embedding 序列上的形态如下：

```
[<soc>]  V1  [<eoc>]  V2  [<eoc>]  ...  Raw_Tokens
```

其中 `V1..Vk` 是压缩向量槽位（1 个向量占 1 个位置），由 compressor+projector 在 forward 中实时计算并注入到 `inputs_embeds` 中。

## Staircase Mask（魔法阶梯 Mask）

为 `Raw_Tokens` 区域的每个 token 构造自定义 2D attention mask，使其只能访问：

- 该 token 之前的 raw history（标准 causal）
- 以及 **至多第 N 个** 压缩向量/锚点（第 N 段 raw 只能看 V1..VN 与对应 EOC）

## 目录结构

```
iron-cell/
├── README.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── configuration_iron_cell.py
│   ├── modeling_iron_cell.py
│   ├── data_processor.py
│   └── token_utils.py
├── examples/
│   ├── train_phase_1.py
│   └── infer_demo.py
├── scripts/
│   └── run_phase_1.sh
└── docs/
    └── training_guide.md
```

## 快速开始（本地代码组织）

本仓库采用轻量化的 `src/` 目录组织方式。运行示例时，建议从 `iron-cell/` 目录执行，并确保 Python 能找到 `src/`：

```bash
cd iron-cell
python examples/train_phase_1.py
```


