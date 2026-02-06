# Iron-Cell Phase-1 训练指引（MVP）

本文描述 Phase-1（MVP）训练的推荐流程，目标是验证 **Zipper Layout + Staircase Mask** 在工程上可跑通，并确保显存/精度约束满足：

- 8x A800 (80GB)
- BF16 混合精度（强制）
- Gradient Checkpointing（必须）

## 训练目标

Phase-1 以“压缩向量 V 作为外显记忆槽位”作为核心，训练的 loss 仅计算在 `Raw_Tokens` 区域：

- `<soc>`、`V`、`<eoc>` 区域 label 为 `-100`
- `Raw_Tokens` 区域按标准 Causal LM 计算 loss

## 数据构造（Zipper Layout）

每条样本会被 tokenizer 切分为若干个 raw chunk（按 token 数量 `chunk_size` 划分），设 chunk 数为 `C`：

```
[<soc>]  V1  [<eoc>]  V2  [<eoc>]  ...  VC  [<eoc>]  Raw_Tokens
```

其中 `Vi` 不对应真实 token id，而是在 embedding 序列中占据一个“向量槽位”。工程上：

1. 先用占位 token id 生成 base embedding
2. 再用 compressor+projector 计算得到 `Vi`，覆盖写入到对应 position 的 `inputs_embeds`

## Mask 构造（Staircase Mask）

对 `Raw_Tokens` 区域，每个 token 的可见历史由两部分组成：

1. 标准 causal：只能看见过去的 raw tokens
2. 阶梯式压缩记忆：第 N 段 raw（第 N 个 chunk）只能看见 `V1..VN` 与对应 `EOC1..EOCN`

这等价于对 “过去的 prefix token” 进行选择性屏蔽（不是未来屏蔽）。

## 冻结策略（Freeze Strategy）

推荐遵循以下最小可行策略：

- **冻结 compressor**：forward 强制 `torch.no_grad()`，参数 `requires_grad=False`
- **冻结 generator backbone**：Transformer blocks 全冻结
- **解冻 projector**：全量训练
- **解冻 generator.embed_tokens**：MVP 阶段可直接解冻全量 embedding

对应实现入口：

- [IronCellModel.freeze_for_phase_1](file:///d:/code/IronMan/iron-cell/src/modeling_iron_cell.py)

## BF16 + Gradient Checkpointing

在脚本中务必设置：

- `torch.autocast(dtype=torch.bfloat16)`
- `model.generator.gradient_checkpointing_enable()`
- `model.generator.config.use_cache = False`

示例脚本：

- [train_phase_1.py](file:///d:/code/IronMan/iron-cell/examples/train_phase_1.py)

## 重要安全约束

- `tokenizer.add_special_tokens(...)` 之后必须执行 `resize_token_embeddings(len(tokenizer))`
- smart init 只从已有 embedding 拷贝/均值构造，不应引入外部权重文件或硬编码密钥

