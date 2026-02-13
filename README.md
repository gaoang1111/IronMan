# IronCell — Mark 1

IronCell 是一个 **16:1 长文本压缩记忆** 的原型系统：在 **KV/显存占用约变为原来的 6.25%** 的情况下，相比原生模型 PPL **仅从 7.4 上升到 11.2**。

这是一个用于探索当前模型架构的 prototype：

- **模型间协同合作**：通过高维 hidden state / 压缩向量 V 的通信与注入。
- **对已训练模型做能力分化与扩张**：在尽量不破坏 base 能力的前提下，为其“扩张出”解码压缩记忆的专用能力。

Mark 1 是证明可行性的原型。欢迎开源社区在此基础上构建更模块化、更自动化的 Mark 42。

## 细胞分化理论（动机）

我把 LLM 视作一个预训练好的“状态机”，其可扩展性极差：

- 人类面对新任务会通过 **干细胞分化** 获得新结构，并作为整体协作。
- 大模型若长期保持“不可扩张”的单体结构，很难支撑 AGI 时代的持续能力增长。

IronCell Mark 1 的尝试是：选择同源 base（这里使用 **Llama 3.1 8B** 作为“干细胞”），诱导其分化为：

- **Compressor（cmp）**：擅长把 raw chunk 压缩成语义向量。
- **Generator（gen）**：擅长利用压缩向量进行重建与生成。

并在训练中对 loss 可控、可稳定下降。

## 核心组件

- **Compressor（cmp）**：从文本 chunk 抽取语义表示（Phase1 通常冻结，Phase2 可训练）
- **Projector（proj）**：线性映射，将 compressor hidden 映射到 generator hidden size，产出压缩向量 V
- **Generator（gen）**：因果 LM，接收拼接后的 `inputs_embeds` 与自定义 `attention_mask` 训练/生成

## Zipper Layout（训练 vs 推理）

### 训练用 Layout（Masked Parallel Training 适配）

训练侧序列（token-id 视角）是“控制链 + raw chunks”，类似：

```
[<bos>][<soc>]  V-1  [<eoc>]  V0  [<eoc>]  V1  [<eoc>] ... Raw_Token chunks
```

- `V_k` 的下标 k 表示第 k 个 raw chunk 的压缩结果。
- `V-1` 可以理解为“初始组/占位组”的记忆向量槽，用于让 mask/labels 的几何结构在不同样本上统一。
- 这种结构是为了做 Masked Parallel Training：配合 Staircase(Zipper) Mask，让不同 raw 段只允许访问对应范围的控制链信息，避免信息泄漏并实现并行训练。

Zipper/Staircase mask 与 position_ids/labels 的构造实现位于：

- [build_zipper_mask_posid](file:///default-vepfs/public/user/ga/Iron/IronMan/src/data_processor.py#L41-L151)
- [build_zipper_labels](file:///default-vepfs/public/user/ga/Iron/IronMan/src/data_processor.py#L177-L242)
- 由 [IronCellCollator](file:///default-vepfs/public/user/ga/Iron/IronMan/src/data_processor.py#L304-L419) 汇总输出 `ZipperBatch`

### 推理用 Layout（计划 / TODO）

推理侧更接近：

```
[<bos>][<soc>]  V-1  V0  V1 ... [<eoc>] Raw_Token chunk
```

说明：

- 推理 forward 需要对“V 的生成/滚动更新、`[<eoc>]` 的安排、attention mask 的推理版构造”做专门适配。
- 当前仓库主要实现训练侧（collator+mask+训练循环），推理版 forward 适配尚未完善（TODO）。

## Special Tokens

IronCell 使用 3 个 special tokens：`<soc>`, `<eoc>`, `<v_none>`。

- 可开启只训练新增 special token embedding 小表：base embedding 冻结，只更新 special 小表（训练脚本中 `TRAIN_ONLY_SPECIAL=true`）。

## 数据

训练/评估数据为 JSONL，每行至少包含：

```json
{"text": "..."}
```

数据来源：

- 从 HuggingFace 上的 FineWeb-Edu 下载。
- Phase2 使用 10000 条文本，单条字符串长度约 10k–30k。
- 在 150 steps 的窗口内没有重复语料（单轮推进、zero-overlap）。

## 训练与结果（Mark 1）

### Phase1（对齐阶段：只训 proj + special）

目标：只训练 projector + 新增 special token，快速完成“压缩信号对齐”。

- 约 20 steps：训练 loss 从 12.8 → 4.12
- 训练过程中 grad norm 健康

运行：

```bash
TRAIN_ONLY_SPECIAL=true PARALLEL=ddp bash scripts/run_train.sh
```

### Phase2（功能分化：解冻 cmp+gen 并加入约束）

目标：在 Phase1 基础上解冻模型权重，让 cmp/gen 分化出稳定解码压缩记忆的能力。

- 使用 L2 正则约束 cmp→proj 输出向量的范围（限制 V 的幅度）：见 [TrainStepModule](file:///default-vepfs/public/user/ga/Iron/IronMan/src/train.py#L97-L139)
- 验证集 loss（每 30 steps 记录一次）大致：
  - Step 30: 2.72
  - Step 60: 2.49
  - Step 90: 2.44
  - Step 120: 2.43
  - Step 150: 2.41

运行：

```bash
bash scripts/run_phase2.sh
```

## 评估（PPL）

脚本：`scripts/eval_ppl.py`

```bash
python scripts/eval_ppl.py \
  --ckpt_dir checkpoints/phase2/phase2_step_XXX \
  --data_path ../data/phase1_eval.jsonl \
  --phase phase2 \
  --max_batches 50
```

输出：`eval_loss` 与 `ppl=exp(loss)`。

## 工程约定（简要）

- **Weight Decay 分组**：包含 `bias/layer_norm/layernorm/ln_` 的参数强制 `weight_decay=0`，其它 trainable 参数使用传入值。
- **FSDP dtype**：FSDP flatten 要求同一单元参数 dtype 一致；工程中已确保 projector/special embedding 与 generator dtype 对齐（bf16）。

## Mark 42（邀请）

Mark 1 只是原型，欢迎社区推进 Mark 42：

- 完整推理版 forward（V 的更新与推理 mask）
- 更强的可扩展协作协议（多模型、多路由、容错与冗余）
- 更标准的复现实验与基准
