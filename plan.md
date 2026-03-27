# Plan：梳理 eval_ppl.py 与 eval_test.sh（chunking / adaptive / 启动正确性）

## 目标

1. 梳理 [eval_ppl.py](file:///default-vepfs/public/user/ga/Iron/IronMan/scripts/eval_ppl.py#L1-L141) 的行为是否自洽、有哪些隐含前提与风险点。
2. 核对 [eval_test.sh](file:///default-vepfs/public/user/ga/Iron/IronMan/scripts/eval_test.sh#L1-L64) 启动参数是否与实际 entrypoint 匹配、默认配置是否容易踩坑。
3. 明确“是不是得用 adaptive”在两套评测入口中的适用范围，并给出推荐用法与后续修改项（如需要）。
4. 针对“整份 jsonl 求平均”的目标，给出最简洁的使用路径与必要的数据格式约束。

## 已确认的现状（基于源码阅读）

### eval_ppl.py 的定位与前提

- 该脚本核心是在 `no_grad + autocast(bf16)` 下用 `TrainStepModule(model, phase)` 对 DataLoader 输出的 batch 求平均 loss，然后 `ppl = exp(loss)`。
- 只要 `--ckpt_dir` 是一个完整的 `save_pretrained` checkpoint 目录（包含模型与 tokenizer），`load_tokenizer/load_model` 会从 `resume_path=ckpt_dir` 加载。
- `--chunking adaptive` 分支要求数据 jsonl 每行必须提供 `chunk_lens` 或 `raw_chunk_lens`，脚本通过 `_JsonlDatasetWithChunkLens` 强制校验。
- 脚本不会主动初始化分布式；只有“外部已 init_process_group”的情况下才会 all_reduce 聚合 loss。

### eval_with_buffer.py（eval_test.sh 实际调用）的定位与前提

- `eval_test.sh` 实际调用的是 `python -m eval.eval_with_buffer`，该入口内部是固定 chunk_size 的“固定分块”评测；不会根据数据自适应 chunk。
- `eval_data_path` 在该入口里只读取第一行 JSON，并只取 `text` 字段，因此它并不是“遍历 jsonl 计算整体 ppl”的入口。
- `MODEL_NAME` 在脚本里默认空字符串：在 `mark42 + resume_path` 场景通常不影响，但切换到 `oracle/amnesiac` 时需要显式提供有效模型路径/名字。

### modeling_iron_cell.py 是否“针对不同 chunk 排布做适配”

- 模型主干 `forward` 本身并不写死 `chunk_size=16`；它只消费 collator 构造好的 `inputs_embeds + attention_mask (+ position_ids)`，因此是否 fixed/adaptive 主要由数据处理决定。
- `compute_compressed_vectors` 只要求 `chunk_input_ids` 是 `[B, C, L]` 的 3D 张量；其中 `L` 可以是 fixed 的 `chunk_size`（默认 16），也可以是 adaptive collator pad 后的 `max_cmp_chunk_len`，真正的有效长度由 `chunk_attention_mask` 表达。
- 代码里只有一个可选 helper `build_student_attn_mask(..., chunk_size)` 使用 `raw_pos // chunk_size` 这种等长块假设；它不适用于 adaptive 的可变 `raw_chunk_lens`，并且在当前训练/评测主路径里并未被使用（主路径直接用 collator 给的全量 mask）。

## 关键问题与判断点（回答“是否合理/启动是否有问题/是否需要 adaptive”）

1. **eval_ppl.py 是否合理**
   - 逻辑整体自洽，且对 “adaptive 数据必须带 chunk_lens” 这一点做了明确校验。
   - 主要风险点是：它把 `_Args.model_name` 固定为空字符串，因此它本质是“只能评测 resume ckpt”；如果要评测未 resume 的 base 模型，需要扩展参数。
   - 分布式方面：如果你用 `torchrun` 直接起它但脚本内部不 init process group，那么它仍按单进程逻辑跑（且可能多进程各自打印/算一遍）。

2. **eval_test.sh 启动是否有问题**
   - 纯“参数名是否匹配”层面：它传的 flags 与 `eval_with_buffer.py` 的参数是匹配的。
   - 但默认配置存在两类高频踩坑：
     - `EVAL_DATA_PATH` 指向一个 jsonl 文件名，但入口只读第一行；如果你以为它会遍历整个文件，会得到错误预期。
     - `MODEL_NAME` 默认空；切换 `oracle/amnesiac` 会直接失败或加载异常。

3. **是否必须用 adaptive**
   - 对 `eval_ppl.py`：当你的评测数据本身带 per-sample 的 chunk_lens（或你想按自适应切分的 chunk 边界来喂入），才需要 `--chunking adaptive`；否则用 fixed + `--chunk_size`。
   - 对 `eval_with_buffer.py`：当前实现就是 fixed chunk_size，不存在“开 adaptive”这一分支；所谓 adaptive 不适用于这个入口。

4. **“整份 jsonl 求平均”的最简路径（你当前想要的）**
   - 推荐入口：`scripts/eval_ppl.py`，它会遍历 DataLoader（整份 jsonl）并汇总 loss/ppl。
   - fixed vs adaptive 的差别本质在“数据组织/几何构造”：
     - fixed：按 `chunk_size=16` 等长切分，collator 产出对应的 `attention_mask_2d/position_ids/labels/chunk_*`。
     - adaptive：按样本内的 `chunk_lens/raw_chunk_lens` 切分，collator 产出另一套 `attention_mask_2d/position_ids/labels/chunk_*`（但 batch schema 与 step module 一致）。
   - 模型内部与 `TrainStepModuleForFullLayersKVInjection` 不需要为 adaptive 做额外改动；它们只依赖 batch 里这些字段（mask/posid/labels/chunk_input_ids/chunk_attention_mask/memory_positions/prefix_lens/valid_lens），而这些字段由 collator 负责构造。

## 执行计划（若你希望我直接把脚本改得更不容易踩坑）

1. 明确你要的评测形态，并统一入口：
   - “整份 jsonl 求平均 ppl”：走 `scripts/eval_ppl.py`。
   - “单条长文本 + buffer/mark42 行为分析”：走 `eval/eval_with_buffer.py`。
2. 如果继续用 `eval_test.sh`：
   - 将默认 `EVAL_DATA_PATH` 改成单条 JSON（或在脚本里显式说明它只读第一行）。
   - 对 `oracle/amnesiac` 强制要求 `MODEL_NAME` 非空（脚本里加检查）。
3. 如果希望 `eval_with_buffer.py` 也支持“遍历 jsonl + 汇总指标”：
   - 扩展其数据读取逻辑，支持逐行读取并累积输出（并保持原有“只读第一行”的兼容选项）。
4. 如果希望 `eval_ppl.py` 支持分布式评测：
   - 增加 `--parallel`（none/ddp）与必要的 `init_process_group` 逻辑，使 `all_reduce` 生效且 rank0 输出不重复。
5. 最终用最小化的可复现实例做验证：
   - 对 fixed/adaptive 两条分支各跑一次（含数据格式校验）。
   - 对 `oracle/amnesiac/mark42` 三种 eval_mode 各跑一次（至少验证参数与加载路径正确）。
