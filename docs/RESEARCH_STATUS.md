# IronCell 当前研究状态（快照）

> **最后更新**: 2025-03
> **当前分支**: `ga/adaptive-seg`
> **最新 commit**: `47c22e6` — `chore: save all for reset`

---

## 最新 Checkpoint

| 项目 | 值 |
|------|-----|
| Checkpoint | `phase-full_step_480` |
| Phase | full（所有参数解冻） |
| Fixed Eval Loss | ~2.3 |
| Adaptive Eval Loss | ~2.6（分布偏移，未 fine-tune） |
| Chunk Size | 16 tokens (fixed) / variable (adaptive) |
| Queries per Chunk | 2 |
| Compression Ratio | 8:1 (16 tokens → 2 vectors) |
| Deep KV Injection Layers | ~15, ~23, 31 |

---

## 当前工作重点

**Adaptive Chunking**：按语义边界切分 chunk，利用 Tail Bias 而非对抗它。
- 离线切分逻辑已实现（天然分隔符 + surprisal 辅助）
- Pipeline 验证通过（fixed16_as_adaptive 测试无 bug）
- 待做：在 adaptive 数据上 fine-tuning

---

## 阻塞点

1. **Loss 卡在 ~2.3**（fixed chunking）：不确定是信息瓶颈还是需要更多训练步数/数据
2. ~~**压缩信息有效性不确定**~~：**已部分验证** — Buffer size 实验证明压缩信息有效（mark42-buf0 loss 2.27 vs amnesiac 2.82，PPL 9.66 vs 16.76）。仍需 random vector ablation 做进一步确认。
3. **Deep KV "冲击"问题**：Needle 测试中不用 inject 反而更好，直接残差注入可能不是最优方式

---

## 下一步行动

- [ ] Adaptive chunking fine-tuning（从 phase-full_step_480 继续）
- [ ] 压缩信息有效性 ablation（random V / zero V / shuffled V）
- [ ] Sparse Attention 解耦方案设计与实现
- [ ] 整理 wandb 实验记录，补充 EXPERIMENT_LOG.md 中的 [TODO] 项
- [ ] 确定 ROADMAP.md 中各假设的验证优先级

---

## 文档体系索引

| 文档 | 内容 | 更新频率 |
|------|------|---------|
| [RESEARCH_OVERVIEW.md](./RESEARCH_OVERVIEW.md) | 研究动机、核心方法、与 related work 的关系 | 低（基本稳定） |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 技术架构细节：模型、数据流、注入机制 | 架构变更时 |
| [EVOLUTION.md](./EVOLUTION.md) | Mark-1 → Mark-42 → Adaptive 的完整迭代记录 | 每次重大迭代 |
| [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) | 所有实验的假设/配置/结果/结论 | 每次实验后 |
| [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) | 已知问题、技术决策及其原因 | 发现新问题时 |
| [ROADMAP.md](./ROADMAP.md) | 未来方向、待验证假设、优先级 | 方向调整时 |
| **本文档** | 当前状态快照 | 每次重要进展后 |
| [training_guide.md](./training_guide.md) | Phase-1 训练操作指南 | 低 |

---

## 关键数据文件

| 文件 | 用途 |
|------|------|
| `data/phase1_train.jsonl` | Phase 1 训练数据 |
| `data/phase2_train.jsonl` | Phase 2 训练数据 |
| `data/eval_fixed.jsonl` | Fixed chunking 评测数据 |
| `data/eval_adaptive.jsonl` | Adaptive chunking 评测数据 |
| `_data/eval_results/*.json` | 各种 eval 结果（mark42-buffer*, amnesiac*, oracle*） |
| `_data/needle_scale*/` | Needle-in-haystack 实验数据 |
| `wandb/` | 15 个 wandb run 记录 |
