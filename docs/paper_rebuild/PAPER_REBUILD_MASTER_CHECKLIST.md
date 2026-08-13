# Paper Rebuild Master Checklist

> **For agentic workers:** Execute only one stage at a time. Before implementing a stage, create its focused plan and use the appropriate execution workflow; do not treat this roadmap as permission to skip tests, evidence gates, or user checkpoints.

**Goal:** 将现有项目推进到证据可信、语言完整、通过投稿前审查并可正式提交的论文；录用由外部编辑和审稿人决定，不能预先保证。

**Architecture:** 完整规则保存在 [FROZEN_EXPERIMENT_PROTOCOL.md](./FROZEN_EXPERIMENT_PROTOCOL.md)。本文件只负责顺序、交付物和通过条件；每完成一个阶段再进入下一个阶段，避免同时修改代码、实验和论文叙事。

**Tech Stack:** Python、PyTorch、Transformers、Hugging Face Datasets、RoBERTa-base、RTX 5080、Colab A100、Markdown/DOCX。

## Global Constraints

- Canonical 数据：固定 100k MNLI train、5k validation，`data_seed=42`。
- Training seeds：`[42, 123, 2024, 3407, 777]`。
- 主要指标：HANS non-entailment accuracy。
- Utility constraint：MNLI 相对 Standard LoRA 不低于 0.5 percentage points。
- Official HANS evaluation 不用于调参或中间 checkpoint 选择。
- 历史 `ties_results/` 仅作 exploratory evidence，不覆盖、不冒充 canonical results。
- 每个阶段结束后必须由用户确认，才能进入下一阶段。

---

## 当前状态

```text
[已完成] 阶段 0：审计与协议冻结
[已完成，待用户验收] 阶段 1：最小代码改造
[等待用户指令] 阶段 2：Smoke tests 与环境冻结
[等待] 阶段 3–10
```

## 使用方法

1. 每次只处理一个阶段。
2. 将完成项从 `[ ]` 改成 `[x]`。
3. 一个阶段的“完成门槛”全部通过后，再让我更新清单并进入下一阶段。
4. 遇到失败结果时执行转向规则，不为了保留旧标题而继续堆实验。

## 一页总览

- [x] 0. 确认冻结协议和本清单。
- [x] 1. 完成最小代码改造与测试。
- [ ] 2. 通过 RTX 5080/A100 smoke tests 并冻结环境。
- [ ] 3. 完成 30/30 核心 canonical results。
- [ ] 4. 完成统计分析和 Gate A–D，冻结论文主线。
- [ ] 5. 只补必要的 baselines、rank controls 或第二路线。
- [ ] 6. 根据 canonical evidence 全文重写论文。
- [ ] 7. 通过引用、数据、统计与复现完整性检查。
- [ ] 8. 完成模拟审稿、修订、复审和 final integrity。
- [ ] 9. 选择 venue，完成并正式提交投稿包。
- [ ] 10. 完成真实返修、response 和 camera-ready。

```text
协议 -> 代码 -> Smoke -> Canonical -> 决策 -> 必要扩展
     -> 重写 -> 完整性 -> 模拟审稿 -> 投稿 -> 返修/接收
```

---

## 阶段 0：审计与协议冻结

**交付物：** 冻结实验协议和本总控清单。

- [x] 阅读并审计现有代码、实验驱动、测试和历史结果。
- [x] 将历史结果标记为 exploratory。
- [x] 冻结 100k MNLI、5 seeds、主要指标和 utility constraint。
- [x] 冻结 6-condition 核心归因矩阵和 Gate A–D。
- [x] 保存完整协议到 `docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md`。
- [x] 用户确认本简明清单，并明确说“开始阶段 1”。

**完成门槛：** 用户确认后才修改代码。

**给 Codex 的指令：**

```text
开始清单阶段 1：为冻结实验协议制定并执行最小代码改造计划。
```

---

## 阶段 1：最小代码改造

**目标：** 只修复 canonical 实验必须具备的能力，不重写主架构。

- [x] 将 `data_seed` 与 `training_seed` 分离。
- [x] 创建确定性的 HANS-train 80% build / 20% dev split，并验证与 evaluation 无交集。
- [x] 禁止 Phase 1/2 和 Phase 3 中间阶段读取 official HANS evaluation。
- [x] 增加 `staged_neither` 和 `class_prior_reweight` 两个条件。
- [x] 保存逐样本 HANS predictions、pair ID、heuristic 和 subcase。
- [x] 统一 config、environment、Git commit、status 和 metrics schema。
- [x] 标准 JSON 禁止 `NaN/Infinity`。
- [x] 创建支持共享 Phase-1/2 checkpoint 的 `run_canonical.py`。
- [x] 为上述每项补测试，并先验证测试能捕获旧行为，再验证修复通过。
- [x] 提交一个独立、可回滚的 canonical infrastructure commit。

**完成门槛：** 全部新增测试通过；旧实验结果未被覆盖；Git working tree clean。

**不要做：** 不运行 30 个正式实验，不改论文结论，不新增第二模型。

---

## 阶段 2：Smoke tests 与环境冻结

**目标：** 在花费 A100 预算前证明流水线可运行、可恢复、可复算。

- [ ] 在 RTX 5080 上运行全部单元测试。
- [ ] 在 RTX 5080 上完成 tiny-data 端到端 smoke run。
- [ ] 验证 official HANS evaluation 未出现在中间日志中。
- [ ] 验证两条分支加载相同 Phase-2 checkpoint hash。
- [ ] 验证 aggregate HANS metrics 可由 predictions 完全复算。
- [ ] 在 A100 上完成一个小规模 smoke run。
- [ ] 在同一 A100 环境重复一个 smoke condition，主要 accuracy 差异不超过 0.5 pp。
- [ ] 冻结 Python、PyTorch、Transformers、Datasets、CUDA 和 GPU 版本。
- [ ] 创建 `canonical_v1` 的 protocol、data 和 environment manifests。

**完成门槛：** 所有检查通过；`canonical_v1` 正式结果目录为空；canonical commit 已记录。

**给 Codex 的指令：**

```text
开始清单阶段 2：执行本地和 Colab 的 A100 smoke tests，并冻结 canonical 环境。
```

---

## 阶段 3：运行核心 Canonical 实验

**目标：** 生成 6 conditions × 5 seeds = 30 个完整 method-seed result cells。

- [ ] 为每个 seed 生成一份共享 Phase-1/2 checkpoint。
- [ ] 运行 `standard_lora`。
- [ ] 运行 `full_sr`。
- [ ] 运行 `subtraction_only`。
- [ ] 运行 `reweight_only`。
- [ ] 运行 `staged_neither`。
- [ ] 运行 `class_prior_reweight`。
- [ ] 每个 run 保存 config、status、environment、metrics、predictions 和日志。
- [ ] 失败 run 只按原 seed/原配置重跑；不得临时调参。
- [ ] 完成 30/30 后进行 checksum、样本 ID 和 checkpoint 完整性检查。

**完成门槛：** 30/30 成功；五个预定 seeds 齐全；无配置漂移；HANS 指标全部可复算。

**红线：** 不根据 official HANS evaluation 选择更有利的 rank、beta、gamma、layer top-k 或 seed。

---

## 阶段 4：统计分析与论文主线决策

**目标：** 先让结果决定论文，再开始重写。

- [ ] 报告每个方法的五种子 mean、sample SD、min 和 max。
- [ ] 计算 seed-wise paired differences 和 95% CI。
- [ ] 对 HANS predictions 执行分层 paired bootstrap。
- [ ] 分别报告 HANS entailment/non-entailment、heuristic 和 subcase。
- [ ] 执行 Gate A：是否存在满足 MNLI utility 的 mitigation signal。
- [ ] 执行 Gate B：subtraction 是否优于 `reweight_only`。
- [ ] 执行 Gate C：N-guided weighting 是否优于 staged 和 class-prior controls。
- [ ] 记录 Gate D 状态：当前是否有资格保留 `Rank-Differential`。
- [ ] 将决定写入机器可读 JSON 和 Markdown 决策报告。

**决策规则：**

```text
Gate A 不通过 -> 停止扩展，重新设计方法。
Gate A 通过、Gate B 不通过 -> 主线转为 N-guided shortcut-aware reweighting。
Gate C 不通过 -> 降级为 label-aware mitigation，重新评估论文新颖性。
Gate B 通过 -> subtraction 可保留为核心组件。
Gate D 未通过 -> 从标题和摘要删除 Rank-Differential 的因果暗示。
```

**完成门槛：** 论文暂定标题、核心贡献、主表结构和允许使用的结论全部由 Gate 结果确定。

**给 Codex 的指令：**

```text
开始清单阶段 4：分析 canonical results，执行 Gate A–D，并冻结论文主线。
```

---

## 阶段 5：只做必要的条件性扩展

**目标：** 补足审稿人真正会要求的证据，不重复无效网格搜索。

- [ ] Gate A 通过后，先用 3 seeds 运行 JTT、PoE 和 z-filter。
- [ ] 只有进入主表的 baseline 才补足 5 seeds。
- [ ] 只有 Gate B 通过后才运行 scale-controlled rank controls。
- [ ] 只有核心主线稳定后才选择第二模型或新的 held-out benchmark。
- [ ] 对任何新扩展先写 protocol addendum，再运行实验。
- [ ] 更新统计汇总和证据边界。

**完成门槛：** 主方法、强 baseline、核心消融和泛化证据足以支撑目标投稿层级；没有未解释的关键反例。

**不要做：** 不重跑全部旧 sensitivity grid；不对已塌缩的 NegMerge/naive subtraction 做 5-seed 全预算。

---

## 阶段 6：按新证据重写论文

**目标：** 重建论文论证，不在旧稿上局部润色。

- [ ] 根据 Gate 结果冻结最终标题和一句话贡献。
- [ ] 重写 Abstract：问题、方法、主要结果、限制各一层，不写 unlearning 过强结论。
- [ ] 重写 Introduction：研究缺口、RQ、贡献和证据边界。
- [ ] 重写 Related Work：NLI shortcuts、debiasing、LoRA/task arithmetic、unlearning/mitigation。
- [ ] 重写 Method：P/N adapters、Phase 1–3、layer selection、subtraction/reweighting。
- [ ] 重写 Experiments：数据 split、baselines、seeds、指标、统计和复现信息。
- [ ] 重写 Results：先主结果，再因果消融、泛化和诊断。
- [ ] 重写 Discussion/Limitations：标签先验、HANS test exposure、单模型边界和机制限制。
- [ ] 重写 Conclusion，使结论不超过 Gate A–D。
- [ ] 所有表格、图和正文数字都链接到 canonical artifacts。

**完成门槛：** 获得结构完整、数字一致、无旧结论残留的 `paper_draft_v1`。

---

## 阶段 7：引用、数据与统计完整性检查

**目标：** 在模拟审稿前消除可验证的拒稿风险。

- [ ] 逐条验证参考文献作者、标题、年份、venue 和 DOI。
- [ ] 检查每个外部事实是否有直接支持的引用。
- [ ] 检查所有方法/结果数字是否能追溯到 canonical JSON/predictions。
- [ ] 核对正文、表格、图注和补充材料中的数字与精度。
- [ ] 审查推断单位、paired comparisons、CI 和 multiple-comparison policy。
- [ ] 检查数据泄漏、test exposure、排除规则和失败 run 报告。
- [ ] 检查代码、数据、模型和限制声明是否可复现。
- [ ] 完成 pre-review integrity report；P0 问题必须为 0。

**完成门槛：** 引用与数据完整性 PASS；不存在无法追溯的核心主张。

---

## 阶段 8：模拟审稿、修订与复审

**目标：** 在真实投稿前经历一次完整 Major Revision 循环。

- [ ] 进行方法、统计、NLI 领域、novelty 和 devil's-advocate 模拟审稿。
- [ ] 汇总 P0/P1/P2 问题和模拟编辑决定。
- [ ] 为每个问题建立 revision roadmap。
- [ ] 完成第一轮修订并保留修改记录。
- [ ] 对修订稿进行独立 re-review。
- [ ] 若仍有 Major issues，完成最后一轮针对性修订。
- [ ] 从头执行 final integrity check，而不是只复查旧问题。

**完成门槛：** P0=0；关键 P1 已解决或明确写入 limitations；final integrity PASS。

---

## 阶段 9：选择 venue 并完成投稿包

**目标：** 形成与目标 venue 完全匹配的正式投稿材料。

- [ ] 根据最终证据强度选择主会议、Findings 或其他合适 venue。
- [ ] 核对最新 deadline、页数、匿名、引用和补充材料规则。
- [ ] 按模板完成匿名稿、附录和 reproducibility checklist。
- [ ] 准备 title page、author contributions、data/code availability 和 AI disclosure。
- [ ] 准备 cover letter、highlights 和推荐审稿人材料（如 venue 要求）。
- [ ] 对最终 PDF/DOCX/LaTeX 做视觉和内容检查。
- [ ] 保存 submission-ready 版本、hash 和材料清单。
- [ ] 用户最终确认后提交。

**完成门槛：** 投稿系统确认成功，保存 manuscript ID 和提交版本。

---

## 阶段 10：投稿后修订直到最终决定

**目标：** 对编辑和审稿意见进行可追踪、证据化响应。

- [ ] 保存 decision letter 和每位 reviewer 的独立意见。
- [ ] 将每条意见分类为文字修改、分析补充、实验补充或合理反驳。
- [ ] 先制定 response/revision plan，再修改论文。
- [ ] 新实验必须记录新协议版本，不覆盖 canonical v1。
- [ ] 写逐点 response letter，并给出稿件位置和证据。
- [ ] 完成修订稿、redline、补充材料和完整性复检。
- [ ] 返修后再次核对引用、数字、代码和图表。
- [ ] 接收后完成 camera-ready、版权和公开材料。

**最终可控成功标准：** 论文证据、写作、引用、统计、复现和投稿材料均达到审稿就绪；外部录用决定及其时间不在作者或 Codex 的控制范围内。

---

## 现在只做一件事

- [ ] 阅读 `STAGE1_CANONICAL_INFRASTRUCTURE_REPORT.md`；若验收阶段 1，回复：

```text
开始清单阶段 2。
```

阶段 2 完成前，不启动 30 个正式 A100 canonical runs，也不全面润色旧稿。
