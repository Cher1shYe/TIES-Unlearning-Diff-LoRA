# Frozen Experiment Protocol v1.0

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-07
- Verification Status: UNVERIFIED
- Version Label: frozen_experiment_protocol_v1.0
- Upstream Evidence: repository audit at Git commit `360df20`
- Working Directory: `E:/Learning/LoRA Project/TIES-Unlearning-Diff-LoRA`
- Intended Study Type: stochastic, environment-sensitive machine-learning training experiment

## 0. 文档状态与冻结规则

本文件是论文重构阶段的正式实验协议，适用于 `canonical_v1` 实验。它冻结的是实验设计和决策规则，而不是预先保证某一种方法会胜出。

在本协议下：

1. 现有 `ties_results/` 中的历史结果一律视为 **exploratory evidence**，不得直接替代 canonical results。
2. Canonical 实验开始前，必须完成第 4 节列出的最小代码修正和第 14 节的预运行检查。
3. Canonical 实验开始后，不得根据 HANS evaluation、ANLI、SNLI-hard、WANLI 或其他最终测试结果修改超参数、数据划分、主要指标或成功标准。
4. 任何会影响模型、数据、训练、评价或统计结论的修改，都必须：
   - 停止当前 canonical 批次；
   - 将协议版本升级为 `v1.1` 或 `v2.0`；
   - 在变更日志中记录原因；
   - 使用新的结果目录，禁止覆盖 `canonical_v1`。
5. 仅修正日志文字、表格排版或不影响数值的结果展示错误时，可以保留版本号，但必须记录修正内容。
6. 本协议未授权自动运行实验；实验执行需在代码修正、测试和用户复核后另行开始。

## 1. 研究定位

### 1.1 暂定研究问题

在保持 MNLI 任务效用的条件下，低秩 shortcut-oriented adapter 所提供的训练信号，能否缓解 RoBERTa-base 对 HANS 句法捷径的依赖？观察到的改善究竟来自：

1. selective adapter subtraction；
2. N-guided Phase-3 reweighting；
3. 两者的交互；
4. 或者仅仅来自标签先验重平衡和分阶段训练？

### 1.2 论文主张边界

Canonical results 出现之前，论文只允许使用以下中性表述：

> We evaluate whether an asymmetric dual-adapter pipeline can mitigate shortcut reliance while preserving task utility, and isolate the respective contributions of selective subtraction and shortcut-aware recovery.

禁止预先使用以下结论：

- “shortcut 已被 unlearned/removed/erased”；
- “低秩适配器天然捕获 shortcut”；
- “rank differential 导致 shortcut specialization”；
- “TIES subtraction 是性能提升的原因”；
- “候选层分析证明了内部机制”。

### 1.3 术语冻结

- `shortcut mitigation`：论文默认目标术语。
- `P-adapter`：task/utility-oriented adapter；不称为“纯语义适配器”。
- `N-adapter`：shortcut-oriented 或 bias-oriented adapter；在标签先验控制通过前，不称为“shortcut adapter”。
- `selective rank-differential subtraction`：仅描述运算形式，不自动包含机制因果含义。
- `shortcut-aware recovery`：N-guided Phase-3 example reweighting。
- `candidate-layer prediction stability`：替代现有代码中的 prediction-depth 机制性表述。

### 1.4 预设假设

- **H1（mitigation）**：`reweight_only` 的 HANS non-entailment accuracy 高于 `staged_neither` 和 `standard_lora`，且满足 MNLI utility constraint。
- **H2（subtraction contribution）**：`full_sr` 相对 `reweight_only` 提供额外 HANS non-entailment 增益。现有探索性结果不支持预设其必然成立，因此该假设按双向不确定结果处理。
- **H3（beyond class prior）**：`reweight_only` 高于 `class_prior_reweight`，表明逐样本 N-guided signal 包含超越 gold-class 质量分配的信息。
- **H4（rank mechanism）**：当前不作为 canonical v1 核心假设；只有 Gate B 通过后才通过单独 addendum 启动。

## 2. 现有证据与重构动机

以下结果仅用于设计本协议，不作为 canonical conclusion：

| Exploratory condition, seed 42 | MNLI matched | HANS non-entailment |
|---|---:|---:|
| Standard LoRA | 85.00% | 20.21% |
| Subtraction + N-guided reweighting | 85.58% | 30.79% |
| N-guided reweighting only | 85.20% | 39.14% |
| Subtraction only | 85.38% | 22.26% |
| Neither subtraction nor reweighting | 85.44% | 27.27% |

当前单种子快照显示，N-guided reweighting 可能是主要增益来源，而 subtraction 可能降低 HANS non-entailment。该模式必须通过相同数据、相同训练种子、共享 Phase-1/2 初始化的配对实验重新验证。

小规模 rank controls 同样仅属探索性：默认 P16/N4 并未优于 equal-rank controls，且所有已记录的 N-only 模型在 HANS non-entailment 上为 0%。因此 rank-differential 机制不是预设结论。

## 3. 核心设计原则

1. **保留现有架构**：不重写双 LoRA 注入、TIES-style merge、Phase-1/2/3 主流程或已有数据接口。
2. **最小归因矩阵**：使用 subtraction × reweighting 的 2×2 消融，加上 Standard LoRA 和 class-prior control。
3. **配对控制**：同一种子下的双适配器变体共享完全相同的 Phase-1/2 checkpoint。
4. **固定数据**：数据抽样和模型训练随机性分离。
5. **开发/最终评估分离**：从现在起只用 HANS-train dev 进行开发决策；官方 HANS evaluation 仅用于冻结后的最终评估。
6. **效果量优先**：报告百分点差异、种子间变异和置信区间，不以单个 p 值决定论文结论。
7. **证据决定叙事**：若 subtraction 或 rank asymmetry 不成立，论文必须转向 N-guided mitigation，而不是隐藏负结果。

## 4. Canonical 前的必要最小代码修正

以下修正是 canonical v1 的硬性前置条件。

### 4.1 分离数据种子与训练种子

新增并冻结：

```text
data_seed = 42
hans_split_seed = 42
training_seeds = [42, 123, 2024, 3407, 777]
```

`data_seed` 决定：

- 100,000 条 MNLI train 子集；
- 5,000 条 MNLI validation-matched 子集；
- PoE/z-filter 等 baseline 的对应数据子集；
- 需要固定的分析样本来源。

`training_seed` 仅决定：

- 模型和 LoRA 初始化；
- dropout；
- batch shuffle；
- random layer control；
- 其他训练随机过程。

所有方法在同一个 training seed 下必须使用相同 MNLI train/validation 样本。

### 4.2 建立 HANS-train build/dev 划分

使用官方 `heuristics_train_set.txt`，按以下联合字段进行确定性分层：

```text
gold_label × heuristic × subcase
```

冻结比例：

- HANS-train build：80%；
- HANS-train dev：20%；
- HANS evaluation：官方 `heuristics_evaluation_set.txt`，不参与上述划分。

划分算法同样冻结：在每个联合 stratum 内，先按稳定 `pairID` 排序，再使用 NumPy `default_rng(42)` 生成排列；前 `floor(0.20 × n_stratum)` 条进入 dev，其余进入 build。若某个 stratum 少于 5 条，则全部进入 build并在 manifest 中记录。不得因模型结果重新抽取 split。

用途：

- build：Phase 2 shortcut-oriented training、Phase 2.5 reference/query sampling 和层选择；
- dev：smoke/pilot 检查和实现验证；任何会改变本协议的开发决策必须先发布 addendum；
- evaluation：canonical 模型完成后的一次性最终评估。

任何样本不得同时出现在 build 和 dev。需要保存原始 row ID/pair ID 及 split checksum。

### 4.3 禁止中间阶段读取官方 HANS evaluation 指标

Canonical driver 中：

- Phase 1 结束后不评估官方 HANS evaluation；
- Phase 2 结束后不评估官方 HANS evaluation；
- Phase 3 每个 epoch 不评估官方 HANS evaluation；
- 不允许用官方 HANS evaluation 选择 checkpoint、层、rank、alpha、beta、gamma 或 stopping epoch。

允许在 HANS-train dev 上进行中间诊断。最终 evaluation 必须在配置冻结并完成该 seed 的训练后执行。

### 4.4 补齐核心实验标签

新增两个最小实验条件：

1. `staged_neither`：`no_subtraction=True` 且 `phase3_debias_reweight=False`；
2. `class_prior_reweight`：不做 subtraction，Phase 3 权重只依赖 gold class，不读取 N-adapter 的逐样本置信度。

Class-prior control 使用以下冻结定义。首先在固定 MNLI 训练集上用共享 Phase-2 N-adapter 计算：

```text
r_i = (1 - p_N(y_i | x_i))^gamma
a_c = mean(r_i | y_i = c)
```

其中 `gamma=2.0`，`c` 为 MNLI gold class。Class-prior variant 对样本 `i` 使用 `a_{y_i}` 作为 raw weight，再与 N-guided variant 一样在每个 batch 内归一化到均值 1。`a_c` 只允许由训练集计算，并随共享 checkpoint 一起保存；不得使用 MNLI validation 或任何 OOD/evaluation labels 估计。

### 4.5 统一结果 schema

每个方法/种子必须保存：

- 完整 config；
- `data_seed`、`hans_split_seed`、`training_seed`；
- Git commit 和 dirty/clean 状态；
- Python、PyTorch、Transformers、Datasets、CUDA、GPU 和驱动版本；
- 完整运行命令；
- 起止时间、wall time、退出状态和峰值显存；
- aggregate metrics；
- HANS 逐样本预测；
- 其他评估集逐样本预测或可重建的稳定 ID；
- selected layers 和 Phase-2.5 analysis output；
- checkpoint 来源与 SHA-256 checksum。

JSON 必须符合标准：缺失值写为 `null`，禁止写入 `NaN`、`Infinity` 或 `-Infinity`。

### 4.6 保存 HANS 逐样本预测

每条 HANS 记录至少保存：

```text
pair_id
gold_label
predicted_label
entailment_probability
heuristic
subcase
training_seed
method_tag
checkpoint_hash
```

推荐格式为 UTF-8 JSONL。聚合指标必须能够从该文件重新计算。

### 4.7 测试补充

在 canonical 运行前，测试至少覆盖：

- `data_seed` 不随 training seed 改变样本 ID；
- HANS build/dev/evaluation 三者无样本交集；
- 六个核心条件只改变其声明的实验因素；
- 同一种子下双适配器变体加载同一 Phase-2 checkpoint hash；
- `trim_ratio=0.2` 的实际语义为保留绝对值最大的 20% N-delta 元素；
- 标准 JSON 序列化拒绝非有限数；
- HANS aggregate metrics 可由逐样本文件完全复算；
- official HANS evaluation 未在中间训练日志中出现。

## 5. 冻结模型与训练配置

除核心实验条件明确覆盖的字段外，canonical v1 使用以下配置：

| 类别 | 冻结值 |
|---|---|
| Base model | `roberta-base` |
| Number of labels | 3 |
| Max sequence length | 128 |
| MNLI train size | 100,000 |
| MNLI validation-matched size | 5,000 |
| Batch size | 32 |
| Main learning rate | `1e-3` |
| Weight decay | `0.01` |
| Warmup ratio | `0.06` |
| Max gradient norm | `1.5` |
| Precision | FP32, `fp16=False` |
| Target modules | `query`, `value` |
| P rank | 16 |
| N rank | 4 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Merge alpha | 1.25 |
| Subtraction beta | 0.5 |
| Trim ratio | 0.2, 即保留 N-delta 绝对值最大的 20% |
| Phase 1 epochs | 3 |
| Phase 2 epochs | 2 |
| Phase 3 epochs | 5 |
| Phase 2 MNLI mix ratio | 0.10 |
| Phase 2 batches/epoch | 3,125 |
| N learning-rate multiplier | 2.0 |
| Phase-3 N reweight gamma | 2.0 |
| Layer selection | enabled where subtraction is enabled |
| KL calibration batches | 8 |
| KL candidates | 8 |
| Selected layers | 4 |
| kNN mode | `pd_and_early_wrong` |
| kNN k | 10 |

### 5.1 Phase 2 数据构成

保留现有 Phase-2 设计以避免不必要的算法重写：每个 epoch 共 100,000 个训练位置，其中约 10% 为 MNLI，约 90% 为 HANS-build entailment。HANS 数量不足时允许按现有实现有放回采样。

论文 Methods 必须披露该有放回采样和明显的 entailment label concentration。它是设置 N-adapter 的设计条件，也是第 6 节 class-prior control 必须存在的原因。

### 5.2 层分析的解释限制

保留现有 KL + kNN 层选择，以最小化方法改动，但论文必须把相关量称为 candidate-layer stability。除非后续另有完整层序列和独立机制验证，否则不得把它解释为真实 prediction depth 或 shortcut 的因果定位。

## 6. 核心 Canonical 矩阵

核心阶段产生：

```text
6 conditions × 5 training seeds = 30 canonical method-seed result cells
```

物理执行作业还包括每个 seed 一次共享 Phase-1/2 checkpoint 准备，因此预计为 5 个共享准备作业加 30 个方法分支/最终结果作业。共享准备作业不是额外统计条件，也不计入 `n`。

| Tag | Subtraction | Phase-3 weighting | 目的 |
|---|---:|---|---|
| `standard_lora` | 否 | 普通 CE | 端到端 Standard LoRA 基线 |
| `full_sr` | 是 | N-guided | 完整方法 |
| `subtraction_only` | 是 | 普通 CE | subtraction 的主效应 |
| `reweight_only` | 否 | N-guided | N-guided reweighting 的主效应 |
| `staged_neither` | 否 | 普通 CE | 分阶段训练与 schedule-matched control |
| `class_prior_reweight` | 否 | class-only | 排除标签先验重平衡解释 |

### 6.1 配对 checkpoint 规则

对每个 training seed：

1. 训练一份共享 Phase-1/2 双适配器 checkpoint；
2. 记录 checkpoint hash；
3. `full_sr`、`subtraction_only`、`reweight_only`、`staged_neither` 和 `class_prior_reweight` 都从该 checkpoint 分支；
4. 不同分支不得重复训练 Phase 1/2 后再声称是严格配对消融；
5. `standard_lora` 独立训练，但使用相同 MNLI 数据、training seed 和总 epoch/compute-matched 设计；
6. `staged_neither` 用于弥补 Standard LoRA 与分阶段 schedule 不完全相同的问题。

### 6.2 Canonical 运行顺序

为减少选择性报告风险，使用固定循环轮转。基础顺序为：

```text
[standard_lora, full_sr, subtraction_only,
 reweight_only, staged_neither, class_prior_reweight]
```

按 training seed 在冻结列表中的索引向左轮转：seed 42 轮转 0 位、123 轮转 1 位、2024 轮转 2 位、3407 轮转 3 位、777 轮转 4 位。共享 Phase-1/2 checkpoint 必须先完成。运行失败不得通过改变方法顺序、超参数或 seed 来规避。

## 7. 指标与统计分析计划

### 7.1 主要终点

**Primary robustness endpoint**：HANS evaluation non-entailment accuracy。

选择原因：该指标直接反映模型在 shortcut-conflicting HANS 示例上的表现，避免 overall accuracy 被高 entailment accuracy 掩盖。

### 7.2 Utility constraint

**Primary utility endpoint**：MNLI validation-matched accuracy。

相对 `standard_lora` 的预设非劣容忍度：

```text
delta_MNLI >= -0.5 percentage points
```

最终报告同时给出 paired mean difference 及其 95% CI。只有点估计满足阈值但区间跨越较大负效应时，应表述为“utility preservation 不确定”，不得写成已证明非劣。

### 7.3 次要终点

- HANS overall accuracy；
- HANS entailment accuracy；
- lexical overlap、subsequence、constituent 三类 heuristic accuracy；
- 每个 HANS subcase 的 accuracy；
- e-SNLI、ANLI、SNLI-hard、WANLI accuracy；
- P-only/N-only branch diagnostics；
- selected-layer stability；
- 参数量、训练步数、wall time 和峰值显存。

### 7.4 推断单位与重复定义

- 主要推断单位：独立 training seed，`n=5`。
- 同一 seed 下不同方法共享数据和 Phase-1/2 checkpoint，因此方法差异按 paired comparison 处理。
- 30,000 条 HANS evaluation 样本不是 30,000 次独立模型训练重复，不能用于夸大方法层面的精度。
- HANS 逐样本 bootstrap 仅量化固定模型下的测试集采样不确定性，是种子间推断的补充，不替代 `n=5` 的训练重复。

### 7.5 汇总与区间

每个方法报告：

- 五种子 mean；
- sample SD，使用 `ddof=1`；
- 最小值和最大值；
- 每个预设方法对比的 seed-wise paired difference；
- paired mean difference 的 95% t interval，`df=4`。

对固定 seed 的 HANS 方法对比，执行分层 paired bootstrap：

- 10,000 次重采样；
- 在 `gold_label × heuristic × subcase` 内分层；
- 每次对相同 pair ID 同步抽样两种方法；
- 报告 accuracy difference 的 percentile 95% CI。

### 7.6 预设主要比较

1. `reweight_only - standard_lora`：端到端 mitigation 是否存在；
2. `reweight_only - staged_neither`：N-guided reweighting 是否提供增益；
3. `full_sr - reweight_only`：subtraction 是否提供额外增益；
4. `reweight_only - class_prior_reweight`：增益是否超越标签先验控制。

主要报告以 effect size 和 CI 为主，不强制依赖 NHST。若论文报告上述比较的 p 值，则：

- 使用 paired seed-level test；
- 报告精确 p 值；
- 对同一主要比较族采用 Holm correction；
- 不得用“一个显著、另一个不显著”替代直接差异或交互比较。

### 7.7 缺失、失败与排除

- 不预设排除任何成功完成的 seed。
- 基础设施错误、OOM、断连或损坏输出必须写入 `status.json`，修复后以相同配置和相同 seed 重跑。
- 若修复改变数值计算、数据或模型行为，必须升级协议版本并重启受影响的 canonical 批次。
- 不允许把失败 run 记为 `NaN` 后继续计算均值。
- 不允许因结果不利而替换 seed。

## 8. 证据闸门与论文转向规则

### Gate A：是否存在可投稿的 mitigation signal

`reweight_only` 或 `full_sr` 必须同时满足以下机械判定规则：

1. HANS non-entailment 相对 Standard LoRA 的 paired mean difference > 0；
2. 该差异的双侧 95% paired t interval 下界 > 0；
3. 至少 4/5 seeds 的差异为正；
4. MNLI paired mean difference 不低于 -0.5 pp；
5. MNLI 非劣性使用单侧 95% paired t lower bound，且下界不低于 -0.5 pp。

若 Gate A 不通过：停止扩展实验，重新设计方法，不进入第二模型或 rank mechanism 阶段。

### Gate B：subtraction 是否能作为核心贡献

比较 `full_sr - reweight_only`：

- 只有当 HANS non-entailment paired mean difference > 0、双侧 95% CI 下界 > 0、至少 4/5 seeds 为正，并且 MNLI 非劣性条件通过时，才保留 selective subtraction 为核心组件；
- 若效果接近 0、不稳定或为负：论文主线转为 N-guided shortcut-aware reweighting；subtraction 作为负结果、trade-off 或诊断性消融；
- 不得因 full method 名称已写入旧稿而保留未经支持的 subtraction 主张。

### Gate C：N-guided reweighting 是否成立并排除标签重平衡解释

首先比较 `reweight_only - staged_neither`：只有当 HANS non-entailment paired mean difference > 0、双侧 95% CI 下界 > 0 且至少 4/5 seeds 为正时，才判定 N-guided reweighting 本身提供稳定增益。

随后比较 `reweight_only - class_prior_reweight`：

- 只有当 HANS non-entailment paired mean difference > 0、双侧 95% CI 下界 > 0 且至少 4/5 seeds 为正时，才允许称为 shortcut-aware，但仍不称为机制性 unlearning；
- 若两者相当：只能主张 label-aware/class-aware mitigation；需要重新评估论文新颖性；
- 若 class-prior 更优：停止 shortcut-specific 叙事，优先重新设计 N-adapter 训练信号。

### Gate D：是否保留 Rank-Differential 标题

只有在 Gate B 通过后，才允许启动 scale-controlled rank experiments。要保留 `Rank-Differential`，必须证明：

1. P16/N4 相对 equal-rank 和 reversed-rank controls 有一致优势；
2. rank 变化与 LoRA scaling 变化被分离控制；
3. N-only 行为不是单纯 entailment-class collapse；
4. 结论在至少 3 个 full-budget seeds 上成立，并在最终主张中披露不确定性。

若不满足，标题和摘要必须删除 `Rank-Differential` 的因果暗示。

## 9. 条件性扩展，而非立即执行

### 9.1 Baseline 扩展

Gate A 通过后，再执行：

- JTT；
- PoE with hypothesis-only bias model；
- z-filtering；
- 必要时加入一个经过核实的当前强相关 shortcut-mitigation baseline。

执行规则：

- 首先 3 seeds 作为资源闸门；
- 任何进入论文主表的 baseline 必须补足相同 5 seeds；
- 仅 3 seeds 的结果只能标记为 exploratory/appendix，不能与 5-seed 主方法作对等确定性比较；
- NegMerge 和 naive subtraction 若持续出现类别塌缩，可保留为单种子诊断，不必浪费 5-seed 预算。

### 9.2 Rank controls

仅 Gate B 通过后执行。候选矩阵：

- default asymmetric：P16/N4；
- equal low：P4/N4；
- equal high：P16/N16；
- reversed：P4/N16。

在开始前必须通过协议 addendum 冻结 branch-specific scaling，避免把 rank capacity 与 `lora_alpha/rank` 的幅度变化混为一谈。

### 9.3 第二模型或第二数据路线（C 条件性扩展）

仅当核心信号通过 Gate A，且论文主贡献已由 Gate B/C 确定后，才选择一个扩展：

- 第二 encoder architecture；或
- 一个未参与现有开发的新 shortcut/counterfactual NLI benchmark。

该扩展用于验证边界和泛化，不用于挽救核心实验失败。具体模型/数据必须另行冻结 addendum，不能临时选择有利结果。

## 10. 明确不执行的工作

Canonical v1 目前不包含：

- 全量 393k MNLI 训练；
- 立即运行第二模型；
- 重写 LoRA 注入或 merge architecture；
- 重跑全部 alpha/beta/trim/top-k/target-module 一次一变量网格；
- 对已明显塌缩的 naive subtraction/NegMerge 做 5-seed 全预算；
- 根据 HANS evaluation 挑选最佳超参数；
- 在核心证据闸门前制作机制性可视化或强结论图。

## 11. 计算环境与可复现性

### 11.1 硬件角色

- Local RTX 5080：代码测试、smoke run、数据 split 验证、输出 schema 验证；
- Colab A100：canonical training runs；
- Canonical core 必须使用 A100；同一 seed 的全部配对条件必须在同一 GPU 型号与精度设置下完成。

如果 A100 无法继续提供，停止 canonical core，并在切换 GPU 前发布协议 addendum 和新结果目录；不得在 `canonical_v1` 内混用 GPU 型号。

### 11.2 软件环境

Canonical 前必须生成锁定环境记录，但不在本协议中虚构尚未读取的版本号。每个 run 自动记录实际版本，并保存：

```text
python --version
torch.__version__
transformers.__version__
datasets.__version__
CUDA runtime/driver
GPU model
pip freeze or equivalent lock snapshot
```

### 11.3 随机性分类

本实验属于 stochastic 且 environment-sensitive。相同 seed/commit/environment 应高度接近，但不要求跨 GPU bitwise identical。复现检查关注：

- 数据和 split checksum 完全一致；
- 配置和 checkpoint hash 一致；
- 输出 schema 与样本数量一致；
- 主要指标在预先声明的容忍范围内；
- 不比较 wall-clock timing 的精确一致性。

## 12. Canonical 结果目录

禁止覆盖历史 `ties_results`。Canonical v1 使用：

```text
ties_results/
  canonical_v1/
    protocol_snapshot/
      FROZEN_EXPERIMENT_PROTOCOL.md
      protocol_sha256.txt
    manifests/
      data_manifest.json
      environment_manifest.json
      run_matrix.json
    seed_42/
      shared_phase2/
      standard_lora/
      full_sr/
      subtraction_only/
      reweight_only/
      staged_neither/
      class_prior_reweight/
    seed_123/
    seed_2024/
    seed_3407/
    seed_777/
    aggregate/
      canonical_runs.jsonl
      canonical_summary.json
      paired_effects.json
      canonical_summary.md
```

每个方法目录至少包含：

```text
config.json
run_manifest.json
status.json
metrics.json
hans_predictions.jsonl
selected_layers.json
stdout.log
stderr.log
```

若保存 checkpoint，放入 `model/` 或 `checkpoints/`，并记录 checksum。

## 13. 预期执行入口

后续实现计划应提供单一 canonical driver，预期接口冻结为：

```powershell
python run_canonical.py `
  --stage core `
  --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md `
  --output-dir ties_results/canonical_v1
```

Driver 必须：

- 默认 resume，跳过完整且 checksum 合法的 run；
- `--fresh` 只能写入新的空目录，不得删除或覆盖已有 canonical 结果；
- 运行前验证 Git clean、protocol hash、data manifest 和环境 manifest；
- 增量写入 run status；
- 单个 run 失败时停止并报告，不自动改参重试；
- 支持先生成共享 Phase-2 checkpoint，再分支执行五个双适配器条件。

### 13.1 监控与超时

- 每 5 分钟检查进程状态、`status.json` 和最新日志时间戳；
- 连续 60 分钟没有 batch/epoch 进度时标记 `STALL_WARNING`，但不自动终止；
- 单个共享准备作业或方法分支的 hard timeout 为 12 小时；只有超过 hard timeout 才允许自动终止；
- OOM、非有限 loss、数据下载失败、checkpoint hash 不匹配或 prediction 行数不一致时立即把 run 标记为 failed 并停止该 seed 后续分支；
- 不自动修改 batch size、精度、学习率或数据量后重试。

## 14. 预运行检查清单

只有全部通过后，才能开始 canonical core：

- [ ] 协议已由用户复核；
- [ ] Git working tree clean；
- [ ] Canonical commit 已记录；
- [ ] 依赖版本已冻结并保存；
- [ ] `data_seed=42` 与五个 training seeds 已写入 config；
- [ ] MNLI 样本 ID 与 checksum 已保存；
- [ ] HANS build/dev/evaluation overlap test 通过；
- [ ] official HANS evaluation 中间访问被禁用；
- [ ] 六个核心 condition test 通过；
- [ ] checkpoint-sharing test 通过；
- [ ] HANS prediction schema 与重算测试通过；
- [ ] JSON finite-value test 通过；
- [ ] Local 5080 smoke run 完成；
- [ ] A100 单 seed 小规模环境 smoke run 完成；
- [ ] 同一 A100 环境下重复一个 smoke condition，主要 accuracy 的绝对差异不超过 0.5 pp；
- [ ] `canonical_v1` 目录为空或尚不存在；
- [ ] protocol snapshot 和 SHA-256 已写入输出目录。

## 15. 运行完成后的完整性检查

- [ ] 30/30 core runs 状态为 success；
- [ ] 每个方法恰好包含五个预定 seeds；
- [ ] 同 seed 双适配器分支的 Phase-2 checkpoint hash 一致；
- [ ] 所有 HANS 文件行数、pair IDs 和 gold labels 一致；
- [ ] aggregate HANS metrics 可由 predictions 完全复算；
- [ ] 无非有限 JSON 数值；
- [ ] 无未记录的配置漂移；
- [ ] paired effects 使用相同 seed 对齐；
- [ ] 主要与次要终点标记一致；
- [ ] Gate A/B/C/D 决策有机器可读和 Markdown 记录；
- [ ] 论文结论没有超越通过的证据闸门。

## 16. 论文结果写作规则

### 16.1 允许的结果表达

- “Across five training seeds, method X improved HANS non-entailment accuracy by Y percentage points relative to Standard LoRA (paired mean difference, 95% CI ...), while the MNLI difference was ... .”
- “The subtraction component did/did not provide additional improvement over N-guided reweighting.”
- “Class-prior controls suggest that the gain is/is not explained solely by label-level reweighting.”
- “These results support mitigation under the evaluated model and datasets; they do not demonstrate erasure of shortcut knowledge.”

### 16.2 禁止的结果表达

- 用单一 seed 的最好结果代表方法；
- 把 HANS 样本数量写成独立实验 `n`；
- 用 overall HANS 掩盖 entailment/non-entailment trade-off；
- 将无显著差异写成方法相等；
- 将一个比较显著、另一个不显著解释为两者显著不同；
- 只报告有利 heuristic/subcase；
- 把 rank、layer 或 branch correlation 写成机制因果结论。

## 17. 风险登记

| 风险 | 严重度 | 缓解措施 |
|---|---|---|
| N-adapter 只学习 entailment label prior | P0 | class-prior control、N-only diagnostics、Gate C |
| Subtraction 不提供额外增益 | P0 for old framing | Gate B；转向 reweighting 主线 |
| HANS evaluation 已在历史开发中被多次观察 | P1 | 从现在起使用 HANS-train dev；冻结后一次性 canonical evaluation；论文如实限定 |
| 五 seeds 的 CI 较宽 | P1 | 配对设计、报告原始 seed 值、谨慎语言，不扩大伪重复 |
| Rank 与 LoRA scaling 混杂 | P0 for rank claim | Gate D 前实施 scale-controlled addendum |
| Baseline 指标 schema 不统一 | P1 | canonical schema、完整 manifest、统一 WANLI evaluation |
| JSON `NaN` 或不完整 run 污染汇总 | P1 | `allow_nan=False`、status validation、失败 run 不聚合 |
| GPU/环境漂移 | P1 | A100 canonical、环境 manifest、同 seed 配对同硬件 |
| 运行成本过高 | P2 | 共享 Phase-1/2 checkpoint、条件性扩展、停止规则 |

## 18. 协议变更日志

### v1.0 — 2026-08-07

- 基于现有代码与结果的只读审计建立首个冻结版本；
- 将立即执行的 55-run 方案改为 30-run 核心归因矩阵；
- 冻结 100k MNLI、5 training seeds、独立 data seed；
- 新增 HANS-train build/dev 和官方 evaluation 最终评估规则；
- 新增 class-prior reweight 与 staged-neither controls；
- 新增统计推断单位、配对区间、逐样本 bootstrap 和 Holm policy；
- 新增 Gate A-D，用结果决定 subtraction、reweighting 和 rank-differential 叙事；
- 第二模型/数据与 rank controls 保持条件性扩展。

---

**当前状态**：协议设计已获用户口头批准；尚未执行 canonical v1，Verification Status 保持 `UNVERIFIED`。
