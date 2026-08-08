# 阶段 1 实验报告：Canonical Infrastructure 最小代码改造

> **Material Passport**
>
> - Origin Skill: `academic-research-suite / experiment-agent`
> - Origin Mode: `validate`
> - Date: `2026-08-07`
> - Verification Status: `VERIFIED`（仅限阶段 1 的轻量单元测试、契约测试、编译与 Git 完整性）
> - Real ML/GPU Integration: `UNVERIFIED — Stage 2`
> - Version: `stage1-canonical-infrastructure-v1`
> - Frozen Protocol SHA-256: `33c91083745b63730266d02626fb6d3057976c5b0a770904eeb9ee644f89d004`
> - Branch: `codex/stage1-canonical-infrastructure`
> - Isolated Worktree: `E:\Learning\LoRA Project\TIES-Unlearning-Diff-LoRA\.worktrees\stage1-canonical-infrastructure`
> - Verified Code Head: `b1d5a2920e4fc337617899d752f4ed4ced15f3cf`

## 1. 结论

清单阶段 1 已完成。代码现在具备启动 `canonical_v1` smoke 验证所需的最小基础设施：独立数据/训练随机种子、确定性 HANS build/dev 划分、official HANS 最终评估隔离、冻结六条件矩阵、class-prior control、逐样本 HANS predictions、统一 final evaluation battery、严格 JSON、共享 Phase-1/2 checkpoint、固定轮换顺序、失败即停和带校验和的 resume。

本阶段没有运行 30 个正式实验，没有生成 canonical 结果，没有修改论文结论，没有新增第二模型，也没有覆盖或修改历史 `ties_results/`。

阶段 1 的验证状态不等于实验流水线已在真实 ML/GPU 环境中通过。当前 shell 缺少 PyTorch、Transformers、Datasets、NumPy 和 pytest；真实依赖、数据集、RTX 5080/A100、tiny-data 端到端训练、日志隔离和数值复现必须在阶段 2 验证。

## Stage 2 预结果勘误（2026-08-08）

本报告第 2 节第 2 项及第 4.2 节将 HANS `pairID` 作为 build/dev/evaluation 可直接全局比较的稳定 ID，属于阶段 1 未读取真实数据时的错误假设。阶段 2 首次真实预运行确认：train 与 evaluation 两个官方文件各自使用 `ex0` 至 `ex29999`，这些编号是源文件局部 ID，不能跨文件直接判重。

更正后的契约是：原始 `exN` 只作为同一文件内的冻结 split/cap 排序键，以保持既定数据成员；全局 artifact 分别使用 `hans_train::exN` 与 `hans_evaluation::exN`。同时，`canonical_data_manifest_v3` 持久化不含 pair ID 的精确内容哈希及 ID/content 联合校验和，并要求分区内重复与分区间交集重算结果均为零。本勘误发生在任何成功 smoke 训练或 canonical 结果产生之前，不改变阶段 1 的 seed、80/20 比例、训练配置或模型行为；本报告顶部的 protocol SHA-256 仍是阶段 1 当时快照的历史记录。

## 2. 本阶段范围

### 已完成

1. 将 `data_seed=42`、`hans_split_seed=42` 与五个 `training_seed` 解耦。
2. 对 HANS-train 按 `gold_label × heuristic × subcase` 分层；每层先按 `pairID` 排序，再用新的 `default_rng(42)` 排列；`floor(0.20 × n)` 进入 dev；小于 5 条的层全部进入 build。
3. 分离 HANS build、dev 和 official evaluation loader，并在训练代码中将 official evaluation 延迟到最终评估。
4. 冻结六条件及五种子轮换顺序，加入 `staged_neither` 和 `class_prior_reweight`。
5. 实现训练集上估计的 class-prior 权重、批内均值归一化及共享 checkpoint 持久化。
6. 保存可复算的 HANS JSONL predictions 和严格 aggregate metrics。
7. 统一 Standard LoRA 与双适配器的 MNLI、HANS、e-SNLI、ANLI、SNLI-hard、WANLI final metrics schema。
8. 实现严格、原子 JSON/JSONL 写入，递归拒绝 `NaN`、`Infinity` 和 `-Infinity`。
9. 实现共享 Phase-1/2 checkpoint、同源五分支、固定执行顺序、增量 status、失败停止与 checksum resume。
10. 实现 protocol/data/environment/checkpoint/Git provenance 绑定。

### 明确未做

1. 未下载或实际读取 Hugging Face MNLI/HANS。
2. 未在真实 PyTorch/Transformers/Datasets 环境导入并执行训练模块。
3. 未在 RTX 5080 或 A100 上运行 smoke/canonical training。
4. 未创建正式 `ties_results/canonical_v1/` manifests 或结果目录；代码只提供在 `--fresh` 时创建它们的能力。
5. 未验证 official HANS evaluation 在真实训练日志中只出现一次；阶段 1 只验证了访问策略和代码路径。
6. 未验证其他 OOD 数据源在当前网络/缓存环境中的可用性。
7. 非 HANS OOD 评估目前已统一 aggregate schema，但其逐样本 predictions 或显式稳定 ID manifest 尚未在真实数据上物化；阶段 2 必须在预运行完整性检查中补齐或验证可重建 ID。
8. runner 已实现失败即停与恢复，但冻结协议中的 5 分钟外部监控、60 分钟 `STALL_WARNING` 和 12 小时 hard timeout 尚未在真实作业上实现/验证；正式 core 前必须补齐运行监控层。

## 3. 冻结要求—实现—证据映射

| 冻结要求 / 清单项 | 主要实现 | 自动化证据 | 状态 |
|---|---|---|---|
| `data_seed` 与 `training_seed` 分离 | `configs/config.py`, `data/dataloader.py`, `run_multiseed.py` | `test_legacy_seed_alias_changes_training_seed_only`, `test_training_seed_cannot_change_fixed_data_ids` | VERIFIED |
| HANS 80/20 分层、small-stratum 和无交集 | `canonical/data.py`, `data/dataloader.py` | `tests/test_canonical_data.py` 六项契约测试 | VERIFIED |
| official HANS 不用于中间阶段 | `canonical/evaluation_policy.py`, `training/trainer.py`, `training/baseline.py` | `test_only_final_event_can_request_official_evaluation`, `test_unknown_evaluation_event_fails_closed` | VERIFIED（代码/策略）；真实日志待阶段 2 |
| 六条件与固定轮换顺序 | `canonical/conditions.py` | `tests/test_canonical_conditions.py` | VERIFIED |
| `staged_neither` | `canonical/conditions.py` 的 subtraction=`False`, weighting=`none` | factor matrix 与 config-isolation 测试 | VERIFIED |
| `class_prior_reweight` | `training/weighting.py`, `training/trainer.py` | 手算 prior、输入验证、批内均值 1 测试 | VERIFIED（纯计算）；真实 torch batch 待阶段 2 |
| 逐样本 HANS predictions | `training/evaluate.py`, `canonical/hans.py` | schema、重复 ID、概率范围、literal-row 复算测试 | VERIFIED |
| 统一最终评估 battery | `canonical/results.py`, `training/baseline.py`, `training/trainer.py` | 缺少 WANLI 或必要 metric field 即失败；WANLI 不可用时允许标准 JSON `null` | VERIFIED（schema）；真实各数据源待阶段 2 |
| config/environment/Git/status/metrics schema | `canonical/artifacts.py`, `canonical/backend.py`, `canonical/runner.py` | runner 成功/失败/resume 测试；run manifest 绑定 data/environment SHA-256 | VERIFIED（schema）；真实 manifest 内容待阶段 2 |
| 标准 JSON 禁止非有限数 | `canonical/artifacts.py` | nested JSON、JSONL 原子失败测试 | VERIFIED |
| 精确 TIES trim 比例 | `models/ties_lora.py` | cutoff tie、ratio=1、非法 ratio 测试 | VERIFIED |
| 共享 Phase-1/2 checkpoint | `utils/optim_utils.py`, `training/trainer.py`, `canonical/backend.py`, `canonical/runner.py` | 一次 prepare、五分支收到同一 SHA-256；后端 path/hash 适配测试 | VERIFIED（调度契约）；真实 state load 待阶段 2 |
| 固定顺序、失败停止、checksum resume | `canonical/runner.py` | rotation、non-empty fresh、prepare/branch failure、corruption resume 测试 | VERIFIED |
| 单一入口 | `run_canonical.py` | `python run_canonical.py --help` 在无 ML 依赖环境成功 | VERIFIED |
| 历史结果不覆盖 | `.gitignore`, `--fresh` 空目录规则 | `git diff --name-only 0ef127c -- ties_results` 无输出 | VERIFIED |

## 4. 关键设计与行为

### 4.1 Seed 契约

固定数据成员只由 `data_seed=42` 决定，HANS 划分只由 `hans_split_seed=42` 决定。模型初始化、dropout、batch shuffle 与随机层控制使用 `training_seed ∈ {42, 123, 2024, 3407, 777}`。旧的 `cfg.seed` 仅作为 `training_seed` 兼容别名，不能改变数据成员。

### 4.2 HANS 隔离

- `make_hans_build_loader`：Phase 2 与 Phase 2.5 数据来源。
- `make_hans_dev_loader`：中间诊断。
- `make_hans_evaluation_loader`：只在最终评估构造。
- `make_hans_loader`：保留为 legacy evaluation alias，不在 canonical 中间路径使用。

Split manifest 保存 build/dev/evaluation `pairID`、small strata、计数和 split checksum，并在生成时验证三者无重复、无交集。

### 4.3 六条件与共享 checkpoint

每个 training seed 先生成一份共享 Phase-2 checkpoint，并计算 SHA-256。`full_sr`、`subtraction_only`、`reweight_only`、`staged_neither` 和 `class_prior_reweight` 从同一 path/hash 分支；`standard_lora` 独立训练。runner 测试验证五个双适配器分支收到同一个 hash。

### 4.4 输出与恢复

每个 method 目录要求：

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

`run_manifest.json` 记录 protocol、data manifest、environment manifest、Git commit、命令、seeds 和共享 checkpoint provenance。`status.json` 先写 `running`，只有所有必要文件存在并完成 SHA-256 后才写 `success`。默认 resume 只跳过 `success` 且所有声明输出 hash 仍匹配的 run；任一文件损坏会重跑该 run。`--fresh` 只接受新目录或空目录，不删除已有结果。

冻结入口：

```powershell
python run_canonical.py `
  --stage core `
  --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md `
  --output-dir ties_results/canonical_v1 `
  --fresh
```

首次初始化使用 `--fresh`；后续恢复去掉 `--fresh`。在阶段 2 通过所有 smoke gates 前，不得用该入口启动 30 个正式结果单元。

## 5. Red–Green TDD 证据

每项先观察旧行为失败，再加入最小实现：

| Task | RED 证据 | GREEN 证据 |
|---|---|---|
| Seeds/conditions | `ModuleNotFoundError: canonical.conditions` / 缺少独立 seed 字段 | 5 项条件测试通过 |
| Data/HANS split | `ModuleNotFoundError: canonical.data` | 6 项数据契约测试通过 |
| Strict artifacts/HANS | 缺少 `canonical.artifacts`、`canonical.hans` | 10 项 artifact/prediction 测试通过 |
| Trim semantics | 旧 threshold 在 tie 处保留 `[1,4,5,6]` 而非精确 `[1,4]`；非正 ratio 未拒绝 | 3 项 trim 测试通过 |
| Weighting/evaluation | 缺少 `canonical.evaluation_policy`、`training.weighting` | 7 项 weighting/policy 测试通过 |
| Runner | `ModuleNotFoundError: canonical.runner` | 6 项初始 orchestration 测试通过 |
| Real backend | `ModuleNotFoundError: canonical.backend` | 2 项配置/path/hash 适配测试通过 |
| CLI | `run_canonical.py` 不存在 | `--help` 成功且未导入 ML stack |
| Manifest binding | `KeyError: data_manifest_sha256` | data/environment SHA-256 绑定回归通过 |
| Final metrics schema | 缺少 `canonical.results`；旧 Standard LoRA 无 WANLI 且最终指标位置不同 | 3 项统一 battery/schema/location 测试通过 |

## 6. 最终验证记录

### 6.1 单元与契约测试

```powershell
python -m unittest discover -s tests -v
```

结果：`Ran 46 tests ... OK`。

覆盖内容包括 42 个新增 canonical/TIES 测试和 4 个原有 sensitivity 测试；所有测试通过。

### 6.2 编译检查

```powershell
python -m compileall -q canonical configs data models training utils run_canonical.py
```

结果：exit code `0`。

### 6.3 入口检查

```powershell
python run_canonical.py --help
```

结果：exit code `0`；输出只接受 `--stage {core}`、`--protocol`、`--output-dir` 和可选 `--fresh`。

### 6.4 Git 与历史结果检查

```powershell
git diff --check
git diff --name-only 0ef127c -- ties_results
git status --short
```

阶段 1 代码检查时：diff check 通过；`ties_results` diff 无输出。文档提交后再次执行并要求 working tree clean。

## 7. 提交记录

| Commit | 内容 |
|---|---|
| `0ef127c` | 阶段 1 设计 |
| `bd9f878` | 可执行实现计划 |
| `9b6c7b0` | 冻结 seeds 与六条件 |
| `c6da810` | 确定性数据与 HANS split |
| `c8f38dd` | 严格 artifacts 与 HANS predictions |
| `644d967` | 精确 TIES trim fraction |
| `5408e91` | shared Phase-2、weighting 与 evaluation hooks |
| `87e9ff9` | resumable canonical driver |
| `00bfd55` | run-to-manifest SHA-256 绑定 |
| `dc67f50` | 统一 Standard LoRA/双适配器 final evaluation schema 与 WANLI |
| `b1d5a29` | 将所有 canonical 最终指标统一到 `metrics.final` |

这些提交均位于隔离分支，按功能拆分、可独立审查和回滚。没有合并到主分支，也没有改写历史实验结果。

## 8. 当前环境与证据边界

验证解释器：`Python 3.14.6`。

当前 shell 中：

```text
pytest=NOT_INSTALLED
torch=NOT_INSTALLED
transformers=NOT_INSTALLED
datasets=NOT_INSTALLED
numpy=NOT_INSTALLED
```

因此本报告不能声称以下事项已验证：真实 tensor 计算、LoRA 注入、checkpoint state_dict 加载、Hugging Face 数据 schema、CUDA AMP、GPU memory、下载重试、真实 HANS 行数、训练数值或 accuracy。`compileall` 只证明语法可编译，不证明依赖兼容和端到端可执行。

## 9. 阶段 2 交接条件

下一执行人应从本分支/工作树开始，先阅读冻结协议、本报告和阶段 2 清单，不得直接启动 30 个正式实验。阶段 2 至少完成：

1. 在项目真实依赖环境运行全部测试并确认版本兼容。
2. 在 RTX 5080 做 tiny-data 端到端 smoke，覆盖 Standard LoRA、shared prepare 和至少两个双适配器分支。
3. 检查真实日志，证明 Phase 1/2/2.5/Phase-3 epochs 未构造或读取 official HANS evaluation。
4. 验证两个分支加载相同 Phase-2 checkpoint path/hash，class-prior 随 checkpoint 恢复。
5. 从真实 `hans_predictions.jsonl` 独立复算 aggregate metrics 并逐字段比对。
6. 验证 MNLI/HANS IDs、split checksum、数据行数和 manifest schema。
7. 在 A100 完成小规模 smoke，并在同一环境重复一个 condition，主要 accuracy 绝对差异不超过 0.5 pp。
8. 冻结 Python、PyTorch、Transformers、Datasets、NumPy、CUDA driver/runtime、GPU 和 `pip freeze`。
9. smoke 全部通过后，确保正式 `canonical_v1` 结果目录为空或尚不存在，记录 canonical commit，再向用户申请进入阶段 3。
10. 为 e-SNLI、ANLI、SNLI-hard 和 WANLI 保存逐样本 predictions，或在 data manifest 中保存经过真实数据验证的稳定/可重建 ID 与 checksum。
11. 为正式作业接入并演练 5 分钟状态检查、`STALL_WARNING` 和 12 小时 hard timeout；不得用自动改参替代失败处理。

若阶段 2 发现任何会影响模型、数据、训练、评估或统计结论的问题，应停止、修复、重新执行阶段 1/2 相关验证，并按冻结协议判断是否需要 addendum；不得静默修改配置后开始正式实验。
