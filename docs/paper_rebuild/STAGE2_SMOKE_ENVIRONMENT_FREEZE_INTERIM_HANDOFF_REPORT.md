# Stage 2 Smoke Tests 与 Canonical 环境冻结：暂停交接报告

## Material Passport

| Field | Value |
|---|---|
| 阶段 | Paper rebuild / Stage 2 |
| 报告类型 | 中途暂停与续执行交接，不是完成报告 |
| 阶段状态 | `PAUSED / PARTIAL / BLOCKED` |
| Verification Status | `BLOCKED`（不得写为 `VERIFIED`） |
| Task 8 | 暂停；本地环境与测试通过，但真实 smoke 未通过 |
| Task 9 | 暂停；尚未启动 Colab A100 |
| Task 10 | 未启动；正式 Stage 2 报告与 checklist 验收尚不可执行 |
| Stage 3 授权 | `NO`；不得启动 30 个正式 canonical runs |
| 报告时间 | 2026-08-08 15:09:50 +08:00 / 2026-08-08 07:09:50 UTC |
| 工作树 | `E:\Learning\LoRA Project\TIES-Unlearning-Diff-LoRA\.worktrees\stage1-canonical-infrastructure` |
| 分支 | `codex/stage1-canonical-infrastructure` |
| WIP 父提交 | `c3c8a7f6ceb249945905cacf41ca83e941425f88` |
| 冻结协议 | `docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md` |
| 执行计划 | `docs/superpowers/plans/2026-08-08-stage2-smoke-environment-freeze.md` |
| 设计 | `docs/superpowers/specs/2026-08-08-stage2-smoke-environment-freeze-design.md` |

本报告将 Round 4 部分实现与本交接文档保存为一个明确的 WIP 交接提交。该提交用于恢复现场，不是 smoke-verified code commit，不得用于打包或实验。

## 1. 执行结论

Stage 2 尚未完成。

已经完成并独立审查通过的是 Stage 2 的 smoke profile、数据访问审计、runner、validator、monitor、source-only transport、A100 notebook/freeze/evidence 基础设施，以及 Task 7 最终合同。RTX 5080 的 Python/CUDA/ML 环境也已建立，真实依赖下的完整测试套件通过。

尚未完成的是：

- 一个成功的 RTX 5080 tiny-data 端到端 smoke；
- 本地 smoke 的 production validator、`pip freeze` 与最终 artifact checksums；
- Colab A100 primary smoke；
- 同一 A100 runtime 的 `full_sr` repeat 与 `<= 0.5 pp` 比较；
- canonical-targeted freeze bundle；
- A100 evidence transport verification；
- Master checklist 的 Stage 2 勾选与正式 Stage 2 完成报告。

本地 smoke 在训练前被 HANS identity gate 拦截。调查证明这是官方两个 HANS 文件重复使用 file-local `pairID=ex0..ex29999` 导致的基础设施误报，而不是 train/evaluation 真实内容泄漏。修复已推进到 Round 4，但尚未完成 v5 source-integrity 全链路接线和独立复审。

因此：旧本地 runtime 只作为审计证据保留；不得从它继续训练；不得复用旧 source ZIP 进入 Task 9；不得宣称 Stage 2 通过。

## 2. Master Checklist 状态

`PAPER_REBUILD_MASTER_CHECKLIST.md` 中 Stage 2 的以下项目必须继续保持未勾选：

- RTX 5080 tiny-data 端到端 smoke；
- 独立脚本复算 HANS 指标并核对 prediction integrity；
- smoke 输出不写入 `ties_results/canonical_v1/` 的最终证据；
- Python / CUDA / PyTorch / Transformers / Datasets / NumPy 完整环境冻结；
- A100 smoke；
- 同一 A100 runtime repeat；
- canonical 环境与完整 data IDs/checksums freeze。

虽然部分先决条件已有证据，但 Stage 2 的 checklist 是结果门，不应因基础设施单测或环境安装成功而提前勾选。

## 3. Stage 2 内部任务进度

| 内部任务 | 内容 | 状态 | 关键提交 / 证据 |
|---|---|---|---|
| Task 0 | Stage 2 plan/spec 合同 | 完成 | `8136518`，设计基线 `9789bdc` |
| Task 1 | 隔离 smoke profile | 完成 | `f274701`，修复 `a55207e` |
| Task 2 | HANS final-access 审计边界 | 完成 | `d8d5cca`，修复 `71a8ec9` |
| Task 3 | Stage 2 smoke matrix driver | 完成 | `170a918`，修复 `64466af`、`8c36e03` |
| Task 4 | 稳定 data identities / manifests | 完成后发现真实 HANS 局部 ID 假设错误 | `4aba1bb`，修复 `d098118` |
| Task 5 | 独立 artifact validator | 完成 | `3d0dcaf`，修复 `d80b5bb`、`ee9122f` |
| Task 6 | 非自动调参 monitor | 完成并复审通过 | `a2aa69e`，修复至 `49295ce` |
| Task 7 | source-only transport、A100 notebook、freeze/evidence | 完成并最终复审通过 | 最终 `a90ee3c`；当时 origin full 149/149 |
| Task 8 | 本地 RTX 5080 real smoke | `BLOCKED / PAUSED` | 环境与 full tests 通过；smoke 未进入训练 |
| Task 9 | Colab A100 primary + repeat | `NOT STARTED / PAUSED` | 无 A100 runtime、无 A100 artifacts |
| Task 10 | 最终验证、checklist、Stage 2 完成报告 | 未启动 | 本报告不是 Task 10 完成报告 |

## 4. Task 7 已冻结且获批的基础设施

Task 7 经过五轮独立高强度审查，最终提交为：

`a90ee3cfce22a83253d95cec7ceaa4747fb10588` — `fix: enforce stage2 monitor terminal state machine`

最终获批能力包括：

- source-only、parentless、deterministic execution commit；
- origin commit 与 execution commit 双 provenance；
- source package 排除 `ties_results/`、weights、环境、cache、private plan/spec/history；
- packaged execution clone 内完整测试门；
- A100 notebook fail-fast；
- primary/repeat commands 与 monitor argv/cwd 交叉绑定；
- production validator、repeat comparator 与 freeze verifier；
- no-weight evidence transport 的 exact inventory、semantic validation 与 safe extraction；
- monitor 的 unique STARTED、unique terminal、STATUS_CHECK return-code 状态机；
- sibling `ties_results/.stage2_monitor/` 的保留、忽略、导出与 repo-root 导入合同。

Task 7 最终独立复审结论为 `APPROVED`。后续 HANS 修复改变了 source/protocol/data-manifest 合同，所以真正恢复 Task 8 时必须在新 clean commit 上重新打 source package；不能把 `a90ee3c` 的旧 package 当成当前代码。

## 5. RTX 5080 环境与测试证据

### 5.1 旧 runtime（只保留，不继续实验）

| Field | Value |
|---|---|
| Runtime root | `E:\Learning\LoRA Project\stage2_runtime_local_a90ee3c_20260808T103954438` |
| Execution clone | `...\stage2_execution` |
| Origin commit | `a90ee3cfce22a83253d95cec7ceaa4747fb10588` |
| Execution commit | `e53a67b1ad125a85c2ee2368d13ac3650ad0ef9a`（detached） |
| Execution Git status | clean（2026-08-08 交接时只读复核） |
| Source ZIP SHA-256 | `8c0fe4a23ee8fced8694782e57ba38197ec9e498f8e4cf7a4baca26f2b773dcc` |
| Expectations SHA-256 | `f9a97e888793506fece3e2a22cfb1c9a09a013bbb9b71cdfebb790c03af5da05` |
| Source manifest SHA-256 | `f8851b0de59bfd5e9b973ed78a66645b33d165dae4195a123486ab26dca89b81` |

该 ZIP 只对应旧 execution commit，现已过时。它只能用于复核历史，不得上传 A100，不得作为下一次 Task 8 输入。

### 5.2 软件与硬件

| Component | Evidence |
|---|---|
| Python | CPython `3.12.13` |
| PyTorch | `2.11.0+cu128` |
| torch CUDA runtime | `12.8` |
| GPU | `NVIDIA GeForce RTX 5080` |
| Driver | `610.74` |
| GPU memory | `16303 MiB` |
| Transformers | `5.14.1` |
| Datasets | `5.0.1` |
| NumPy | `2.5.1` |
| venv size | `4,891,427,243` bytes |
| uv cache size | `4,878,117,729` bytes |

精确 GPU/import 验证最终退出 0，用时 22.1 秒。`nvidia-smi` 退出 0。真实 ML 依赖环境下的完整测试为：

`149 tests ran in 133.920s; OK`

首次 import 命令被 120.4 秒外层 runner timeout 杀死；之后获批的单假设诊断证明是外层首次加载超时，不是 PyTorch/CUDA 损坏。诊断证据：

| File | SHA-256 |
|---|---|
| `torch_importtime_diagnostic/result.json` | `229aa89342d7130e3505588720ed515f817f0cd2fdfa3a41df5d06f001c5df7f` |
| `stdout.txt` | `702fdfe87d990f14bd69dc5be87f86f9b3ff1fd5546a0a5180ca4e057f4560b9` |
| `stderr.txt` | `6e0f3388b4e441e41230f76a8ebca2f6b2b870ea65414dac3e05da5bbd89cf2d` |

大小写不影响 SHA-256 语义；路径位于 runtime root。

## 6. 本地 smoke 尝试谱系

| Attempt | 结果 | 分类 | 处理 |
|---|---|---|---|
| network sandbox attempt | Hugging Face 请求 WinError 10013 | 沙箱网络边界 | 原命令获批后在外部网络环境重新执行；证据保留 |
| accidental attempt 1b | 在未先隔离 fresh root 时误启动 | agent-control error | 被 controller 停止；合并证据保留，不作验证样本 |
| authorized formal run | `git status --porcelain` exit 128 | elevated 用户与 repo owner 不同导致 dubious ownership | 仅对一个进程使用 `GIT_CONFIG_COUNT/KEY_0/VALUE_0=safe.directory`，未改 global Git config |
| ownership-fixed formal run | HANS build/evaluation overlap `ex0...` | 真实 pre-training integrity failure | 停止；未重试；进入独立代码修复 |

最后一次正式 run：

- start: `2026-08-08T03:14:56.8542855Z`；
- monitor terminal: `CRASHED`, return code `1`；
- wall: 约 300.3 秒；
- 未创建 shared checkpoint；
- 无任何 method 训练成功；
- 未运行 validator；
- 未生成 `pip_freeze.txt`、`stage2_validation.json` 或最终 smoke checksums。

证据索引：

| Evidence | SHA-256 |
|---|---|
| final commands.json | `9fd47b6602b2dbb5b4848e56c6b0dd7d2eb096bec0cc5afe8ec3732312f84c21` |
| final `local_rtx5080.events.jsonl` | `27613bb95041d00335c506a49c6d98e4f08007dde7635562f7bc9f992be432ea` |
| git-ownership attempt events | `a686b302b132ce1ac9177ced6f720acb21adacb1b978da78e8499107ae8b9d5f` |
| network/control attempts events | `9b9f6cf11dafe19ca48d24570b0cd5e14eadf68390a255c1ad8f56b9fca0b6a1` |

历史输出根：

- `...\ties_results\stage2_smoke\local_rtx5080`；
- `...\ties_results\stage2_smoke\local_rtx5080_network_sandbox_attempt`；
- `...\ties_results\stage2_smoke\local_rtx5080_git_ownership_attempt`（空目录）；
- `...\ties_results\.stage2_monitor\` 下三份 JSONL。

不得删除、覆盖或把这些失败根当作新的 `--fresh` root。

## 7. HANS 根因的只读事实

对已授权 run 下载并缓存的官方文件进行只读解析，得到：

| Fact | Train | Evaluation |
|---|---:|---:|
| 行数 | 30,000 | 30,000 |
| file-local `pairID` 唯一数 | 30,000 | 30,000 |
| `pairID` 范围 | `ex0..ex29999` | `ex0..ex29999` |
| 两文件 local-ID 交集 | 30,000 | 30,000 |
| 五字段精确内容交集 | 0 | 0 |

`ex0` 在两个文件中的 premise/hypothesis 不同。因此 raw `pairID` 是 file-local，不是 global identity。正确合同必须同时满足：

- raw `exN` 只作为同一物理文件内的 frozen split/cap ranking key；
- global artifact identity 为 `hans_train::exN` 或 `hans_evaluation::exN`；
- build/dev 共享 train namespace；
- prefix 不进入 cap hash，保持既定 384 membership；
- 独立内容/source evidence 仍必须防止真实泄漏或上游文件语义漂移。

## 8. HANS 修复轮次与审查结论

| Round | Commit / 状态 | 通过证据 | 未通过原因 |
|---|---|---|---|
| 1 | `60aa355` | full 159/159 | qualified prefix 进入 cap hash，384 中仅 6 条与 raw-key membership 重合；内容证据未持久化；ID grammar 过宽 |
| 2 | `93dee56` | full 163/163；packaged clone 通过 | split checksum 未持久化；content joint hash 只自洽、无官方锚；2-row fake smoke 可通过；root audit 未绑定 manifest |
| 3 | `c3c8a7f` | affected 79/79；freeze 24/24；packaged inner 172 OK；full 173/173；独立 review full 173 OK | `source_file_sha256` 是 dead anchor，修改 `template/parse` 后仍可通过；协议 raw/qualified 措辞过宽 |
| 4 | WIP，随本报告交接 | RED 6/6 已复现；两个 11-field digest 已独立重算一致 | builder/validator 仅部分写入，尚未接 backend/manifest/audit/Stage2/freeze/evidence/docs，尚未运行 Round 4 GREEN |

Round 3 已关闭且独立复核通过的部分：

- official split/raw/qualified/content/joint/cap-384/full-selection anchors；
- raw-only split/small-strata 与 qualified cross-binding；
- actual production 384 membership 与独立 raw-key reference 有序完全一致；
- content record-to-ID binding、duplicate/overlap 与 rehashed tamper rejection；
- selection seed/algorithm/strata/cap/order/raw-to-qualified mapping；
- exact Stage 2 profile 和 two-ID rejection；
- root manifest audit 对 identity/split/content/selection summary 的 exact binding；
- freeze/evidence/source/package tamper tests。

Round 3 唯一 P1 是 source-file byte hashes 只写入常量/文档却没有进入 manifest/validator。审查同意的修复是：不把非语义 newline/encoding byte 差异作为 fail-closed 门，而是用 v5 `source_integrity` 严格绑定官方 11 个解析字段。

## 9. 当前 Round 4 WIP 现场

在编写本报告前，`c3c8a7f` 之上有 3 个 tracked 文件的部分修改：约 295 insertions / 6 deletions，`git diff --check` 退出 0。它们随本交接 WIP 提交保存：

- `canonical/data.py`；
- `tests/test_canonical_data.py`；
- `tests/test_stage2_validation.py`。

已完成的 Round 4 RED：6/6 预期失败，无无关 fixture error：

- v5 source-integrity builder 不存在；
- 只修改 `template` 并重建 v4 five-field evidence 仍完全相同；
- missing/extra official field 未被拒绝；
- caller-supplied digest API 未被禁止；
- v4 source-less root 未被拒绝；
- source-less audit 未被拒绝。

已部分写入但尚未完成/验证：

- `HANS_OFFICIAL_ANCHORS_V2`；
- informational raw-byte SHA-256 字段；
- train 11-field digest `841ffee28e0310f1f95d692a534f362a8a171a69d7f659ec3ed07a4205840cf5`；
- evaluation 11-field digest `5d170c471cde96e61c24d640cb50652bf7c594c4800e40d7ebf8133ec7d5df6b`；
- `build_hans_source_integrity_manifest()`；
- `validate_hans_source_integrity_manifest()`；
- pure-data tests。

11 个官方字段必须恰好是：

1. `gold_label`
2. `sentence1_binary_parse`
3. `sentence2_binary_parse`
4. `sentence1_parse`
5. `sentence2_parse`
6. `sentence1`
7. `sentence2`
8. `pairID`
9. `heuristic`
10. `subcase`
11. `template`

source-integrity 排序冻结为 `(int(pairID[2:]), raw pairID)`，train/evaluation 分开计算 canonical JSON UTF-8 SHA-256。builder 必须从 records 实算，不得接受 caller-supplied digest。

当前 WIP 不可打包、不可跑 smoke、不可称为 GREEN。

## 10. 下一执行人的精确恢复步骤

### 10.1 先完成 Round 4 v5 合同

1. 保留本 WIP 提交；不要 reset、checkout 或丢弃三个部分修改文件。
2. 将 `source_integrity` 接入 `validate_hans_manifest_identities()`，并使用 `HANS_OFFICIAL_ANCHORS_V2`。
3. `RealCanonicalBackend.initialize_manifests()` 必须从真实 train/evaluation records 实算 `hans.source_integrity`。
4. 将 data manifest schema 升级为 `canonical_data_manifest_v5`；Stage 2 validator 与 freeze 必须严格拒绝 v4。
5. `hans_manifest_identity_summary()` 和 root `manifest_identity_summary` 必须包含 source-integrity summary，并与 manifest exact cross-bind。
6. Stage 2 validator、freeze build/verify、evidence transport 必须验证两个 30,000-row source digests；重建 manifest/run/status/inventory 后的 `template/parse` tamper 仍应失败。
7. 增加 freeze/evidence rebuilt-inventory RED/GREEN；不能只测试 pure builder。
8. 协议必须把 raw byte SHA 明确标为 informational retrieval reference；fail-closed 门是全 11 parsed fields semantic digest。
9. 协议中“只输出 qualified ID”必须限定为 HANS identity entries、loader batches 和 predictions；raw `exN` 按设计保存在 split/selection integrity 子对象。
10. 同步 Stage 1 erratum、Stage 2 design/plan 与 ignored implementation report。

### 10.2 Round 4 验证门

至少重跑：

```powershell
python -m unittest tests.test_canonical_data tests.test_stage2_data_audit tests.test_canonical_runner tests.test_stage2_validation -v
python -m unittest discover -s tests -p "test_stage2_freeze.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q canonical data tests
python run_stage2_smoke.py --help
python validate_stage2_smoke.py --help
python freeze_stage2_environment.py --help
python package_stage2_source.py --help
git diff --check
```

随后提交 Round 4，并交给独立 reviewer。只有 reviewer 明确 `APPROVED`、零 open findings，才允许恢复 Task 8。

### 10.3 重新开始 Task 8，而不是续跑旧 runtime

1. 在最终获批且 clean 的 origin commit 上重新生成 source ZIP 与 expectations。
2. 创建全新的 timestamped runtime directory；不得覆盖旧 runtime。
3. clone 新 parentless execution commit，验证 detached HEAD/clean status。
4. 可复用外部包下载 cache，但 execution source/archive/commit 必须是新的。
5. 创建/验证 Python 3.12 + torch 2.11.0 cu128 环境，运行完整 real-ML tests。
6. 运行一个新的 monitored `--fresh` local primary smoke。
7. elevated context 如仍需要，只使用 process-scoped `safe.directory`，不得修改 global Git config。
8. smoke 成功后运行 production validator、严格 monitor validator、`pip freeze`、hash/size/status 证据。
9. 写/更新 Task 8 报告，保存新 source ZIP/expectations 的精确 SHA-256。

### 10.4 Task 9 仍保持未启动

只有新的 Task 8 成功后，才能把**完全相同的新 source ZIP 和 expectations**上传 Colab A100。Task 9 必须执行：

- A100 GPU gate；
- full tests；
- monitored primary；
- 同一 runtime 的新 shared checkpoint + `full_sr` repeat；
- `abs(HANS non-entailment accuracy run1 - repeat) <= 0.005`；
- full canonical data manifest；
- environment freeze；
- no-weight evidence archive；
- 下载后先 transport-verify，再 repo-root safe extraction。

## 11. 禁止宣称与禁止动作

在完成上述工作前，下一执行人不得：

- 把 Stage 2 写为完成或 VERIFIED；
- 勾选 Master Checklist 的 Stage 2 项；
- 复用旧 `stage2_source.zip` 进入 A100；
- 从旧 `local_rtx5080` failure root resume 或覆盖证据；
- 启动 Task 9；
- 启动 Stage 3 formal canonical runs；
- 把 149/173 单测通过等同于端到端 smoke 通过；
- 把 HANS local-ID overlap 说成真实内容泄漏；
- 把未完成的 Round 4 WIP 当成获批代码。

## 12. 交接证据索引

| Artifact | Path |
|---|---|
| 本报告 | `docs/paper_rebuild/STAGE2_SMOKE_ENVIRONMENT_FREEZE_INTERIM_HANDOFF_REPORT.md` |
| Frozen protocol | `docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md` |
| Master checklist | `docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md` |
| Stage 2 plan | `docs/superpowers/plans/2026-08-08-stage2-smoke-environment-freeze.md` |
| Stage 2 design | `docs/superpowers/specs/2026-08-08-stage2-smoke-environment-freeze-design.md` |
| SDD progress ledger | `.superpowers/sdd/2026-08-08-stage2-smoke-environment-freeze/progress.md` |
| Task 8 local report | `.superpowers/sdd/2026-08-08-stage2-smoke-environment-freeze/task-8-report.md` |
| HANS fix report | `.superpowers/sdd/2026-08-08-stage2-smoke-environment-freeze/task-8-hans-id-fix-report.md` |
| HANS fix brief | `.superpowers/sdd/2026-08-08-stage2-smoke-environment-freeze/task-8-hans-id-fix-brief.md` |
| Old runtime | `E:\Learning\LoRA Project\stage2_runtime_local_a90ee3c_20260808T103954438` |

`.superpowers/sdd/` 内的 task reports/briefs 受该目录自身的 ignore 规则管理，但在当前共享 workspace 中保留。真正跨 Git/机器交接时，以本 tracked 报告为主，并确认 SDD 文件是否随工作环境一并提供。

## 13. 最终状态标签

```text
Stage 2: PAUSED / PARTIAL / BLOCKED
Task 8: PAUSED; environment and tests passed; real smoke failed before training
Task 9: NOT STARTED
Task 10: NOT STARTED
Stage 3: NOT AUTHORIZED
Last independently reviewed implementation: c3c8a7f (NOT APPROVED for smoke)
Round 4: WIP handoff; not validated
```
