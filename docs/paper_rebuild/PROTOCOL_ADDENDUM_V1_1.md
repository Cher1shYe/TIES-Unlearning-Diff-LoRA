# Frozen Experiment Protocol — Addendum v1.1

## Material Passport

- Base Protocol: `FROZEN_EXPERIMENT_PROTOCOL.md` (v1.0, 2026-08-07)
- Addendum Date: 2026-08-08
- Addendum Reason: 用户决定将 smoke 与 canonical 实验全部迁移至 Colab，并将过程性防护精简到与 CCF-C/CCF-B 投稿目标匹配的程度
- Result Directory: 仍为 `ties_results/canonical_v1`（本 addendum 不改变任何会影响模型、数据、训练、评估或统计结论的配置，故按 v1.0 第 0.5 条保留结果目录；变更内容在此记录）

## 1. 变更内容

### 1.1 硬件角色（替代 v1.0 §11.1 中的本地角色）

- 取消本地 RTX 5080 smoke（原 Stage 2 Task 8）。
- Smoke 与 canonical core 全部在 Colab A100 上执行，且位于同一 runtime 类型。这满足并强化了 v1.0 "同一 seed 的全部配对条件必须在同一 GPU 型号与精度设置下完成" 的要求。
- `run_stage2_smoke.py --environment colab_a100` 的 A100 GPU 名称门保持不变。

### 1.2 代码传输（替代 source-only ZIP 合同）

- 废止 "parentless source ZIP + expectations sidecar + transport verification" 传输链。
- 代码经公共 GitHub 仓库分支 `codex/stage1-canonical-infrastructure` 以 `git clone` 获取，并 checkout 到执行时打印并记录的 commit。
- 代码 provenance 由以下机制保证：runner 的 `_assert_clean_git` 拒绝 dirty tree；每个 run 的 `run_manifest.json` 记录实际 Git commit。
- 与被废止传输链绑定的重型集成测试（打包克隆内递归重跑完整套件，硬编码 240s 子进程超时）默认跳过；如需运行，设置环境变量 `STAGE2_RUN_PACKAGING_INTEGRATION=1`。打包代码本身保留，其余轻量打包单测不变。

### 1.3 HANS source-integrity：v5 fail-closed 降级为 informational-but-verified

- `canonical_data_manifest_v4` 保持为现行 schema；不引入 v5，也不拒绝不含 `hans.source_integrity` 的既有 v4 结构。
- Backend 在初始化 manifest 时，从真实加载的官方 records 实算 `hans.source_integrity`（`hans_source_integrity_v1`：11 个官方解析字段、`(int(pairID[2:]), raw pairID)` 排序、train/evaluation 分开计算 canonical JSON UTF-8 SHA-256）。
- `validate_hans_manifest_identities` 在该字段存在时强制以 `HANS_OFFICIAL_ANCHORS_V2` 校验：结构、字段集、排序声明与两个 30,000 行 digest 任一不符即 fail closed。冻结 digest 不变：
  - train: `841ffee28e0310f1f95d692a534f362a8a171a69d7f659ec3ed07a4205840cf5`
  - evaluation: `5d170c471cde96e61c24d640cb50652bf7c594c4800e40d7ebf8133ec7d5df6b`
- 官方两文件的 raw byte SHA-256 保留为 informational retrieval reference（记录于 `HANS_OFFICIAL_ANCHORS_V2.informational_raw_file_sha256`），不作为 fail-closed 门。
- 依据：v5 全链 fail-closed 防御的是伪造 manifest 的对抗场景；本项目由作者自行执行实验，科学有效性所需的检查（split 无交集、内容哈希防泄漏、选择完整性、种子确定性）在 v4 中已全部 fail-closed 且被测试覆盖。

### 1.4 监控与证据回传（替代 v1.0 §13.1 的外部监控层与 evidence transport）

- 废止 monitor 状态机（STARTED/terminal/STATUS_CHECK）与 no-weight evidence archive 传输校验作为必过门。
- Smoke 与 canonical 输出目录直接位于挂载的 Google Drive，实现跨会话持久化与 resume。
- `validate_stage2_smoke.py`（含 A100 primary/repeat 比较与 `canonical_v1` 空目录检查）仍为进入 canonical core 前的必过门。
- 失败处理规则不变：单 run 失败即停止、按原配置原 seed 重跑、不自动改参。5 分钟轮询与 12 小时 hard timeout 由 Colab 会话时限与人工检查替代。

### 1.5 执行规模：预声明的 seed 截断规则

- 核心矩阵冻结不变：6 conditions × 5 training seeds，seed-major 顺序 `[42, 123, 2024, 3407, 777]`，每 seed 先共享 Phase-1/2 再按冻结轮转跑 6 条件。
- 出于预算原因，允许先连续完成前 3 个 seeds（42、123、2024，共 3 shared prep + 18 method cells）后暂停。
- 预声明规则（防 optional stopping）：
  1. 完成的 seeds 必须是冻结顺序的连续前缀；不得根据结果挑选 seeds 子集或替换 seed。
  2. 若最终以 n=3 报告，论文如实写明 n=3；Gate A–C 中 "至少 4/5 seeds 为正" 相应替换为 "3/3 seeds 为正"，paired t interval 使用 df=2。
  3. 3-seed 中期检查仅用于确认流水线健康与预算决策；不得据此修改任何配置、超参或数据。
  4. 若 Gate A 通过且预算允许，建议补满 5 seeds 后再定稿主表；投 CCF-B 时补满 5 seeds。

### 1.6 OOD 数据身份：重复源 ID 的确定性消歧（2026-08-08，首次成功 smoke 之前）

- 首次真实数据 manifest 构建发现：WANLI test 集的首选 ID 字段存在重复值，触发 `dataset identity entry contains duplicate full IDs`（Stage 1 报告"明确未做"第 7 条预言的缺口）。
- 修正后的契约：`stable_record_ids` 对重复的 stable ID 按源文件顺序做确定性消歧——首个出现保持原 ID，后续重复者追加 `::dupN`（N 为 1 起的重复序号）。该规则同时覆盖重复源 ID 与完全重复的整行内容（后者会使内容哈希回退同样碰撞）。
- 性质：不改变任何数据集的成员、数量、种子或选择算法；对无重复的数据源（MNLI、HANS、e-SNLI、ANLI、SNLI-hard）产出的 ID 逐字节不变。manifest 与 loader 的 cap 选择使用同一函数，ID 全链一致。
- 防护保留：消歧后仍冲突（原始 ID 恰与 `::dupN` 后缀形式碰撞）继续 fail closed；显式 selected_records 中的重复行经消歧后落在 full membership 之外，同样被拒绝。
- 本修正发生在任何成功 smoke 训练或 canonical 结果产生之前。

## 2. 明确不变项

- 数据成员：100k MNLI（data_seed=42）、5k validation、HANS build/dev/evaluation 划分与全部 v4 锚值。
- 全部模型与训练超参（v1.0 §5）。
- 六条件矩阵、共享 checkpoint 配对规则与轮转顺序（§6）。
- 主要/次要终点、utility constraint、统计分析计划（§7）。
- Gate A–D 判定规则与论文转向规则（§8），仅按 §1.5 调整 seed 计数措辞。
- Official HANS evaluation 不用于任何中间决策（§4.3）。
- 历史 `ties_results/` 仅作 exploratory evidence；`canonical_v1` 结果不覆盖。

## 3. 执行入口

Colab 端到端入口为 `notebooks/canonical_colab.ipynb`：GPU 门 → clone/checkout → 依赖安装 → 完整测试门 → smoke primary + repeat + validator → canonical core（Drive 持久化、跨会话 resume）。
