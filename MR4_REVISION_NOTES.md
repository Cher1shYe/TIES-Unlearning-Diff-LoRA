# MR.4 修改说明

本文档给项目同学快速说明本次 MR.4 相关代码、结果和论文表述边界。MR.4 的审稿要求是验证 rank-differential 假设，补充 rank sweep、equal-rank control、reversed-rank control 和 branch-only evaluation。

## 这次改了什么

- `run_sensitivity.py`
  - 新增 `RunSpec`，把普通 one-at-a-time sensitivity run 和多字段 rank-control run 统一管理。
  - 新增 `rank_controls` 运行组，包含默认 rank-differential、两个 equal-rank control，以及一个 reversed-rank control。
  - 新增 `--only rank_controls`，可以只运行 MR.4 控制实验。
  - 新增 `--skip-rank-controls`，可以复现原来的 sensitivity grid，不额外跑 MR.4 控制项。
  - `sensitivity_summary.json` 会把默认 anchor 展开为 `default_differential`，并写出 `rank_control_summary.md`。

- `configs/config.py`
  - 新增 `record_branch_only_metrics`，默认关闭。
  - MR.4 rank-control run 会打开它，用于训练结束后额外记录 P-only 和 N-only 指标。

- `training/trainer.py`
  - 当 `record_branch_only_metrics=True` 时，Phase 3 后会分别切到 `phase1` 和 `phase2` forward mode，记录 final P-only 与 N-only evaluation。
  - 指标写入 `metrics.json` 的 `branch_only.p_only` 和 `branch_only.n_only`。

- `finish_sensitivity.py`
  - 支持 `--only rank_controls` 和 `--skip-rank-controls`。
  - 可以从已有 `metrics.json` 重新汇总 MR.4 控制实验，不需要重复训练。

- `plot_mr4_rank_controls.py`
  - 从 `ties_results/mr4_rank_controls_small` 读取结果。
  - 输出 Supplementary Figure 风格的 SVG/PNG，以及图源 CSV。

- `tests/test_run_sensitivity.py`
  - 轻量测试 MR.4 run list、rank-control-only 参数解析，以及 branch-only 指标展开逻辑。

## 结果放在哪里

MR.4 小预算控制实验已整理到：

- `ties_results/mr4_rank_controls_small/sensitivity_summary.json`
- `ties_results/mr4_rank_controls_small/rank_control_summary.md`
- `ties_results/mr4_rank_controls_small/figure/mr4_rank_control_figure.svg`
- `ties_results/mr4_rank_controls_small/figure/mr4_rank_control_figure.png`
- `ties_results/mr4_rank_controls_small/figure/mr4_rank_control_figure_source_metrics.csv`
- `ties_results/mr4_rank_controls_small/figure/mr4_rank_control_figure_source_layers.csv`

这些结果来自 reduced-budget setting，不能直接替代 full-budget 主实验表格。论文里建议放到 Supplementary Table/Figure，或者明确写作 "reduced-budget rank-control diagnostics"。

## 当前 MR.4 结果怎么解读

小预算设置下，default rank-differential control 是 `r_P=16, r_N=4`。它的 merged MNLI accuracy 最高：

| Control | r_P | r_N | Merged MNLI | Merged HANS non-ent | P-only MNLI | N-only MNLI | N-only HANS non-ent |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default differential | 16 | 4 | 82.45% | 7.33% | 82.05% | 34.80% | 0.00% |
| Equal low | 4 | 4 | 80.10% | 20.95% | 80.75% | 68.95% | 0.00% |
| Equal high | 16 | 16 | 80.95% | 10.83% | 81.40% | 68.35% | 0.00% |
| Reversed | 4 | 16 | 81.75% | 4.60% | 80.60% | 32.75% | 0.00% |

建议论文结论写得谨慎一些：

- 可以说 default rank-differential setting 在这个 reduced-budget 控制组里最能保留 MNLI utility。
- 可以说 N-only branch 在所有 rank-control setting 里 HANS non-entailment 都是 0.00%，说明它强烈偏向 entailment-style shortcut behavior。
- 不要说 rank asymmetry 单调提升 HANS non-entailment，因为 equal-low 在这个小预算结果里 HANS non-entailment 更高。
- 不要说 negative branch 是纯 shortcut subspace，因为 equal-rank N branch 仍保留了较高 MNLI accuracy。
- 推荐把主张改成 partial shortcut mitigation / utility-preserving shortcut mitigation，而不是 complete shortcut removal。

## 复现实验和作图

只跑 MR.4 控制实验：

```bash
python run_sensitivity.py --small --only rank_controls --output-dir ./ties_results/mr4_rank_controls_small
```

如果已有 per-run `metrics.json`，只重新汇总：

```bash
python finish_sensitivity.py --small --assemble-only --only rank_controls --output-dir ./ties_results/mr4_rank_controls_small
```

重画 MR.4 图：

```bash
python plot_mr4_rank_controls.py --results-dir ./ties_results/mr4_rank_controls_small
```

运行轻量测试：

```bash
python -m unittest tests.test_run_sensitivity
```

## 写进论文时的建议

- 方法部分：把 `r_P=16, r_N=4` 说成 inductive bias，不要说成已经证明的 shortcut/reasoning 分离。
- 实验部分：新增 rank-control protocol，说明 equal-rank、reversed-rank 和 branch-only evaluation。
- 结果部分：说明这些是 reduced-budget diagnostics，支持 utility preservation，但也暴露 HANS robustness 对 rank allocation 敏感。
- 讨论/限制：明确 rank asymmetry is useful but not sufficient evidence for complete shortcut unlearning。
