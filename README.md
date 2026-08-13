# Diff-LoRA: A Controlled Attribution Study of Rank-Differential LoRA Subtraction

This repository contains the method, the experimental protocol, and the canonical results behind the paper
**"Subtraction or Threshold Shift? A Controlled Multi-Seed Attribution Study of LoRA Arithmetic for Shortcut Mitigation in NLI."**

The code implements *TIES-Unlearning Diff-LoRA*: a dual-path LoRA pipeline that trains a high-rank task branch and a low-rank "shortcut" branch, then subtracts the latter from the former under a sign-consensual mask, with the aim of reducing reliance on the syntactic heuristics that HANS is built to expose.

> **What this repository reports.** We built this method and, in an early single-seed run, measured HANS non-entailment accuracy rising from 20.21% to 30.46% at essentially unchanged MNLI accuracy. This repository is the audit of that result. Under a controlled six-condition matrix run at five training seeds, the claim does not hold: the subtraction operator contributes nothing in isolation and is *harmful* in combination, and the movement that remains is equivalent to a decision-threshold shift. The code is released so the analysis can be reproduced and applied to other methods in the same family. See [Findings](#findings).

---

## Findings

All numbers below come from `canonical_v1`: 6 conditions × 5 training seeds = 30 runs, all completed, no run excluded.

| Contrast | What it isolates | Δ HANS non-ent. | 95% CI | Positive seeds |
|---|---|---|---|---|
| `staged_neither` − `standard_lora` | the staged pipeline itself | +1.55 pp | [−1.87, +4.96] | 4/5 |
| `subtraction_only` − `staged_neither` | subtraction, alone | +0.23 pp | [−3.49, +3.94] | 3/5 |
| `full_sr` − `reweight_only` | subtraction, added to reweighting | **−7.95 pp** | [−18.70, +2.80] | **0/5** (sign test *p* = 0.031) |
| `reweight_only` − `staged_neither` | N-guided reweighting | +25.60 pp | [−4.31, +55.51] | 4/5 |

1. **Subtraction is not the source of any improvement.** Alone it is indistinguishable from zero; added to reweighting it lowers the primary endpoint in every one of the five seeds.
2. **What movement exists is a threshold shift.** Re-thresholding the unmodified baseline to predict entailment at the same rate reproduces — and slightly exceeds — the method's non-entailment accuracy (mean residual −0.60 pp). Nothing transfers to e-SNLI, ANLI, SNLI-hard or WANLI.
3. **The Phase-2 objective is degenerate.** Trained on a mixture that is 90% HANS-entailment, a constant "always entailment" predictor already scores ≈90%. The low-rank branch reaches that constant solution in all five seeds, scoring exactly 0.500 on each of the three heuristics — it learned a label, not a heuristic. Its *confidence* is unconstrained and effectively drawn by the seed, which is why the downstream weighting rule is algebraically inert on two seeds and reduces to class rebalancing on three.
4. **Single-seed evaluation of this pipeline is unreliable.** Identical configuration, data and hyperparameters yield HANS non-entailment accuracy anywhere from 17.5% to 56.0%.

---

## The six-condition matrix

Within each seed, five conditions load a **byte-identical Phase-1/2 checkpoint** (verified by SHA-256) and diverge only at Phase 3, so any difference between them is attributable to the Phase-3 component rather than to upstream sampling noise.

| Condition | Shared checkpoint | Subtraction | Phase-3 weighting |
|---|---|---|---|
| `standard_lora` | no (independent baseline) | — | none |
| `staged_neither` | yes | off | none |
| `subtraction_only` | yes | **on** | none |
| `reweight_only` | yes | off | **N-guided** |
| `full_sr` (complete method) | yes | **on** | **N-guided** |
| `class_prior_reweight` | yes | off | class-prior |

`class_prior_reweight` replaces the per-example signal with its class-conditional average, keeping the label-level effect while discarding all per-example information. It is the control that separates *"the method identifies shortcut-reliant examples"* from *"the method rebalances label priors."* On three of five seeds it collapses to predicting non-entailment for every HANS example.

---

## Method

Each targeted linear layer of a frozen backbone carries two LoRA branches of asymmetric capacity, ΔP = B_P A_P with rank r_P = 16 and ΔN = B_N A_N with rank r_N = 4. The effective update is

```
ΔW_eff = α·ΔP − β·(M_trim ⊙ M_sign) ⊙ ΔN
```

where `M_trim` keeps the top 20% of |ΔN| by global magnitude and `M_sign` keeps the coordinates on which ΔP and ΔN agree in sign. Note that LoRA scaling `s = α_lora / r` gives `s_N = 4·s_P`, so the low-rank branch enters the merge amplified fourfold.

**Phase 1** trains the head and P on MNLI with N frozen.
**Phase 2** freezes the head and P, and trains N on 10% MNLI + 90% HANS-entailment drawn from a held-out HANS build split.
**Phase 2.5** ranks encoder layers by a hybrid of layer-wise KL divergence between P-only and N-only predictions and kNN probing of CLS representations (prediction depth, early-wrong).
**Phase 3** activates the merge on the selected layers and fine-tunes the head and P with N frozen, applying N-guided reweighting `w(x) ∝ (1 − p_N(y|x))^γ`.

Backbone RoBERTa-base; LoRA on the query and value projections of every encoder layer (24 injected modules); FP32 throughout.

---

## Experimental protocol

The protocol was frozen **before any canonical run was executed** and was not revised afterwards. Full text: [`docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md`](docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md), with [addendum v1.1](docs/paper_rebuild/PROTOCOL_ADDENDUM_V1_1.md).

- **Data.** 100,000 MNLI training and 5,000 validation examples at `data_seed = 42`, identical across every condition and seed.
- **HANS partitioning.** Deterministic build / dev / official-evaluation split. Phase 2 draws **only** from the build split; the official evaluation set shares no example with it, and disjointness plus content integrity are enforced by hashed manifests.
- **Seeds.** `[42, 123, 2024, 3407, 777]`, fixed in advance. Every condition runs at every seed.
- **Primary endpoint.** HANS non-entailment accuracy. Utility constraint: MNLI must not fall more than 0.5 pp below `standard_lora`.
- **Evaluation hygiene.** The official HANS evaluation set is never read during Phase 1, 2, 2.5 or intermediate Phase-3 checkpoint selection. Access is logged and validated.
- **Pre-registered gates.** Four decision gates, each with its paper-level consequence, were specified in advance so that the narrative could not be chosen after seeing the numbers. All four failed; the paper reports that outcome.

Statistics: seed-wise paired differences, 95% CIs from the *t* distribution at df = 4, plus pre-specified one-sided exact sign and Wilcoxon signed-rank tests (minimum attainable *p* = 2⁻⁵ = 0.031).

---

## Reproducing the canonical results

### Requirements

Python 3.10+, one A100-class GPU. Approximately 3 hours per seed (shared Phase-1/2 preparation plus six condition branches), about 15–17 GPU-hours for the full matrix.

```bash
pip install -r requirements.txt
```

Versions are pinned in the Colab notebook (`transformers==5.14.1`, `datasets==5.0.1`); older `datasets` releases import `torchvision.io.VideoReader` without a guard and crash on the first training batch when torchvision is present.

### Run the matrix

```bash
python run_canonical.py \
  --stage core \
  --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md \
  --output-dir /path/to/canonical_v1 \
  --fresh
```

Execution is seed-major with a protocol-frozen rotation of condition order within each seed. Runs are resumable: drop `--fresh` to continue, and completed runs are skipped. Every run records its configuration, resolved environment, Git commit, terminal status, metrics, per-example HANS predictions and logs, so all aggregate metrics are recomputable from the prediction records.

`run_canonical.py` refuses to start from a dirty Git working tree.

### Colab

[`notebooks/canonical_colab.ipynb`](notebooks/canonical_colab.ipynb) runs the whole thing end to end: GPU gate → clone at a fixed commit → dependencies → full test gate → smoke tests → canonical matrix, with results persisted to Google Drive and resumable across sessions.

### Smoke tests and environment freeze

```bash
python run_stage2_smoke.py --mode primary --environment colab_a100 \
  --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md \
  --output-dir /path/to/smoke --fresh

python validate_stage2_smoke.py --root /path/to/smoke \
  --conditions standard_lora full_sr class_prior_reweight \
  --compare-repeat /path/to/smoke_repeat \
  --canonical-dir /path/to/canonical_v1
```

The validator checks metric recomputability from predictions, checkpoint pairing, evaluation-access hygiene, and that the formal result directory is still empty before the canonical matrix starts.

### Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

184 tests; one heavy packaging integration test is skipped by default (set `STAGE2_RUN_PACKAGING_INTEGRATION=1` to include it).

---

## Repository layout

```
canonical/       Canonical experiment machinery: condition matrix, runner, data
                 identity and integrity manifests, HANS metric recomputation,
                 Stage-2 validation, environment freeze
configs/         TrainConfig / LoRAConfig — all hyperparameters
data/            Dataset loading, HANS partitioning, Phase-2 mixture construction
models/          ties_lora.py (dual-path LoRA), surgery.py, analyzer.py (Phase 2.5)
training/        Phase 1/2/2.5/3 trainer, baselines, weighting rules
tests/           184 tests covering the above
docs/paper_rebuild/   Frozen protocol, addenda, stage reports
notebooks/       Colab runners
ties_results/    Exploratory results predating the canonical protocol — see note
run_canonical.py         Canonical six-condition matrix
run_stage2_smoke.py      Pre-flight smoke matrix
validate_stage2_smoke.py Strict artifact validation
main.py                  Single-run entry point for the method
```

> **`ties_results/` is exploratory.** Those directories predate the frozen protocol and were produced under single-seed, pre-registration conditions. They are kept for transparency about how the project developed. **They are not the canonical results and no claim in the paper rests on them.**

---

## Applying this design to other methods

The attribution design is not specific to rank-differential subtraction. Reusing it requires three things and no new machinery:

1. a **staged control** that runs the full pipeline with every intervention disabled, separating the operator from the training procedure around it;
2. **shared-checkpoint pairing**, so that all conditions branch from one byte-identical checkpoint per seed;
3. a **signal-stripped control** that preserves the intervention's coarse effect while removing the information it claims to use.

Two diagnostics are worth adding for free: report the proxy's per-class confidence and per-heuristic accuracy after the concentration stage, and report the **threshold-matched residual** alongside any challenge-set gain — on a binary challenge set derived from a three-way task, accuracy on one class can always be bought with accuracy on the other.

---

## Citation

```bibtex
@misc{diff-lora-attribution,
  title  = {Subtraction or Threshold Shift? A Controlled Multi-Seed Attribution
            Study of LoRA Arithmetic for Shortcut Mitigation in NLI},
  author = {Chen, Nuo and Huang, Sicheng and Shi, Dayou and Ye, Bohou and Yuan, Weitu},
  year   = {2026},
  note   = {All authors contributed equally and should be considered co-first
            authors; author order is alphabetical.},
  url    = {https://github.com/LeafTraces/Diff-LoRA}
}
```

## License

See [LICENSE](LICENSE).