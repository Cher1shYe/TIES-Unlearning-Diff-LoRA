# Stage 1 Canonical Infrastructure Design

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-07
- Verification Status: UNVERIFIED
- Version Label: stage1_canonical_infrastructure_design_v1
- Upstream Protocol: `docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md`
- Upstream Checklist: `docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md`
- Target Branch: `codex/stage1-canonical-infrastructure`

## Objective

Implement only the infrastructure required by checklist Stage 1 so the frozen `canonical_v1` experiment can be smoke-tested in Stage 2. Preserve the existing dual-LoRA architecture, TIES merge, Phase 1/2/2.5/3 algorithm, historical result directories, and legacy experiment drivers.

Stage 1 does not run the 30 canonical method-seed cells, change paper conclusions, add a second model, or perform GPU smoke tests.

## Selected Approach

Add a small `canonical/` package containing protocol-specific condition, artifact, checkpoint, and orchestration logic. Make narrow adaptations to the existing configuration, data, training, and evaluation modules so they expose the capabilities the canonical layer needs.

This is preferred over putting all behavior in `training/trainer.py` because seed selection, HANS splitting, artifact validation, and run orchestration can be tested without importing or executing the full ML stack. A full training-pipeline rewrite is explicitly out of scope.

## Architecture

### 1. Configuration and condition contract

`TrainConfig` will expose three independent seeds:

- `data_seed=42`: MNLI subset selection and other fixed data/analysis samples.
- `hans_split_seed=42`: deterministic HANS-train build/dev partitioning.
- `training_seed=42`: model/LoRA initialization, dropout, shuffling, and random-layer controls.

Canonical training seeds are frozen as `(42, 123, 2024, 3407, 777)`. Existing callers that assign `cfg.seed` will remain compatible by treating it as an alias for `training_seed`; all data-selection call sites will be migrated to `data_seed` so changing the alias cannot change sampled row IDs.

The canonical condition table will contain exactly:

| Tag | Subtraction | Phase-3 weighting |
|---|---:|---|
| `standard_lora` | no | `none` |
| `full_sr` | yes | `n_guided` |
| `subtraction_only` | yes | `none` |
| `reweight_only` | no | `n_guided` |
| `staged_neither` | no | `none` |
| `class_prior_reweight` | no | `class_prior` |

For dual-adapter conditions, absence of subtraction means P-only Phase-3 merging, not absence of the shared N-adapter checkpoint. The class-prior condition uses class means calculated from the fixed MNLI training set only:

```text
r_i = (1 - p_N(y_i | x_i))^2
a_c = mean(r_i | y_i = c)
```

Each batch uses `a_{y_i}` and normalizes weights to mean 1.

The frozen trim contract keeps exactly `max(1, floor(trim_ratio × numel))` elements. Elements are ordered by descending absolute N-delta magnitude; equal magnitudes are resolved by ascending flattened tensor index. This prevents a threshold tie from retaining more than the declared 20%.

### 2. Data contract

HANS-train will be partitioned by `gold_label × heuristic × subcase`. Within each stratum, records are first sorted by stable `pairID`, then a fresh NumPy `default_rng(hans_split_seed)` generates that stratum's permutation. The first `floor(0.20 × n)` records enter dev; the rest enter build. Strata smaller than five records remain entirely in build and are recorded in the split manifest.

The manifest will contain build/dev pair IDs, small-stratum notices, counts, the split seed, and a SHA-256 checksum derived from a canonical UTF-8 serialization of the split membership. Build, dev, and official evaluation pair-ID sets must be disjoint before a canonical run can proceed.

MNLI sampling will retain stable source IDs before tokenization so data manifests can prove that all training seeds use the same sampled rows.

### 3. Evaluation isolation and prediction contract

The canonical training path will not construct or evaluate the official HANS evaluation loader during Phase 1, Phase 2, Phase 2.5, or Phase-3 epochs. Intermediate implementation diagnostics may use HANS-train dev. Official HANS evaluation is constructed only after the method's frozen training branch finishes.

Final HANS evaluation will emit UTF-8 JSONL with one record per example and these required fields:

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

Aggregate overall, entailment, non-entailment, heuristic, and subcase accuracies will be computed from those prediction records through one shared aggregation function. The saved metrics must equal a fresh recomputation from the JSONL data.

### 4. Shared checkpoint contract

For each training seed, one preparation job trains Phase 1 and Phase 2 once and writes a shared checkpoint. The checkpoint stores model state, completed-phase metadata, config, class-prior weights, and its SHA-256 checksum. The five dual-adapter branches load model state from this checkpoint without replaying Phase 1 or Phase 2.

Every dual-adapter run manifest records the same shared checkpoint path and hash for its seed. A hash mismatch is fatal and prevents later branches for that seed from starting. `standard_lora` remains an independent compute-matched training job that uses the same data IDs and training seed.

### 5. Artifact and status contract

All machine-readable output will use a strict JSON writer with `allow_nan=False`. Missing values are represented by `null`; non-finite values cause an explicit failure before an artifact is committed.

Each canonical method directory will support:

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

`run_manifest.json` records the full command, Git commit and dirty state, protocol hash, environment, seed fields, method tag, timestamps, shared checkpoint provenance, and output schema version. `status.json` moves through `pending`, `running`, `success`, or `failed`; failed runs record the error and are never aggregated as missing numeric values.

### 6. Canonical driver

`run_canonical.py` will provide the frozen entry point:

```powershell
python run_canonical.py `
  --stage core `
  --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md `
  --output-dir ties_results/canonical_v1
```

The driver will:

1. Validate the protocol hash, Git cleanliness, output policy, data manifest, and environment manifest.
2. Use the frozen five training seeds and six conditions.
3. Prepare one shared Phase-1/2 checkpoint per seed.
4. Run conditions in the frozen, seed-rotated order.
5. Resume by skipping only runs whose status and recorded checksums are complete and valid.
6. Stop the current seed after any failed preparation, branch, checkpoint hash, non-finite output, or prediction-integrity check.
7. Treat `--fresh` as permission to use a new empty directory only; it never deletes or overwrites an existing canonical directory.

Process monitoring and GPU hard-timeout execution are Stage 2 concerns. Stage 1 supplies the status and timestamp fields Stage 2 will monitor.

## File Boundaries

Expected new files:

- `canonical/__init__.py`: public canonical constants and types.
- `canonical/conditions.py`: six-condition table and seed-rotated order.
- `canonical/artifacts.py`: strict JSON/JSONL, hashes, manifests, status, and environment metadata.
- `canonical/runner.py`: preflight, resume checks, shared-checkpoint/branch orchestration.
- `run_canonical.py`: command-line entry point.
- Focused tests under `tests/` for seeds, split, conditions, artifacts, predictions, checkpoint sharing, driver isolation, and trim semantics.

Expected narrow modifications:

- `configs/config.py`: independent seeds and Phase-3 weighting mode.
- `data/dataloader.py`: fixed data seed, HANS build/dev/evaluation loaders, stable IDs.
- `training/evaluate.py`: prediction records and aggregate recomputation.
- `training/trainer.py`: shared Phase-2 stop/load hooks, weighting modes, delayed official evaluation.
- `training/baseline.py` and legacy drivers: training-seed migration without changing historical output locations.
- `utils/optim_utils.py`: shared checkpoint metadata and model-state loading support.
- `models/ties_lora.py`: no behavioral change unless a test reveals the frozen 20% trim contract is violated.
- `docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md`: Stage 0/1 status only after all gates pass.
- `docs/paper_rebuild/STAGE1_CANONICAL_INFRASTRUCTURE_REPORT.md`: final handoff report.

Files in `ties_results/` are read-only for Stage 1.

## Error Handling

- Invalid seed, condition, weighting mode, or non-empty `--fresh` output directory: fail before training.
- HANS overlap, missing pair ID, duplicate pair ID, or split checksum mismatch: fail preflight.
- Shared checkpoint missing or hash mismatch: mark failed and stop the seed.
- Non-finite loss or serialized value: mark failed; do not substitute `NaN`.
- Prediction count, ID, label, or metric-recomputation mismatch: mark failed and do not mark the run successful.
- Training exception: record failure once; do not auto-retry or change configuration.

## Testing Strategy

Use red-green TDD for each behavior. The current shell lacks PyTorch, Transformers, Datasets, NumPy, and pytest, so the Stage 1 test suite will use `unittest`, pure canonical modules, dependency injection, and narrowly scoped module stubs where importing a legacy ML module is unavoidable. This validates orchestration contracts without downloading models or datasets. Tests that require the actual ML stack and GPU are reserved for Stage 2 smoke testing.

Required Stage 1 behaviors:

1. Changing `training_seed` does not change fixed MNLI sample IDs.
2. HANS build/dev/evaluation IDs are disjoint and the split is deterministic.
3. The six conditions differ only in declared factors.
4. Five dual-adapter branches for a seed receive the same Phase-2 checkpoint hash.
5. `trim_ratio=0.2` retains the largest-magnitude 20% of N-delta elements, including a deterministic tie policy.
6. Strict JSON and JSONL reject all non-finite floats recursively.
7. HANS aggregates are exactly reproducible from prediction records.
8. Canonical Phase 1/2/2.5 and Phase-3 epoch hooks cannot request official HANS evaluation.
9. Resume skips only complete, checksum-valid runs.
10. Existing legacy unit tests remain green.

The implementation report will distinguish lightweight unit verification in Stage 1 from real-dependency/GPU smoke verification pending in Stage 2.

## Acceptance Criteria

Stage 1 is complete only when:

- Every checklist Stage 1 capability is implemented.
- Every new test has been observed failing for the intended missing behavior and then passing after the minimal implementation.
- All legacy and new tests pass in the available local test environment.
- No file under historical `ties_results/` changed.
- The implementation code, tests, checklist update, and handoff report include an independently revertible canonical-infrastructure commit; the approved design specification may remain in its preceding documentation commit.
- The Stage 1 branch working tree is clean.

## Explicit Non-Goals

- No formal 30-cell canonical execution.
- No RTX 5080 or A100 smoke run.
- No environment freeze claim beyond collecting metadata support.
- No statistical Gate A-D analysis.
- No paper conclusion or title change.
- No new baseline, second model, rank grid, or sensitivity sweep.
