# Stage 2 Smoke Tests and Canonical Environment Freeze Design

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent` and `superpowers:brainstorming`
- Origin Mode: `plan`
- Origin Date: `2026-08-08`
- Verification Status: `UNVERIFIED`
- Version Label: `stage2-smoke-environment-design-v1`
- Upstream Protocol: `docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md`
- Upstream Report: `docs/paper_rebuild/STAGE1_CANONICAL_INFRASTRUCTURE_REPORT.md`
- Target Worktree: `E:/Learning/LoRA Project/TIES-Unlearning-Diff-LoRA/.worktrees/stage1-canonical-infrastructure`

## 1. Objective

Stage 2 must demonstrate, before any formal canonical run, that the real ML pipeline:

1. imports and passes all tests in a real PyTorch/Transformers/Datasets environment on the local RTX 5080;
2. completes a deterministic tiny-data end-to-end path for Standard LoRA, one shared Phase-1/2 checkpoint, and the `full_sr` and `class_prior_reweight` branches;
3. keeps official HANS evaluation unavailable until final evaluation;
4. restores both dual-adapter branches from the same shared checkpoint path and SHA-256, including class-prior metadata;
5. produces HANS predictions from which every reported aggregate HANS metric can be recomputed exactly;
6. records deterministic dataset identities and checksums for MNLI, HANS, e-SNLI, ANLI, SNLI-hard, and WANLI;
7. completes the same smoke path on a Colab A100 and repeats `full_sr` from a fresh tiny training run in the same runtime, with an absolute difference of at most 0.005 in HANS non-entailment accuracy;
8. records local and A100 software/hardware environments and exercises the frozen monitoring policy; and
9. leaves `ties_results/canonical_v1/` absent or empty and records the commit that is eligible to start Stage 3.

This design does not authorize the 30 formal canonical result cells.

## 2. Invariants and Scope Boundaries

### 2.1 Frozen-core invariants

- `run_canonical.py --stage core` retains the frozen 100,000/5,000 data sizes, five training seeds, six conditions, FP32 precision, epoch counts, hyperparameters, and final evaluation battery.
- Smoke-only controls have safe defaults that are inactive in formal core runs.
- No smoke command accepts `ties_results/canonical_v1` as its output directory.
- Historical `ties_results/` artifacts are read-only exploratory evidence and are never overwritten.
- Official HANS evaluation is not used for checkpoint, layer, epoch, or hyperparameter selection.

### 2.2 Stage-2-only outputs

Stage 2 runtime artifacts live in two ignored sibling trees rooted at
`ties_results/`:

```text
ties_results/stage2_smoke/
  local_rtx5080/
  colab_a100_run1/
  colab_a100_repeat_full_sr/
  freeze_bundle/
ties_results/.stage2_monitor/
  local_rtx5080.events.jsonl
  colab_a100_run1.events.jsonl
  colab_a100_repeat_full_sr.events.jsonl
```

The smoke roots and sibling monitor JSONL evidence are excluded from formal
result aggregation but retained in the Stage 2 evidence archive. They are
runtime evidence, not source-package contents. The freeze bundle is targeted at
`canonical_v1` but remains outside the formal result directory until Stage 3
initialization.

### 2.3 Resolving the manifest/empty-directory wording

The master checklist asks Stage 2 to create canonical protocol, data, and environment manifests, while the protocol also requires the formal `canonical_v1` directory to be empty or absent before Stage 3. Stage 2 therefore creates a checksum-bound `freeze_bundle` outside `canonical_v1`. Stage 3 must copy or validate this bundle while initializing the formal directory. No seed or method result is written to `canonical_v1` during Stage 2.

## 3. Architecture

### 3.1 Smoke profile

A dedicated smoke profile constructs a `TrainConfig` with tiny, explicitly recorded budgets while preserving all algorithmic factors:

- `data_seed=42`, `hans_split_seed=42`, `training_seed=42`;
- `max_seq_length=64`, `batch_size=8`, `mnli_train_size=96`, and `mnli_val_size=96`;
- one epoch for each training phase and four Phase-2 batches;
- `kl_batches=1`, two KL candidates, one selected layer, `knn_k=3`, and Phase-2.5 reference/query counts of 16/8 for MNLI and 8/4 for each HANS label group;
- deterministic smoke-only evaluation caps of 384 HANS rows and 128 rows each for e-SNLI, ANLI, SNLI-hard, and WANLI;
- FP32 and the same RoBERTa/LoRA/TIES factors as canonical v1.

Exact smoke counts are constants in one profile object, appear in every run config, and are covered by tests. They are diagnostics, not canonical evidence.

### 3.2 Separate smoke driver

Add a dedicated `run_stage2_smoke.py` entry point. It runs only:

```text
standard_lora
shared_phase2
full_sr
class_prior_reweight
```

It supports fresh/resume behavior in an explicitly non-canonical output root. The formal `run_canonical.py` CLI remains unchanged.

The orchestration layer may extract a private reusable matrix helper from the existing core runner, but `run_core()` retains its current public defaults and regression tests. Smoke condition selection is not exposed through the formal core CLI.

### 3.3 Deterministic smoke evaluation subsets

Optional evaluation caps are added to `TrainConfig` with `None` defaults. `None` means the complete frozen evaluation set and remains the canonical behavior. The smoke profile sets finite caps.

Sampling uses stable source IDs where available and a content-derived SHA-256 identity otherwise. HANS sampling is stratified so both gold-label groups and all three heuristic families are represented when the requested cap permits it. The selected IDs and their ordered checksums are written to the smoke data manifest.

Official HANS `pairID` values are local to each physical source file. Raw
loading retains the grammar-validated `exN` value solely as the within-file
split/cap ranking key and also creates `hans_train::<pairID>` or
`hans_evaluation::<pairID>` for global artifacts; logical build and dev retain
the same train namespace. The prefix never enters the deterministic cap hash,
so smoke membership remains the frozen raw-key membership. A separate
exact-content identity over gold label, premise, hypothesis, heuristic, and
subcase excludes `pairID` and rejects within-partition duplicates or
cross-partition overlap. `canonical_data_manifest_v3` persists the ordered
content hashes, ordered and joint ID/content checksums, declarations, and
recomputed zero duplicate/overlap counts. Source qualification therefore
resolves local-ID reuse without weakening the leakage gate.

The freeze bundle has a separate canonical-targeted data manifest. It records the complete frozen 100,000-row MNLI training membership, 5,000-row MNLI validation membership, full HANS build/dev/evaluation membership, and full stable or reconstructable identities for e-SNLI, ANLI, SNLI-hard, and WANLI. Smoke-subset checksums never replace these full canonical identities.

### 3.4 Data-access audit trail

Real loader construction emits structured data-access events containing:

```text
dataset
split
purpose
training_phase_or_event
timestamp
```

Training uses HANS build/dev events. Official HANS evaluation emits an event only after a `FINAL_EVALUATION_START` marker. A validator fails closed if an evaluation event appears before that marker or if the marker/event is missing from a completed method.

This instrumentation changes logging only; it does not alter model inputs or numerical decisions.

### 3.5 Artifact validator

A Stage 2 validator independently checks:

- required files and strict finite JSON;
- config seeds and smoke profile identity;
- data counts, IDs, disjointness, and checksums;
- source shared-checkpoint path/hash equality for `full_sr` and `class_prior_reweight`;
- class-prior weights stored in and restored from the shared checkpoint;
- official HANS access ordering;
- exact recomputation of aggregate HANS metrics from `hans_predictions.jsonl`;
- prediction schema, row counts, unique pair IDs, and checkpoint hashes;
- exact ordered equality between every method's HANS prediction IDs and the
  data-manifest evaluation `selected_ids`, including namespace and checksum
  validation;
- environment manifest fields; and
- absence or emptiness of `ties_results/canonical_v1`.

It writes a machine-readable validation record and a concise Markdown summary without editing experiment outputs.

### 3.6 Monitoring layer

A generic subprocess monitor wraps local and A100 smoke commands. Production thresholds are frozen as:

- process/status/log check every 300 seconds;
- `STALL_WARNING` after 3,600 seconds without batch or epoch progress;
- hard timeout after 43,200 seconds;
- immediate failed state for OOM, non-finite loss, download failure, checkpoint mismatch, or prediction-row mismatch;
- no automatic changes to batch size, precision, learning rate, sample count, or any other experiment factor.

Tests and a monitor drill use accelerated wall-clock thresholds while recording that the production profile remains 300/3,600/43,200 seconds. Only the hard-timeout path terminates automatically.

`canonical.stage2_contract` is the dependency-light production authority for
the notebook, primary/repeat roots and conditions, sibling monitor paths, exact
runner outputs, fixed omitted checkpoints, evidence inventory, and extraction
root. Runtime tests import this contract and never depend on the excluded
private plan/spec. The producer and verifier also share
`canonical.monitoring.validate_monitor_jsonl`: successful evidence must be
strict finite JSONL with exact discriminated event fields, timezone-aware and
nondecreasing timestamps, nondecreasing elapsed time, production policy and
initial fingerprints in the first `STARTED`, at least one `STATUS_CHECK`, and a
final zero-return `COMPLETED`. Its normalized command and cwd must bind exactly
to the recorded child command; failure events are forbidden.

Monitor evidence is deliberately outside every child `--fresh --output-dir`
root: use a sibling evidence directory such as
`ties_results/.stage2_monitor/local_rtx5080.events.jsonl`, while `--watch`
continues to name the actual smoke output root. This is a required contract for
the future Task 7 notebook as well: preflight may create the evidence parent,
but must never create or pollute a fresh child output root.

### 3.7 Colab A100 transport

A generated Colab notebook is a thin executor, not a second implementation. It:

1. verifies that the assigned GPU name contains `A100` and stops otherwise;
2. installs the tested dependency set;
3. verifies an external expectations sidecar, then unpacks a source-only bundle whose deterministic parentless `execution_commit` contains exactly allow-listed blobs from the clean `origin_commit`, including the thin notebook but excluding this private plan/spec, runtime artifacts, and origin history;
4. runs unit tests;
5. executes the A100 smoke run and production validator;
6. repeats `full_sr` from a fresh tiny training path in the same runtime and runs its production validator;
7. compares HANS non-entailment accuracy using the frozen absolute 0.005 tolerance;
8. captures `pip freeze`, Python, PyTorch, Transformers, Datasets, NumPy, CUDA runtime/driver, GPU, command, timestamps, and logs; and
9. reruns both validators and freeze verification, then exports a no-weight evidence ZIP with an exact v2 inventory.

The local Task 8 run and Colab Task 9 run clone the same archive, configure
checkout newline conversion off, and execute the identical recorded
`execution_commit`; Task 9 must not repackage from the execution clone. The
external sidecar remains outside the source ZIP and is preserved in the final
evidence inventory.

The evidence ZIP is independently transport-verifiable without model weights.
Its inventory lists exact transported `files` plus exactly the primary and
repeat `seed_42/shared_phase2/checkpoints/shared.pt` omissions. Each omitted
path/hash/reason is bound to status, shared checkpoint JSON, checkpoint
metadata, the shared manifest, and every applicable branch manifest. Before any
user extraction, a local verifier safely reconstructs a fresh temporary tree,
reruns freeze semantic verification, and reuses the production validator in a
tightly scoped exact omitted-weight mode. That mode reruns all non-weight
config/seed/manifest/audit/HANS aggregation/class-prior/checkpoint/repeat
semantics and compares the independently recomputed results with both stored
JSON and Markdown reports. It also validates both producer-owned monitor JSONL
records against exact primary/repeat argv and cwd. Only then may it extract safe
`ties_results/` paths without overwriting. Ordinary production validation is
not claimed after extraction because it correctly requires the omitted payloads.

Only source code and experiment artifacts are transferred. No manuscript, private notes, credentials, or historical result corpus is uploaded.

## 4. Environment Strategy

### 4.1 Local RTX 5080

Create a repository-local isolated environment rather than modifying the MSYS2 Python 3.14 installation. Select a supported Python/PyTorch/CUDA wheel combination that detects the RTX 5080 and record the resolver output. The full test suite must run under this exact interpreter before the local smoke run.

### 4.2 Canonical A100 freeze

The A100 runtime is the canonical training environment. The freeze bundle records both the primary and fresh-repeat immutable command records, and their commit chain must equal the clean source-package `execution_commit`; the separate clean `origin_commit` remains frozen as provenance. The freeze bundle records:

- exact Python and package versions;
- PyTorch CUDA runtime and NVIDIA driver;
- A100 model string;
- full `pip freeze`;
- OS/platform metadata;
- protocol, data-manifest, source-archive, and commit checksums; and
- commands used for both A100 runs.

Stage 3 must reconstruct or validate this environment before starting formal jobs. If the A100 GPU model or incompatible runtime changes, execution stops and the protocol addendum rule applies.

## 5. Error Handling

- Dependency installation, dataset/model download, import, test, smoke, or validation failure stops the current step; there is no silent retry or automatic parameter adjustment.
- A missing A100 allocation stops Colab execution rather than falling back to another GPU.
- A dirty Git tree is allowed only for pre-run development. The approved checklist wording change is included in the final Stage 2 documentation commit; the verified smoke archive and freeze bundle must bind to a recorded clean commit with no exceptions.
- Any numerical-behavior fix triggers targeted Stage 1 regression tests plus the affected Stage 2 smoke rerun and an explicit protocol-version assessment.
- Logging-only, report-only, or display-only corrections are recorded but do not change protocol v1.0.

## 6. Test Strategy

Implementation follows red-green TDD. Required automated coverage includes:

1. smoke profile isolation and unchanged canonical defaults;
2. rejection of `canonical_v1` as a smoke output path;
3. exact smoke condition set and one shared checkpoint;
4. deterministic evaluation IDs and stratified HANS sample membership;
5. official HANS access-order validation, including a failing intermediate-access fixture;
6. shared checkpoint path/hash and class-prior restoration validation;
7. independent prediction-to-aggregate recomputation;
8. strict environment/freeze-bundle schema;
9. monitor normal completion, advisory stall, crash, and hard-timeout paths; and
10. A100 repeat-comparison boundary cases at, below, and above 0.005.

After unit tests pass, Stage 2 runs the real local and A100 workflows. Mock-backed tests alone cannot satisfy the stage gate.

## 7. Deliverables

- focused implementation plan in `docs/superpowers/plans/`;
- smoke driver, validator, monitoring layer, and Colab executor;
- automated tests and real command logs;
- local and A100 smoke artifact trees;
- canonical-targeted freeze bundle with checksum inventory;
- updated Stage 2 checklist only for items supported by evidence; and
- `docs/paper_rebuild/STAGE2_SMOKE_ENVIRONMENT_FREEZE_REPORT.md` with commands, versions, results, anomalies, evidence paths, unresolved limits, and the Stage 3 handoff gate.

## 8. Completion Gate

Stage 2 is complete only when every master-checklist item and every Stage 1 handoff item has direct evidence, both A100 runs use the same runtime and pass the 0.005 primary-accuracy tolerance, all validators pass, the formal `canonical_v1` result directory is absent or empty, and a clean canonical-eligible commit is recorded.

If any item cannot be completed, the report status is `PARTIAL` or `BLOCKED`; unchecked checklist items remain unchecked, and Stage 3 is not authorized.
