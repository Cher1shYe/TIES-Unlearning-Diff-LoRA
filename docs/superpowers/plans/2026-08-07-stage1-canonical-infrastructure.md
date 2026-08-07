# Stage 1 Canonical Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the minimum tested infrastructure required to smoke-test and later execute the frozen `canonical_v1` six-condition, five-seed experiment without rewriting the existing model architecture.

**Architecture:** A lightweight `canonical/` package owns pure protocol contracts, deterministic data splitting, strict artifacts, HANS aggregation, and orchestration. Existing configuration, data, training, evaluation, checkpoint, and baseline modules receive narrow hooks for independent seeds, shared Phase-1/2 checkpoints, three Phase-3 weighting modes, and delayed official HANS evaluation.

**Tech Stack:** Python 3, `unittest`, PyTorch, Transformers, Hugging Face Datasets, NumPy, SHA-256, JSON/JSONL.

## Global Constraints

- Implement only checklist Stage 1; do not run the 30 canonical result cells.
- Preserve the existing dual-LoRA injection, TIES-style merge, Phase 1/2/2.5/3 architecture, and historical `ties_results/` files.
- Canonical MNLI data sizes remain 100,000 train and 5,000 validation examples with `data_seed=42`.
- HANS splitting uses `hans_split_seed=42` and `gold_label × heuristic × subcase` strata.
- Training seeds are exactly `[42, 123, 2024, 3407, 777]`.
- Official HANS evaluation is never constructed or read during Phase 1, Phase 2, Phase 2.5, or Phase-3 epochs.
- JSON uses `allow_nan=False`; missing values are `null`, never `NaN` or infinity.
- Real-dependency/GPU smoke execution belongs to checklist Stage 2.
- Use red-green TDD and run the full available `unittest` suite after every task.

---

## File Map

### New production files

- `canonical/__init__.py`: exported seeds, condition tags, schema version.
- `canonical/conditions.py`: immutable six-condition matrix and rotated order.
- `canonical/data.py`: duck-typed fixed sampling, HANS stratified split, disjointness, checksums, manifests.
- `canonical/hans.py`: prediction schema validation and aggregate metric recomputation.
- `canonical/artifacts.py`: recursive finite validation, strict atomic JSON/JSONL, SHA-256, Git/environment/status metadata.
- `canonical/evaluation_policy.py`: explicit dev-versus-final HANS access gate.
- `canonical/backend.py`: lazy adapter from the pure runner to existing training functions.
- `canonical/runner.py`: preflight, shared checkpoint orchestration, resume, fixed run order, failure handling.
- `training/weighting.py`: pure class-prior calculation plus PyTorch batch-weight adapters.
- `run_canonical.py`: frozen command-line entry point.

### New tests

- `tests/test_canonical_conditions.py`
- `tests/test_canonical_data.py`
- `tests/test_canonical_artifacts.py`
- `tests/test_canonical_hans.py`
- `tests/test_ties_trim_contract.py`
- `tests/test_canonical_weighting.py`
- `tests/test_canonical_runner.py`

### Modified production files

- `configs/config.py`: seed split and explicit Phase-3 weighting mode.
- `data/dataloader.py`: data-seed use, HANS build/dev/evaluation loaders, stable metadata.
- `models/ties_lora.py`: exact top-fraction trim mask with stable ties.
- `training/evaluate.py`: optional per-example HANS prediction collection.
- `training/trainer.py`: shared preparation/branch hooks, delayed final loaders, weighting dispatch.
- `training/baseline.py`: training-seed use, delayed final HANS loader, strict artifacts.
- `training/train_jtt.py`, `training/train_poe.py`, `training/train_zfilter.py`: training-seed migration only.
- `utils/optim_utils.py`: checkpoint metadata and model-state-only load.
- `run_multiseed.py`: assign `training_seed`, not a data seed.

### Stage handoff files

- `docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md`: mark Stage 0 confirmation and verified Stage 1 items.
- `docs/paper_rebuild/STAGE1_CANONICAL_INFRASTRUCTURE_REPORT.md`: evidence-backed handoff report.

---

### Task 1: Freeze the seed and six-condition contracts

**Files:**
- Create: `canonical/__init__.py`
- Create: `canonical/conditions.py`
- Modify: `configs/config.py`
- Modify: `run_multiseed.py`
- Test: `tests/test_canonical_conditions.py`

**Interfaces:**
- Produces: `CANONICAL_TRAINING_SEEDS: tuple[int, ...]`, `CONDITIONS: Mapping[str, CanonicalCondition]`, `rotated_condition_order(training_seed: int) -> tuple[str, ...]`.
- Produces: `TrainConfig.data_seed`, `TrainConfig.hans_split_seed`, `TrainConfig.training_seed`, `TrainConfig.phase3_weighting`.
- Compatibility: `cfg.seed` getter/setter maps only to `training_seed`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_legacy_seed_alias_changes_training_seed_only(self):
    cfg = TrainConfig()
    cfg.seed = 123
    self.assertEqual(42, cfg.data_seed)
    self.assertEqual(42, cfg.hans_split_seed)
    self.assertEqual(123, cfg.training_seed)

def test_conditions_match_frozen_factor_matrix(self):
    got = {tag: (c.subtraction, c.weighting) for tag, c in CONDITIONS.items()}
    self.assertEqual({
        "standard_lora": (False, "none"),
        "full_sr": (True, "n_guided"),
        "subtraction_only": (True, "none"),
        "reweight_only": (False, "n_guided"),
        "staged_neither": (False, "none"),
        "class_prior_reweight": (False, "class_prior"),
    }, got)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_canonical_conditions -v`

Expected: import/attribute failures because `canonical.conditions` and the independent seed fields do not exist.

- [ ] **Step 3: Implement the minimal immutable contracts**

```python
@dataclass(frozen=True)
class CanonicalCondition:
    tag: str
    subtraction: bool
    weighting: Literal["none", "n_guided", "class_prior"]
    standard_lora: bool = False

CANONICAL_TRAINING_SEEDS = (42, 123, 2024, 3407, 777)
BASE_CONDITION_ORDER = (
    "standard_lora", "full_sr", "subtraction_only",
    "reweight_only", "staged_neither", "class_prior_reweight",
)
```

In `TrainConfig`, replace the mutable `seed` field with the three explicit seeds and a compatibility property. Validate weighting mode and frozen canonical defaults without changing unrelated legacy defaults.

- [ ] **Step 4: Migrate training-seed assignment in the multi-seed driver**

Replace `base_cfg.seed = seed` with `base_cfg.training_seed = seed`. Do not change output paths or method lists in the legacy driver.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m unittest tests.test_canonical_conditions -v`

Expected: all Task 1 tests pass.

Run: `python -m unittest discover -s tests -v`

Expected: 4 legacy tests plus Task 1 tests pass.

- [ ] **Step 6: Commit the task checkpoint**

```powershell
git add canonical/__init__.py canonical/conditions.py configs/config.py run_multiseed.py tests/test_canonical_conditions.py
git commit -m "feat: freeze canonical seeds and conditions"
```

### Task 2: Build deterministic data sampling and HANS partitions

**Files:**
- Create: `canonical/data.py`
- Modify: `data/dataloader.py`
- Test: `tests/test_canonical_data.py`

**Interfaces:**
- Produces: `sample_dataset(dataset, count: int, seed: int)` for duck-typed Hugging Face datasets.
- Produces: `split_hans_records(records, seed=42, rng_factory=None) -> HansSplit`.
- Produces: `validate_hans_disjointness(build_ids, dev_ids, evaluation_ids) -> None`.
- Produces: `make_hans_build_loader`, `make_hans_dev_loader`, `make_hans_evaluation_loader`; legacy `make_hans_loader` remains an evaluation alias outside the canonical path.

- [ ] **Step 1: Write failing pure-data tests**

Use an in-memory fake dataset implementing `shuffle(seed)` and `select(indices)`. Assert two configs with identical `data_seed` but different `training_seed` yield identical sampled IDs.

Create records spanning two normal strata and one `<5` stratum. Inject a literal permutation factory, then assert:

```python
self.assertEqual(expected_build_ids, split.build_pair_ids)
self.assertEqual(expected_dev_ids, split.dev_pair_ids)
self.assertEqual([], split.small_strata[0].dev_pair_ids)
self.assertFalse(set(split.build_pair_ids) & set(split.dev_pair_ids))
```

Add failures for duplicate `pairID`, missing `subcase`, and evaluation overlap.

- [ ] **Step 2: Run and verify RED**

Run: `python -m unittest tests.test_canonical_data -v`

Expected: module or symbol-not-found failures.

- [ ] **Step 3: Implement the pure splitter and manifest**

Normalize each record into `{pair_id, gold_label, heuristic, subcase}`. Group by the exact three-field stratum, sort by `pair_id`, then create a fresh `default_rng(seed)` for each stratum and generate that stratum's permutation. Allocate `floor(0.20 * n)` to dev unless `n < 5`.

Serialize checksum input with sorted keys, compact separators, UTF-8, and strict finite JSON before SHA-256.

- [ ] **Step 4: Adapt the Hugging Face data layer**

Replace every MNLI fixed-subset call with `cfg.data_seed`. Use `cfg.training_seed` only for batch/random-process shuffles and random layer selection.

Build HANS-train once into build/dev datasets while preserving `pairID`, `gold_label`, `heuristic`, and `subcase`. Evaluation always comes from `heuristics_evaluation_set.txt`. Include stable metadata columns in HANS evaluation batches.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m unittest tests.test_canonical_data -v`

Expected: deterministic split, small-stratum, overlap, and fixed-data tests pass.

Run: `python -m unittest discover -s tests -v`

- [ ] **Step 6: Commit the task checkpoint**

```powershell
git add canonical/data.py data/dataloader.py tests/test_canonical_data.py
git commit -m "feat: add deterministic canonical data splits"
```

### Task 3: Add strict artifacts and reproducible HANS predictions

**Files:**
- Create: `canonical/artifacts.py`
- Create: `canonical/hans.py`
- Modify: `training/evaluate.py`
- Test: `tests/test_canonical_artifacts.py`
- Test: `tests/test_canonical_hans.py`

**Interfaces:**
- Produces: `write_json(path, value)`, `write_jsonl(path, records)`, `read_jsonl(path)`, `sha256_file(path)`.
- Produces: `aggregate_hans_predictions(records) -> dict[str, object]`.
- Extends: `eval_hans(..., prediction_context=None)`; legacy calls still receive only an aggregate dict, canonical calls receive aggregate metrics plus records.

- [ ] **Step 1: Write failing strict-serialization tests**

```python
for bad in [float("nan"), float("inf"), float("-inf")]:
    with self.assertRaises(ValueError):
        write_json(path, {"nested": [1.0, {"bad": bad}]})
```

Also assert valid `None` becomes JSON `null`, JSONL is UTF-8 and one object per line, and SHA-256 changes when file content changes.

- [ ] **Step 2: Write failing HANS recomputation tests**

Use literal prediction rows covering entailment/non-entailment, all three heuristics, and two subcases. Hand-compute expected overall and subgroup accuracies. Assert malformed/missing fields and duplicate pair IDs fail.

- [ ] **Step 3: Run and verify RED**

Run: `python -m unittest tests.test_canonical_artifacts tests.test_canonical_hans -v`

Expected: missing module/symbol failures.

- [ ] **Step 4: Implement strict atomic writers and metadata collection**

Recursively reject non-finite `float` values before writing, call `json.dump(..., allow_nan=False)`, flush a sibling temporary file, then `os.replace` it. Collect package versions through `importlib.metadata` so manifest collection does not import the ML stack.

- [ ] **Step 5: Implement pure HANS aggregation**

Validate the nine required fields, normalize labels to `entailment`/`non-entailment`, reject duplicate `pair_id`, and calculate aggregate accuracy directly from literal rows. No aggregate expectation may call the evaluator under test.

- [ ] **Step 6: Extend the model evaluator**

For canonical final evaluation, calculate `softmax(logits.float(), dim=-1)[:, 0]`, preserve metadata from the batch, construct prediction records, then call `aggregate_hans_predictions`. Do not duplicate aggregation math inside `training/evaluate.py`.

- [ ] **Step 7: Run focused and full tests**

Run: `python -m unittest tests.test_canonical_artifacts tests.test_canonical_hans -v`

Run: `python -m unittest discover -s tests -v`

- [ ] **Step 8: Commit the task checkpoint**

```powershell
git add canonical/artifacts.py canonical/hans.py training/evaluate.py tests/test_canonical_artifacts.py tests/test_canonical_hans.py
git commit -m "feat: add strict canonical artifacts and predictions"
```

### Task 4: Enforce the exact trim-ratio contract

**Files:**
- Modify: `models/ties_lora.py`
- Test: `tests/test_ties_trim_contract.py`

**Interfaces:**
- Changes only: `TIESUnlearnLoRALinear._trim_mask(dN)` keeps exactly `max(1, floor(numel × trim_ratio))` entries, ordered by descending magnitude and ascending flattened index for ties.

- [ ] **Step 1: Write the failing regression test against the real method**

Stub the minimal `torch`/`torch.nn` surface before importing `models.ties_lora`, and invoke `_trim_mask` unbound on a namespace with `trim_ratio=0.2`. Use ten values where three entries tie at the cutoff; the old threshold implementation keeps more than two and must fail.

```python
self.assertEqual([1, 4], mask.nonzero_indices())
self.assertEqual(2, sum(mask.values))
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m unittest tests.test_ties_trim_contract -v`

Expected: assertion failure showing the old threshold mask retains too many tied values.

- [ ] **Step 3: Implement stable exact selection**

```python
flat = dN.abs().reshape(-1)
k = max(1, int(flat.numel() * self.trim_ratio))
order = torch.argsort(flat, descending=True, stable=True)
mask = torch.zeros_like(flat)
mask[order[:k]] = 1.0
return mask.reshape_as(dN)
```

Reject `trim_ratio <= 0`; retain the existing all-ones fast path for ratios `>= 1`.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m unittest tests.test_ties_trim_contract -v`

Run: `python -m unittest discover -s tests -v`

- [ ] **Step 5: Commit the task checkpoint**

```powershell
git add models/ties_lora.py tests/test_ties_trim_contract.py
git commit -m "fix: enforce exact TIES trim fraction"
```

### Task 5: Add weighting, shared-checkpoint, and evaluation-isolation hooks

**Files:**
- Create: `canonical/evaluation_policy.py`
- Create: `training/weighting.py`
- Modify: `utils/optim_utils.py`
- Modify: `training/trainer.py`
- Modify: `training/baseline.py`
- Modify: `training/train_jtt.py`
- Modify: `training/train_poe.py`
- Modify: `training/train_zfilter.py`
- Test: `tests/test_canonical_weighting.py`

**Interfaces:**
- Produces: `hans_split_for_event(event: str) -> Literal["dev", "evaluation"]`; only `final_evaluation` maps to evaluation.
- Produces: `compute_class_priors(labels, gold_probabilities, gamma, classes) -> dict[int, float]`.
- Produces: `normalize_batch_weights(raw_weights) -> list[float]` for reference testing.
- Extends: checkpoint dictionaries with `config`, `class_prior_weights`, `completed_phase`, and `schema_version`.
- Extends: `train_ties_unlearn(..., stop_after_phase2=False, shared_checkpoint_path=None, method_tag=None)`.

- [ ] **Step 1: Write failing weighting and policy tests**

Hand-calculate class priors from six literal examples, assert only training labels/probabilities are accepted, and assert normalized batch weights have mean 1. Test every canonical event:

```python
for event in ("phase1_end", "phase2_end", "phase2_5", "phase3_epoch"):
    self.assertEqual("dev", hans_split_for_event(event))
self.assertEqual("evaluation", hans_split_for_event("final_evaluation"))
```

Unknown events must raise instead of defaulting to official evaluation.

- [ ] **Step 2: Run and verify RED**

Run: `python -m unittest tests.test_canonical_weighting -v`

Expected: missing module/symbol failures.

- [ ] **Step 3: Implement pure weighting calculations**

Validate equal lengths, finite probabilities in `[0, 1]`, labels in the declared class set, and presence of every class. Calculate `(1-p)^gamma`, class means, and mean-one normalization. Add thin PyTorch adapters in the same module for training batches.

- [ ] **Step 4: Add model-state-only checkpoint loading**

Keep legacy optimizer resume behavior intact. Add a separate loader that validates `completed_phase == 2`, restores only model state/history/metadata, and returns class priors. Shared checkpoint saving records the exact config and class priors before hashing.

- [ ] **Step 5: Adapt the trainer minimally**

- Seed model/training randomness with `cfg.training_seed`.
- Construct HANS dev for intermediate diagnostics; do not construct official HANS or other final OOD loaders before final evaluation.
- When `stop_after_phase2=True`, compute training-only class priors, save the shared checkpoint, and return before Phase 2.5/final evaluation.
- When `shared_checkpoint_path` is provided, build the same architecture, load Phase-2 model state and priors, and start at Phase 2.5/3.
- Dispatch Phase-3 loss by `none`, `n_guided`, or `class_prior` and retain current N-guided math.
- At final evaluation only, construct official HANS, write predictions with method/seed/checkpoint context, and save strict metrics.

- [ ] **Step 6: Adapt Standard LoRA and legacy training seeds**

Standard LoRA delays official HANS construction until training finishes. JTT/PoE/z-filter set model randomness from `training_seed`; their fixed MNLI subsets continue to come from `data_seed` through the shared data layer. Do not change their algorithms or historical output directories.

- [ ] **Step 7: Run focused and full tests**

Run: `python -m unittest tests.test_canonical_weighting -v`

Run: `python -m unittest discover -s tests -v`

- [ ] **Step 8: Compile all modified ML modules without importing dependencies**

Run: `python -m compileall -q canonical configs data models training utils`

Expected: exit code 0.

- [ ] **Step 9: Commit the task checkpoint**

```powershell
git add canonical/evaluation_policy.py training/weighting.py utils/optim_utils.py training/trainer.py training/baseline.py training/train_jtt.py training/train_poe.py training/train_zfilter.py tests/test_canonical_weighting.py
git commit -m "feat: support shared canonical phase 2 branches"
```

### Task 6: Implement the canonical runner and resume rules

**Files:**
- Create: `canonical/backend.py`
- Create: `canonical/runner.py`
- Create: `run_canonical.py`
- Test: `tests/test_canonical_runner.py`

**Interfaces:**
- Produces backend protocol methods: `prepare_shared`, `run_standard`, `run_branch`.
- Produces: `run_core(protocol_path, output_dir, backend, fresh=False, seeds=CANONICAL_TRAINING_SEEDS)`.
- CLI: `python run_canonical.py --stage core --protocol PATH --output-dir PATH [--fresh]`.

- [ ] **Step 1: Write failing orchestration tests with a fake backend**

Use one injected seed for speed. The fake preparation writes a checkpoint file; five branch calls record the received hash. Assert:

```python
self.assertEqual(1, backend.prepare_calls)
self.assertEqual(1, backend.standard_calls)
self.assertEqual(5, len(backend.branch_calls))
self.assertEqual(1, len({call.checkpoint_hash for call in backend.branch_calls}))
```

Add tests for rotated order, non-empty `--fresh` refusal, failed preparation stopping the seed, failed branch stopping later branches, and resume skipping only a `success` run whose declared output hashes still match.

- [ ] **Step 2: Run and verify RED**

Run: `python -m unittest tests.test_canonical_runner -v`

Expected: missing runner/backend failures.

- [ ] **Step 3: Implement pure preflight and orchestration**

Preflight verifies protocol file/hash, clean Git metadata supplied by the artifact helper, required `manifests/data_manifest.json` and `manifests/environment_manifest.json`, and output policy. Each run writes `running` before backend execution and `success` only after required artifacts and hashes validate. Exceptions write `failed` once and propagate to stop the seed.

- [ ] **Step 4: Implement the lazy real backend**

Import training modules only inside backend methods. Map each condition through `CanonicalCondition.apply_to_config`, set method-specific output directories, invoke shared preparation once, and pass the same verified checkpoint reference to all five dual branches.

- [ ] **Step 5: Implement the CLI**

Accept only `--stage core`; reject unknown stages and invalid paths before importing training dependencies. Default to resume. `--fresh` accepts only a new or empty directory and never deletes.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m unittest tests.test_canonical_runner -v`

Run: `python -m unittest discover -s tests -v`

Run: `python run_canonical.py --help`

Expected: help succeeds in the lightweight environment without importing PyTorch.

- [ ] **Step 7: Commit the task checkpoint**

```powershell
git add canonical/backend.py canonical/runner.py run_canonical.py tests/test_canonical_runner.py
git commit -m "feat: add resumable canonical experiment driver"
```

### Task 7: Verify Stage 1 and produce the handoff report

**Files:**
- Modify: `docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md`
- Create: `docs/paper_rebuild/STAGE1_CANONICAL_INFRASTRUCTURE_REPORT.md`

**Interfaces:**
- Produces: a self-contained Markdown handoff with Material Passport, scope, commits, files, red-green evidence, verification output, known limitations, and exact Stage 2 entry conditions.

- [ ] **Step 1: Run fresh full verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q canonical configs data models training utils run_canonical.py
git diff --check
git status --short
git diff --name-only 0ef127c -- ties_results
```

Expected: all tests pass; compile and diff checks exit 0; no historical `ties_results/` path appears.

- [ ] **Step 2: Audit the frozen requirements line by line**

Map protocol §4.1–4.7 and every Stage 1 checklist item to one implementation file and at least one test/evidence command. Record any unmet item as incomplete; do not mark the stage complete merely because tests pass.

- [ ] **Step 3: Write the handoff report**

Use `Verification Status: VERIFIED` only for lightweight Stage 1 unit/contract verification. Explicitly mark real PyTorch/Datasets/GPU integration as `UNVERIFIED — Stage 2`. Include:

- Exact commands and pass counts.
- Red failures observed before each implementation.
- Git commits and branch/worktree.
- Artifact schemas and canonical CLI.
- No canonical training run performed.
- Environment limitation: current shell lacks project ML dependencies and pytest.
- Stage 2 must run real-dependency local/GPU smoke tests before canonical execution.

- [ ] **Step 4: Update only completed checklist boxes**

Mark Stage 0 user confirmation complete. Mark each Stage 1 checkbox only if its evidence is in the report. Update the current-status block and one-page overview to show Stage 1 complete and Stage 2 awaiting user confirmation. Do not mark any Stage 2 item.

- [ ] **Step 5: Commit the documentation handoff**

```powershell
git add docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md docs/paper_rebuild/STAGE1_CANONICAL_INFRASTRUCTURE_REPORT.md
git commit -m "docs: report stage 1 canonical infrastructure"
```

- [ ] **Step 6: Verify the final branch state after commits**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q canonical configs data models training utils run_canonical.py
git diff --check
git status --short
git log --oneline 0ef127c..HEAD
```

Expected: tests and compilation pass, no diff errors, working tree clean, and the Stage 1 implementation/report commits are visible after the approved design commit.

## Plan Self-Review Record

- Spec coverage: protocol §4.1–4.7, checklist Stage 1, result schema, shared checkpoint, trim semantics, and evaluation isolation each map to a task and test.
- Placeholder scan: no unresolved placeholders, unspecified error handling, or deferred test-writing steps remain.
- Type consistency: condition weighting values are exactly `none`, `n_guided`, and `class_prior`; seed fields are exactly `data_seed`, `hans_split_seed`, and `training_seed`; checkpoint provenance uses one path/hash pair across Tasks 5–6.
- Scope: Stage 2 smoke tests, environment freeze, A100 execution, canonical results, statistics, and paper rewriting remain excluded.
