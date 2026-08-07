# Stage 2 Smoke Tests and Canonical Environment Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute an isolated, auditable tiny-data smoke workflow on the local RTX 5080 and Colab A100, prove reproducibility and artifact integrity, and freeze the A100 canonical environment without starting any formal canonical result cell.

**Architecture:** Keep `run_canonical.py --stage core` unchanged and add a separate Stage 2 driver backed by a smoke-only profile, deterministic evaluation caps, structured data-access events, a subset matrix wrapper, an independent validator, and an external monitor. Execute the same committed source archive on RTX 5080 and Colab A100, repeat `full_sr` from fresh tiny training on the same A100 runtime, then export a canonical-targeted freeze bundle outside `ties_results/canonical_v1/`.

**Tech Stack:** Python 3.12, PyTorch 2.11.0 CUDA 12.8 wheel, NumPy, Transformers, Hugging Face Datasets, `unittest`, RTX 5080, Colab A100, JSON/JSONL, PowerShell, Jupyter/Colab.

## Global Constraints

- The formal core remains 100,000 MNLI train rows, 5,000 MNLI validation rows, seeds `[42, 123, 2024, 3407, 777]`, six frozen conditions, FP32, and the protocol-frozen hyperparameters.
- Smoke uses only seed 42 and the primary set `standard_lora`, `full_sr`, and `class_prior_reweight`, plus one shared Phase-1/2 checkpoint; the repeat set is `full_sr` with a fresh shared checkpoint.
- Smoke budgets are fixed at sequence length 64, batch size 8, MNLI 96/96, one epoch per phase, four Phase-2 batches, KL 1/2/1, kNN k=3, analysis counts 16/8 and 8/4 per HANS label group, HANS evaluation 384, and 128 for each other OOD evaluation set.
- `data_seed=42`, `hans_split_seed=42`, and `training_seed=42` remain separated.
- Official HANS evaluation may be accessed only after the `final_evaluation_start` audit event.
- A100 repeat success means `abs(run1_hans_non_entailment - repeat_hans_non_entailment) <= 0.005`.
- Production monitoring is 300-second checks, 3,600-second `STALL_WARNING`, and 43,200-second hard timeout. Only hard timeout auto-terminates.
- Smoke output must be under `ties_results/stage2_smoke/`; any path containing a component named `canonical_v1` is rejected.
- The formal `ties_results/canonical_v1/` directory remains absent or empty throughout Stage 2.
- No automatic retry, parameter adjustment, precision change, batch-size change, or fallback from A100 to another GPU is allowed.
- The user's existing checklist edit changing “A100” to “Colab A100” is preserved and included in the Stage 2 planning documentation commit.

## File Structure

- `canonical/smoke.py`: immutable smoke profile, condition sets, path isolation, and repeat comparison.
- `canonical/access_audit.py`: strict structured data-access event writer and event names.
- `canonical/data.py`: stable record identities and deterministic capped selection.
- `canonical/data_manifest.py`: dataset identity/checksum entries for smoke and full canonical manifests.
- `canonical/runner.py`: private reusable condition-matrix execution while preserving `run_core()` defaults.
- `canonical/backend.py`: smoke/full manifests, per-run audit path, and shared checkpoint metadata.
- `canonical/stage2_validation.py`: independent artifact, checkpoint, audit-order, metric-recomputation, and repeat validation.
- `canonical/monitoring.py`: subprocess monitoring policy and event generation.
- `canonical/freeze.py`: canonical-targeted freeze bundle and checksum inventory.
- `canonical/source_package.py`: clean-Git bundle packaging with commit/protocol metadata.
- `data/dataloader.py`: deterministic evaluation caps, public raw OOD loaders, and access events.
- `training/trainer.py`, `training/baseline.py`: final-evaluation audit marker before final loader construction.
- `configs/config.py`: optional smoke caps and audit path; all defaults inactive for formal core.
- `run_stage2_smoke.py`: local/A100 primary and repeat smoke CLI.
- `validate_stage2_smoke.py`: validator CLI.
- `monitor_stage2_job.py`: monitor CLI.
- `freeze_stage2_environment.py`: freeze-bundle CLI.
- `package_stage2_source.py`: exact clean-commit source package CLI for Colab transfer.
- `notebooks/stage2_colab_a100_smoke.ipynb`: thin A100 executor using the committed source archive.
- `tests/test_stage2_smoke.py`: profile, subset, path, and repeat contracts.
- `tests/test_stage2_data_audit.py`: selection, identities, audit order, and manifests.
- `tests/test_stage2_validation.py`: artifact/checkpoint/metric validator contracts.
- `tests/test_stage2_monitoring.py`: completion, stall, crash, fatal-pattern, and hard-timeout contracts.
- `tests/test_stage2_freeze.py`: freeze schema, hashes, and canonical-directory isolation.
- `docs/paper_rebuild/STAGE2_SMOKE_ENVIRONMENT_FREEZE_REPORT.md`: final evidence and Stage 3 handoff.

---

### Task 0: Record the Approved Stage 2 Plan Baseline

**Files:**
- Create: `docs/superpowers/plans/2026-08-08-stage2-smoke-environment-freeze.md`
- Modify: `docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md`

**Interfaces:**
- Consumes the approved design commit `9789bdc`.
- Produces a clean planning baseline containing this implementation plan and the user's approved “Colab A100” checklist wording.

- [ ] **Step 1: Verify the plan has no unresolved placeholders or malformed diff**

Run:

```powershell
rg -n "TB[D]|TO[D]O|PLACEHOLDE[R]|implement late[r]|fill in detail[s]" docs/superpowers/plans/2026-08-08-stage2-smoke-environment-freeze.md
git diff --check
```

Expected: no placeholder matches and diff check exits 0.

- [ ] **Step 2: Confirm the only pre-implementation changes are the plan and approved checklist wording**

Run: `git status --short`

Expected:

```text
 M docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md
?? docs/superpowers/plans/2026-08-08-stage2-smoke-environment-freeze.md
```

- [ ] **Step 3: Commit the planning baseline**

```powershell
git add docs/superpowers/plans/2026-08-08-stage2-smoke-environment-freeze.md docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md
git commit -m "docs: plan stage 2 smoke validation"
git status --short
```

Expected: commit succeeds and status is clean.

---

### Task 1: Freeze the Smoke Profile and Deterministic Evaluation Selection

**Files:**
- Create: `canonical/smoke.py`
- Modify: `configs/config.py`
- Modify: `canonical/data.py`
- Modify: `data/dataloader.py`
- Test: `tests/test_stage2_smoke.py`
- Test: `tests/test_stage2_data_audit.py`

**Interfaces:**
- Produces: `SMOKE_PROFILE_NAME`, `PRIMARY_CONDITIONS`, `REPEAT_CONDITIONS`, `build_smoke_config(output_dir: Path) -> TrainConfig`, `assert_stage2_output_path(output_dir: Path, repo_root: Path) -> Path`.
- Produces: `stable_record_id(record: Mapping[str, Any], preferred_fields: Sequence[str] = ()) -> str` and `deterministic_cap_records(records, limit, seed, strata_fields=()) -> tuple[list[dict], list[str]]`.
- Adds to `TrainConfig`: `hans_eval_size`, `esnli_eval_size`, `anli_eval_size`, `snli_hard_eval_size`, `wanli_eval_size`, and `data_access_log`, all defaulting to `None`.

- [ ] **Step 1: Write failing profile and isolation tests**

```python
class Stage2SmokeProfileTest(unittest.TestCase):
    def test_profile_has_exact_frozen_budget_without_changing_core_defaults(self):
        core = TrainConfig()
        smoke = build_smoke_config(Path("out"))
        self.assertEqual((core.mnli_train_size, core.mnli_val_size), (100_000, 5_000))
        self.assertIsNone(core.hans_eval_size)
        self.assertEqual((smoke.mnli_train_size, smoke.mnli_val_size), (96, 96))
        self.assertEqual((smoke.batch_size, smoke.max_seq_length), (8, 64))
        self.assertEqual((smoke.phase1_epochs, smoke.phase2_epochs, smoke.phase3_epochs), (1, 1, 1))
        self.assertEqual(smoke.phase2_epoch_batches, 4)
        self.assertEqual(smoke.hans_eval_size, 384)
        self.assertEqual(PRIMARY_CONDITIONS, ("standard_lora", "full_sr", "class_prior_reweight"))
        self.assertEqual(REPEAT_CONDITIONS, ("full_sr",))

    def test_smoke_rejects_any_canonical_v1_path_component(self):
        with self.assertRaisesRegex(ValueError, "canonical_v1"):
            assert_stage2_output_path(Path("ties_results/canonical_v1/run"), Path.cwd())
```

- [ ] **Step 2: Write failing deterministic selection tests**

```python
def test_hans_cap_is_order_independent_and_covers_label_heuristic_strata(self):
    selected_a, ids_a = deterministic_cap_records(
        HANS_ROWS, 12, 42, ("gold_label", "heuristic", "subcase")
    )
    selected_b, ids_b = deterministic_cap_records(
        list(reversed(HANS_ROWS)), 12, 42, ("gold_label", "heuristic", "subcase")
    )
    self.assertEqual(ids_a, ids_b)
    self.assertEqual({row["gold_label"] for row in selected_a}, {"entailment", "non-entailment"})
    self.assertEqual(
        {row["heuristic"] for row in selected_a},
        {"lexical_overlap", "subsequence", "constituent"},
    )
```

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```powershell
python -m unittest discover -s tests -p "test_stage2_smoke.py" -v
python -m unittest discover -s tests -p "test_stage2_data_audit.py" -v
```

Expected: import failures for `canonical.smoke`, missing optional config fields, and missing deterministic selection functions.

- [ ] **Step 4: Implement the immutable profile and inactive core defaults**

```python
SMOKE_PROFILE_NAME = "stage2_smoke_v1"
PRIMARY_CONDITIONS = ("standard_lora", "full_sr", "class_prior_reweight")
REPEAT_CONDITIONS = ("full_sr",)

def build_smoke_config(output_dir: Path) -> TrainConfig:
    return TrainConfig(
        max_seq_length=64, mnli_train_size=96, mnli_val_size=96,
        batch_size=8, fp16=False, data_seed=42, hans_split_seed=42,
        training_seed=42, phase1_epochs=1, phase2_epochs=1,
        phase3_epochs=1, phase2_epoch_batches=4, kl_batches=1,
        kl_topk_candidates=2, layer_selection_topk=1, knn_k=3,
        knn_ref_mnli=16, knn_query_mnli=8,
        knn_ref_hans_entail=8, knn_query_hans_entail=4,
        knn_ref_hans_non_entail=8, knn_query_hans_non_entail=4,
        hans_eval_size=384, esnli_eval_size=128, anli_eval_size=128,
        snli_hard_eval_size=128, wanli_eval_size=128,
        output_dir=str(output_dir),
    )

def assert_stage2_output_path(output_dir: Path, repo_root: Path) -> Path:
    resolved = (repo_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    if "canonical_v1" in resolved.parts:
        raise ValueError("Stage 2 smoke output must not use canonical_v1")
    return resolved
```

- [ ] **Step 5: Implement stable identity and deterministic round-robin strata selection**

Use a source ID from the first non-empty preferred field; otherwise hash canonical JSON of the record. Rank each ID by `sha256(f"{seed}\0{stable_id}")`, sort within each stratum, then round-robin across sorted strata until `limit` is reached. Return rows in selected-ID order. Apply the helper before tokenization in every final evaluation loader; HANS uses `("gold_label", "heuristic", "subcase")`, while other datasets use no strata.

- [ ] **Step 6: Run focused and regression tests and confirm GREEN**

Run:

```powershell
python -m unittest discover -s tests -p "test_stage2_*.py" -v
python -m unittest discover -s tests -v
```

Expected: all tests pass; existing canonical tests still observe full evaluation defaults.

- [ ] **Step 7: Commit Task 1**

```powershell
git add configs/config.py canonical/smoke.py canonical/data.py data/dataloader.py tests/test_stage2_smoke.py tests/test_stage2_data_audit.py
git commit -m "feat: add isolated stage2 smoke profile"
```

---

### Task 2: Add Structured Data-Access Auditing and Shared Metadata

**Files:**
- Create: `canonical/access_audit.py`
- Modify: `data/dataloader.py`
- Modify: `training/trainer.py`
- Modify: `training/baseline.py`
- Modify: `canonical/backend.py`
- Modify: `canonical/runner.py`
- Test: `tests/test_stage2_data_audit.py`

**Interfaces:**
- Produces: `append_access_event(path, *, dataset, split, purpose, event) -> dict`, `record_final_evaluation_start(cfg) -> None`, and `record_dataset_access(cfg, ...) -> None`.
- Produces shared artifact: `shared_checkpoint_metadata.json` with `checkpoint_role`, `checkpoint_path`, `checkpoint_sha256`, and `class_prior_weights`.
- Adds required `data_access.jsonl` to shared and method artifact sets.

- [ ] **Step 1: Write failing access-order and metadata tests**

```python
def test_final_hans_access_follows_final_marker(self):
    append_access_event(path, dataset="hans", split=None, purpose="boundary", event="final_evaluation_start")
    append_access_event(path, dataset="hans", split="evaluation", purpose="final", event="dataset_access")
    events = read_jsonl(path)
    self.assertLess(
        next(i for i, row in enumerate(events) if row["event"] == "final_evaluation_start"),
        next(i for i, row in enumerate(events) if row.get("split") == "evaluation"),
    )

def test_shared_metadata_contains_class_priors_and_checkpoint_hash(self):
    metadata = json.loads((shared_dir / "shared_checkpoint_metadata.json").read_text())
    self.assertEqual(metadata["checkpoint_role"], "canonical_shared_phase2")
    self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
    self.assertEqual(set(metadata["class_prior_weights"]), {"0", "1", "2"})
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `python -m unittest discover -s tests -p "test_stage2_data_audit.py" -v`

Expected: missing `canonical.access_audit` and shared metadata artifact.

- [ ] **Step 3: Implement strict ordered audit events**

```python
def append_access_event(path: Path, **payload: Any) -> dict[str, Any]:
    existing = read_jsonl(path) if path.is_file() else []
    event = {
        "sequence": len(existing),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    write_jsonl(path, [*existing, event])
    return event
```

Each loader records its dataset/split/purpose when `cfg.data_access_log` is set. `training.trainer` and `training.baseline` call `record_final_evaluation_start(cfg)` immediately before constructing official HANS and other final-only loaders.

- [ ] **Step 4: Persist shared checkpoint metadata**

After `train_ties_unlearn(..., stop_after_phase2=True)` returns, `RealCanonicalBackend.prepare_shared()` writes the result’s class priors and verified checkpoint SHA-256 to `shared_checkpoint_metadata.json`. Set `cfg.data_access_log` to the run directory’s `data_access.jsonl` in `_config_for_directory()`.

- [ ] **Step 5: Require audit artifacts in runner success hashes**

Add `data_access.jsonl` to `_METHOD_OUTPUTS` and `_SHARED_OUTPUTS`; add `shared_checkpoint_metadata.json` to `_SHARED_OUTPUTS`. A completed run without either file must fail before `status.json` becomes `success`.

- [ ] **Step 6: Run focused and full tests and confirm GREEN**

Run:

```powershell
python -m unittest discover -s tests -p "test_stage2_data_audit.py" -v
python -m unittest discover -s tests -v
```

Expected: all pass, including fake-backend fixtures updated to emit the new required files.

- [ ] **Step 7: Commit Task 2**

```powershell
git add canonical/access_audit.py canonical/backend.py canonical/runner.py data/dataloader.py training/trainer.py training/baseline.py tests/test_stage2_data_audit.py tests/test_canonical_runner.py
git commit -m "feat: audit canonical data access ordering"
```

---

### Task 3: Add the Subset Matrix and Stage 2 Smoke CLI

**Files:**
- Modify: `canonical/runner.py`
- Create: `run_stage2_smoke.py`
- Modify: `tests/test_canonical_runner.py`
- Modify: `tests/test_stage2_smoke.py`

**Interfaces:**
- Produces: `run_condition_matrix(protocol_path, output_dir, backend, *, seeds, condition_tags, matrix_schema_version, fresh, git_metadata, command, repo_root) -> dict`.
- Preserves: `run_core(...)` public signature and full frozen behavior.
- CLI: `python run_stage2_smoke.py --mode {primary,repeat_full_sr} --environment {local_rtx5080,colab_a100} --protocol PATH --output-dir PATH [--fresh]`.
- Produces root-level `commands.json` containing mode, environment, exact argv, expected condition tags, profile name, and start timestamp.

- [ ] **Step 1: Write failing subset and core-regression tests**

```python
def test_stage2_primary_executes_only_three_methods_and_one_shared_prepare(self):
    result = run_condition_matrix(
        protocol, output, backend, fresh=True, seeds=(42,),
        condition_tags=PRIMARY_CONDITIONS,
        matrix_schema_version="stage2_smoke_matrix_v1",
        git_metadata=CLEAN_GIT,
    )
    self.assertEqual(backend.prepared, [42])
    self.assertEqual([tag for _, tag, _ in backend.methods], list(PRIMARY_CONDITIONS))

def test_run_core_defaults_still_execute_thirty_method_cells(self):
    result = run_core(protocol, output, backend, fresh=True, git_metadata=CLEAN_GIT)
    self.assertEqual(len(result["executed"]), 30)
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
python -m unittest discover -s tests -p "test_stage2_smoke.py" -v
python -m unittest discover -s tests -p "test_canonical_runner.py" -v
```

Expected: `run_condition_matrix` and CLI are absent.

- [ ] **Step 3: Extract the private matrix executor**

Filter `rotated_condition_order(seed)` by the requested validated tags, write the requested schema and filtered order to `run_matrix.json`, prepare one shared checkpoint per seed, and execute only those methods. Implement `run_core()` as a thin wrapper passing all six condition tags and `canonical_run_matrix_v1`.

- [ ] **Step 4: Implement GPU-locked smoke CLI**

```python
def require_expected_gpu(environment: str) -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 smoke requires CUDA")
    name = torch.cuda.get_device_name(0)
    expected = "RTX 5080" if environment == "local_rtx5080" else "A100"
    if expected not in name:
        raise RuntimeError(f"Expected {expected}, found {name}")
    return name
```

`primary` selects `PRIMARY_CONDITIONS`; `repeat_full_sr` selects `REPEAT_CONDITIONS`. Both create a fresh shared checkpoint. The CLI calls `assert_stage2_output_path()` before creating directories and records the exact command in run manifests.

Before invoking the matrix, write `commands.json` at the smoke root with the exact argv, profile, environment, mode, and selected tags. The file is included in later validation and freeze provenance.

- [ ] **Step 5: Run CLI help, focused tests, and full tests**

Run:

```powershell
python run_stage2_smoke.py --help
python -m unittest discover -s tests -p "test_stage2_smoke.py" -v
python -m unittest discover -s tests -v
```

Expected: help exits 0 without importing torch; all tests pass; core still yields 30 fake method cells.

- [ ] **Step 6: Commit Task 3**

```powershell
git add canonical/runner.py run_stage2_smoke.py tests/test_canonical_runner.py tests/test_stage2_smoke.py
git commit -m "feat: add stage2 smoke matrix driver"
```

---

### Task 4: Materialize Smoke and Full Canonical Data Manifests

**Files:**
- Create: `canonical/data_manifest.py`
- Modify: `data/dataloader.py`
- Modify: `canonical/backend.py`
- Modify: `tests/test_stage2_data_audit.py`

**Interfaces:**
- Produces: `dataset_identity_entry(records, *, source, split, preferred_id_fields, selected_limit, seed, strata_fields=()) -> dict`.
- Produces data manifest groups for MNLI, HANS, e-SNLI, ANLI, SNLI-hard, and WANLI with full and selected counts, ID arrays, and SHA-256 values.
- Public raw loaders: `load_esnli_raw()`, `load_anli_raw()`, `load_snli_hard_raw()`, and `load_wanli_raw()`.

- [ ] **Step 1: Write failing identity-manifest tests**

```python
def test_identity_entry_separates_full_and_smoke_selected_membership(self):
    entry = dataset_identity_entry(
        ROWS, source="fixture", split="test", preferred_id_fields=("uid",),
        selected_limit=2, seed=42,
    )
    self.assertEqual(entry["full_count"], 4)
    self.assertEqual(entry["selected_count"], 2)
    self.assertEqual(len(entry["full_ids_sha256"]), 64)
    self.assertEqual(len(entry["selected_ids_sha256"]), 64)
    self.assertEqual(entry["id_strategy"], "preferred_field_or_content_sha256")
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `python -m unittest discover -s tests -p "test_stage2_data_audit.py" -v`

Expected: missing `canonical.data_manifest` and raw loader interfaces.

- [ ] **Step 3: Implement strict identity entries**

Hash ID lists with canonical JSON (`sort_keys=True`, compact separators, UTF-8). Reject duplicate full IDs, empty datasets, missing selected rows, or a selected ID not present in the full set. Include source, split, selection seed, cap, count, IDs, and checksums.

- [ ] **Step 4: Expose raw OOD loaders and remove duplicate loading logic**

Rename the existing private raw loaders to the public names and make final loader constructors call them. ANLI returns concatenated `test_r1/test_r2/test_r3`; every public loader returns raw, filtered examples before tokenization.

- [ ] **Step 5: Extend backend manifest initialization**

`RealCanonicalBackend.initialize_manifests()` uses its base config to distinguish smoke-selected caps from full canonical selection. MNLI records the sampled fixed membership; HANS records build/dev/evaluation full IDs plus selected evaluation IDs; every OOD dataset records full and selected identities. The top-level manifest includes `schema_version="canonical_data_manifest_v2"` and `scope="stage2_smoke"` when any cap is set, otherwise `scope="canonical_v1"`.

- [ ] **Step 6: Run manifest and regression tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_stage2_data_audit.py" -v
python -m unittest discover -s tests -v
```

Expected: all pass without network because tests inject fixture records.

- [ ] **Step 7: Commit Task 4**

```powershell
git add canonical/data_manifest.py canonical/backend.py data/dataloader.py tests/test_stage2_data_audit.py
git commit -m "feat: bind stage2 datasets to stable identities"
```

---

### Task 5: Build the Independent Stage 2 Validator

**Files:**
- Create: `canonical/stage2_validation.py`
- Create: `validate_stage2_smoke.py`
- Create: `tests/test_stage2_validation.py`

**Interfaces:**
- Produces: `validate_smoke_root(root: Path, *, expected_conditions: Sequence[str], canonical_dir: Path) -> dict`.
- Produces: `compare_a100_repeat(primary_root: Path, repeat_root: Path, tolerance: float = 0.005) -> dict`.
- Produces: `compare_metric_values(primary: float, repeat: float, *, tolerance: float) -> dict`.
- CLI writes `stage2_validation.json` and `stage2_validation.md` beside the validated smoke root.

- [ ] **Step 1: Write failing metric, audit, checkpoint, and repeat tests**

```python
def test_validator_recomputes_hans_and_matches_metrics_exactly(self):
    report = validate_smoke_root(ROOT, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=CANONICAL)
    self.assertEqual(report["checks"]["hans_recomputation"]["state"], "pass")

def test_validator_rejects_evaluation_access_before_final_marker(self):
    write_jsonl(run / "data_access.jsonl", [EVALUATION_EVENT, FINAL_MARKER])
    with self.assertRaisesRegex(ValueError, "before final_evaluation_start"):
        validate_smoke_root(ROOT, expected_conditions=PRIMARY_CONDITIONS, canonical_dir=CANONICAL)

def test_repeat_tolerance_is_inclusive_at_half_percentage_point(self):
    report = compare_metric_values(0.400, 0.405, tolerance=0.005)
    self.assertTrue(report["within_tolerance"])
```

- [ ] **Step 2: Run validator tests and confirm RED**

Run: `python -m unittest discover -s tests -p "test_stage2_validation.py" -v`

Expected: validator module and CLI are absent.

- [ ] **Step 3: Implement strict artifact and HANS checks**

Load JSON with `parse_constant` rejection. Verify every `status.json` success hash against disk. Read `hans_predictions.jsonl`, run `aggregate_hans_predictions()`, and compare the returned object exactly with `metrics.json["final"]["hans"]`. Require each row’s method, seed, and checkpoint hash to match its run manifest.

- [ ] **Step 4: Implement audit-order and shared-checkpoint checks**

Require all evaluation accesses to follow `final_evaluation_start`. Require both dual branches’ `run_manifest.json["shared_phase2_checkpoint"]` objects to be identical and to match `shared_checkpoint.json`. Require `shared_checkpoint_metadata.json` to contain finite positive class-prior weights for labels `0`, `1`, and `2`; require the class-prior branch log to state that class-prior weighting loaded those values.

- [ ] **Step 5: Implement repeat comparison and report writing**

Compare HANS non-entailment as the primary value and record MNLI accuracy as a secondary diagnostic. Use absolute difference and inclusive `<= 0.005`. Require matching protocol, data, environment, commit, seed, config, and GPU fields, excluding output path and timestamps.

- [ ] **Step 6: Run focused and full tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_stage2_validation.py" -v
python validate_stage2_smoke.py --help
python -m unittest discover -s tests -v
```

Expected: all pass and CLI help exits 0 without ML imports.

- [ ] **Step 7: Commit Task 5**

```powershell
git add canonical/stage2_validation.py validate_stage2_smoke.py tests/test_stage2_validation.py
git commit -m "feat: validate stage2 smoke artifacts"
```

---

### Task 6: Add the Frozen Monitoring and Timeout Layer

**Files:**
- Create: `canonical/monitoring.py`
- Create: `monitor_stage2_job.py`
- Create: `tests/test_stage2_monitoring.py`

**Interfaces:**
- Produces: `MonitorPolicy(check_interval_seconds, stall_seconds, hard_timeout_seconds)` and `PRODUCTION_POLICY = MonitorPolicy(300, 3600, 43200)`.
- Produces: `monitor_command(command, *, cwd, events_path, watched_paths, policy=PRODUCTION_POLICY, clock=time.monotonic, sleep=time.sleep, popen_factory=subprocess.Popen) -> int`.
- CLI syntax: `python monitor_stage2_job.py --events PATH --watch PATH... -- COMMAND...`.

- [ ] **Step 1: Write failing monitor state-machine tests with fake clock/processes**

```python
def test_production_policy_is_frozen(self):
    self.assertEqual(PRODUCTION_POLICY, MonitorPolicy(300, 3600, 43200))

def test_stall_is_advisory_and_does_not_terminate(self):
    process = FakeProcess(exit_after_checks=4, exit_code=0)
    result = monitor_command(CMD, cwd=ROOT, events_path=EVENTS, watched_paths=[LOG],
                             policy=MonitorPolicy(1, 2, 10), clock=CLOCK, sleep=SLEEP,
                             popen_factory=lambda *a, **k: process)
    self.assertEqual(result, 0)
    self.assertIn("STALL_WARNING", event_names(EVENTS))
    self.assertFalse(process.terminated)

def test_hard_timeout_terminates_then_kills_if_needed(self):
    process = FakeProcess(exit_after_checks=None, ignores_terminate=True)
    result = monitor_command(CMD, cwd=ROOT, events_path=EVENTS, watched_paths=[LOG],
                             policy=MonitorPolicy(1, 20, 3), clock=CLOCK, sleep=SLEEP,
                             popen_factory=lambda *a, **k: process)
    self.assertEqual(result, 124)
    self.assertTrue(process.terminated)
    self.assertTrue(process.killed)
```

- [ ] **Step 2: Run monitor tests and confirm RED**

Run: `python -m unittest discover -s tests -p "test_stage2_monitoring.py" -v`

Expected: monitoring module and CLI are absent.

- [ ] **Step 3: Implement progress fingerprinting and event output**

Fingerprint watched files by existence, size, and nanosecond mtime. Emit strict JSONL events for `STARTED`, `STATUS_CHECK`, `PROGRESS`, `STALL_WARNING`, `FATAL_PATTERN`, `CRASHED`, `HARD_TIMEOUT`, and `COMPLETED`. Fatal patterns include CUDA OOM, non-finite loss, download failure, checkpoint hash mismatch, and prediction-row mismatch; they record failure and wait for process exit without modifying the command.

- [ ] **Step 4: Implement timeout termination**

At hard timeout call `terminate()`, wait up to 10 seconds, then call `kill()` only if still alive. Return 124. All other return codes mirror the child process.

- [ ] **Step 5: Run accelerated drill tests and full suite**

Run:

```powershell
python -m unittest discover -s tests -p "test_stage2_monitoring.py" -v
python monitor_stage2_job.py --help
python -m unittest discover -s tests -v
```

Expected: all monitor paths pass without real waits; help exits 0.

- [ ] **Step 6: Commit Task 6**

```powershell
git add canonical/monitoring.py monitor_stage2_job.py tests/test_stage2_monitoring.py
git commit -m "feat: monitor canonical jobs without auto tuning"
```

---

### Task 7: Build the Freeze Bundle and Thin Colab A100 Executor

**Files:**
- Create: `canonical/freeze.py`
- Create: `canonical/source_package.py`
- Create: `freeze_stage2_environment.py`
- Create: `package_stage2_source.py`
- Create: `notebooks/stage2_colab_a100_smoke.ipynb`
- Create: `tests/test_stage2_freeze.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `build_freeze_bundle(protocol_path, smoke_root, output_dir, repo_root, *, source_archive_path, commands_path, backend_factory=...) -> dict`.
- Produces: `build_source_package(repo_root, protocol_path, output_path) -> dict`, which places `stage2_source.bundle` and `source_metadata.json` inside `stage2_source.zip`.
- Freeze bundle contains protocol snapshot/hash, full canonical data manifest, A100 environment manifest, `pip_freeze.txt`, `source_commit.txt`, `source_archive_sha256.txt`, `commands.json`, and `checksum_inventory.json`.
- Notebook input filename is `/content/stage2_source.zip`; exported lightweight evidence filename is `/content/stage2_a100_evidence.zip`.
- Freeze CLI supports creation with `--source-archive` and `--commands`, plus isolated `--verify-only --output-dir PATH` verification.

- [ ] **Step 1: Write failing freeze isolation and checksum tests**

```python
def test_freeze_bundle_is_canonical_targeted_but_outside_canonical_directory(self):
    result = build_freeze_bundle(
        PROTOCOL, SMOKE, FREEZE, ROOT,
        source_archive_path=ARCHIVE, commands_path=COMMANDS,
        backend_factory=FAKE_BACKEND,
    )
    self.assertEqual(result["target_schema"], "canonical_v1")
    self.assertFalse(CANONICAL.exists())
    manifest = json.loads((FREEZE / "manifests/data_manifest.json").read_text(encoding="utf-8"))
    self.assertEqual(manifest["scope"], "canonical_v1")

def test_checksum_inventory_matches_every_declared_file(self):
    inventory = json.loads((FREEZE / "checksum_inventory.json").read_text(encoding="utf-8"))
    for relative, expected in inventory["files"].items():
        self.assertEqual(sha256_file(FREEZE / relative), expected)

def test_source_package_binds_clean_commit_protocol_and_git_bundle(self):
    metadata = build_source_package(ROOT, PROTOCOL, ARCHIVE)
    self.assertFalse(metadata["git"]["dirty"])
    self.assertEqual(len(metadata["git"]["commit"]), 40)
    self.assertEqual(len(metadata["protocol_sha256"]), 64)
    self.assertEqual(len(metadata["bundle_sha256"]), 64)
```

- [ ] **Step 2: Run freeze tests and confirm RED**

Run: `python -m unittest discover -s tests -p "test_stage2_freeze.py" -v`

Expected: freeze module and CLI are absent.

- [ ] **Step 3: Implement freeze bundle creation**

Require clean Git and successful validated A100 smoke. Instantiate the real backend with default `TrainConfig()` to create the full canonical data manifest in the bundle, copy the A100 environment manifest, write full pip freeze as text, and compute the checksum inventory last. Reject output paths containing `canonical_v1` and refuse non-empty `--fresh` destinations.

- [ ] **Step 4: Implement exact clean-commit source packaging**

`build_source_package()` fails on dirty Git, runs `git bundle create <temp>/stage2_source.bundle HEAD`, writes `source_metadata.json` with commit, branch, protocol SHA-256, and bundle SHA-256, then creates `stage2_source.zip` with those two files. `package_stage2_source.py` exposes `--repo-root`, `--protocol`, and `--output`.

- [ ] **Step 5: Create the Colab notebook as a thin command runner**

Notebook cells perform these exact gates:

```python
gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip().splitlines()[0]
if "A100" not in gpu:
    raise RuntimeError(f"Stage 2 requires A100, found {gpu}")
```

```bash
python -m unittest discover -s tests -v
python monitor_stage2_job.py --events ties_results/.stage2_monitor/colab_a100_run1.events.jsonl --watch ties_results/stage2_smoke/colab_a100_run1 -- python run_stage2_smoke.py --mode primary --environment colab_a100 --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md --output-dir ties_results/stage2_smoke/colab_a100_run1 --fresh
python validate_stage2_smoke.py --root ties_results/stage2_smoke/colab_a100_run1 --canonical-dir ties_results/canonical_v1
python monitor_stage2_job.py --events ties_results/.stage2_monitor/colab_a100_repeat_full_sr.events.jsonl --watch ties_results/stage2_smoke/colab_a100_repeat_full_sr -- python run_stage2_smoke.py --mode repeat_full_sr --environment colab_a100 --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md --output-dir ties_results/stage2_smoke/colab_a100_repeat_full_sr --fresh
python validate_stage2_smoke.py --root ties_results/stage2_smoke/colab_a100_run1 --canonical-dir ties_results/canonical_v1 --compare-repeat ties_results/stage2_smoke/colab_a100_repeat_full_sr
python freeze_stage2_environment.py --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md --smoke-root ties_results/stage2_smoke/colab_a100_run1 --source-archive /content/stage2_source.zip --commands ties_results/stage2_smoke/colab_a100_run1/commands.json --output-dir ties_results/stage2_smoke/freeze_bundle --fresh
```

Before these commands, the notebook extracts `stage2_source.zip`, verifies `stage2_source.bundle` against `source_metadata.json`, clones the bundle, checks out the recorded commit, and asserts `git status --porcelain` is empty. The notebook must keep `--events` in the sibling `ties_results/.stage2_monitor/` directory; it must never place monitor evidence inside a child `--fresh --output-dir` root.

The export cell excludes `*.pt` model files but includes checkpoint SHA-256 metadata, configs, manifests, metrics, predictions, logs, validation outputs, sibling `ties_results/.stage2_monitor/` JSONL monitor evidence, and the freeze bundle. The evidence archive must preserve relative paths rooted at `ties_results/`; this runtime evidence is not part of the source package.

- [ ] **Step 6: Ignore only generated Stage 2 runtime files**

Add `.venv-stage2/`, `.uv-cache/`, `ties_results/stage2_smoke/`, `ties_results/.stage2_monitor/`, and `stage2_source.zip` to `.gitignore`. Do not ignore the notebook, tests, plan, or final report.

- [ ] **Step 7: Run notebook JSON, packaging, freeze, and full regression checks**

Run:

```powershell
python -m json.tool notebooks/stage2_colab_a100_smoke.ipynb *> $null
python package_stage2_source.py --help
python freeze_stage2_environment.py --help
python -m unittest discover -s tests -p "test_stage2_freeze.py" -v
python -m unittest discover -s tests -v
git diff --check
```

Expected: all exit 0.

- [ ] **Step 8: Commit Task 7 and verify a clean code commit**

```powershell
git add .gitignore canonical/freeze.py canonical/source_package.py freeze_stage2_environment.py package_stage2_source.py notebooks/stage2_colab_a100_smoke.ipynb tests/test_stage2_freeze.py
git commit -m "feat: freeze stage2 A100 execution environment"
git status --short
git rev-parse HEAD
```

Expected: status is clean and the printed commit becomes the smoke-verified code commit.

---

### Task 8: Create the Local RTX 5080 Environment and Run the Real Local Smoke

**Files:**
- Generate ignored: `.venv-stage2/`
- Generate ignored: `.uv-cache/`
- Generate ignored: `ties_results/stage2_smoke/local_rtx5080/`

**Interfaces:**
- Consumes the clean Task 7 commit and Stage 2 CLIs.
- Produces local tests, environment manifest, primary smoke artifacts, monitor events, and validation report.

- [ ] **Step 1: Create an isolated Python 3.12 environment**

Run:

```powershell
$env:UV_CACHE_DIR = (Resolve-Path '.').Path + '\.uv-cache'
uv venv --python 3.12 .venv-stage2
uv pip install --python .venv-stage2\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-stage2\Scripts\python.exe -r requirements.txt pytest
```

The PyTorch CUDA 12.8 wheel command comes from the official PyTorch version matrix. Stop if resolution or installation fails; do not substitute a nightly or CPU build.

- [ ] **Step 2: Verify exact GPU and library imports**

Run:

```powershell
.venv-stage2\Scripts\python.exe -c "import torch,transformers,datasets,numpy; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),transformers.__version__,datasets.__version__,numpy.__version__); assert torch.cuda.is_available(); assert 'RTX 5080' in torch.cuda.get_device_name(0)"
nvidia-smi
```

Expected: CUDA is available and GPU is RTX 5080.

- [ ] **Step 3: Run the full suite in the real ML environment**

Run: `.venv-stage2\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass. A failure stops Stage 2 for diagnosis; do not run the smoke command.

- [ ] **Step 4: Run the monitored local primary smoke**

Run:

```powershell
.venv-stage2\Scripts\python.exe monitor_stage2_job.py --events ties_results/.stage2_monitor/local_rtx5080.events.jsonl --watch ties_results/stage2_smoke/local_rtx5080 -- .venv-stage2\Scripts\python.exe run_stage2_smoke.py --mode primary --environment local_rtx5080 --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md --output-dir ties_results/stage2_smoke/local_rtx5080 --fresh
```

Expected: exit 0; one shared checkpoint and three method statuses are success.

- [ ] **Step 5: Independently validate local artifacts**

Run:

```powershell
.venv-stage2\Scripts\python.exe validate_stage2_smoke.py --root ties_results/stage2_smoke/local_rtx5080 --canonical-dir ties_results/canonical_v1
.venv-stage2\Scripts\python.exe -m pip freeze > ties_results/stage2_smoke/local_rtx5080/pip_freeze.txt
```

Expected: every local validator check passes and `canonical_v1` is absent or empty.

- [ ] **Step 6: Record local evidence without committing generated artifacts**

Run:

```powershell
git status --short
git rev-parse HEAD
Get-FileHash ties_results/stage2_smoke/local_rtx5080/stage2_validation.json -Algorithm SHA256
```

Expected: worktree remains clean because runtime outputs are ignored; record the hashes for the final report.

---

### Task 9: Execute and Repeat the Smoke on the Same Colab A100 Runtime

**Files:**
- Generate ignored: `stage2_source.zip`
- Import ignored: `ties_results/stage2_smoke/colab_a100_run1/`
- Import ignored: `ties_results/stage2_smoke/colab_a100_repeat_full_sr/`
- Import ignored: `ties_results/stage2_smoke/freeze_bundle/`
- Import ignored: `ties_results/.stage2_monitor/colab_a100_run1.events.jsonl`
- Import ignored: `ties_results/.stage2_monitor/colab_a100_repeat_full_sr.events.jsonl`

**Interfaces:**
- Consumes the clean smoke-verified code commit and notebook.
- Produces same-runtime A100 run 1, repeat, comparison, full canonical data manifest, A100 environment freeze, and checksum inventory.

- [ ] **Step 1: Package the exact clean commit**

Run:

```powershell
git status --short
.venv-stage2\Scripts\python.exe package_stage2_source.py --repo-root . --protocol docs/paper_rebuild/FROZEN_EXPERIMENT_PROTOCOL.md --output stage2_source.zip
Get-FileHash stage2_source.zip -Algorithm SHA256
```

Expected: status is clean; `source_metadata.json` inside the zip records the same HEAD, protocol SHA-256, and Git-bundle SHA-256; record the outer archive SHA-256.

- [ ] **Step 2: Open Colab and allocate A100**

Use the in-app browser with the signed-in Colab session, upload `notebooks/stage2_colab_a100_smoke.ipynb`, select an A100 runtime, and run the GPU gate cell. Stop if the model string does not contain `A100`.

- [ ] **Step 3: Upload the exact source archive and install dependencies**

Upload `stage2_source.zip`. The notebook unpacks it, verifies the expected commit/archive hash recorded in its execution metadata, installs `torch==2.11.0` from the CUDA 12.8 wheel index plus repository dependencies, and records the pre/post environment.

- [ ] **Step 4: Run full tests and A100 primary smoke under monitoring**

Execute the notebook test and primary-smoke cells. Expected: all tests pass, shared prepare succeeds, and `standard_lora`, `full_sr`, and `class_prior_reweight` succeed with A100 recorded in their environment manifest.

- [ ] **Step 5: Repeat `full_sr` from fresh tiny Phase-1/2 training in the same runtime**

Execute the repeat cell without restarting or reconnecting the runtime. Expected: a new shared checkpoint and `full_sr` branch succeed; environment manifest and package freeze match run 1.

- [ ] **Step 6: Compare the A100 runs and build the freeze bundle**

Execute comparison and freeze cells. Expected:

```text
abs(primary HANS non-entailment - repeat HANS non-entailment) <= 0.005
canonical_v1 directory absent or empty
freeze data manifest scope = canonical_v1
GPU contains A100
all checksum inventory entries match
```

- [ ] **Step 7: Export and download lightweight evidence**

Download `/content/stage2_a100_evidence.zip` and extract it at the repository root so its relative paths rooted at `ties_results/` restore both `ties_results/stage2_smoke/` and the sibling `ties_results/.stage2_monitor/` evidence. Do not extract only under `ties_results/stage2_smoke/`. Verify its exported checksum inventory locally. Do not export model `.pt` files; retain their path/hash/class-prior metadata and the A100-side validator evidence.

- [ ] **Step 8: Re-run local validation over imported A100 evidence**

Run:

```powershell
.venv-stage2\Scripts\python.exe validate_stage2_smoke.py --root ties_results/stage2_smoke/colab_a100_run1 --canonical-dir ties_results/canonical_v1 --compare-repeat ties_results/stage2_smoke/colab_a100_repeat_full_sr
.venv-stage2\Scripts\python.exe freeze_stage2_environment.py --verify-only --output-dir ties_results/stage2_smoke/freeze_bundle
```

Expected: exported evidence, repeat tolerance, environment identity, and freeze inventory all pass.

---

### Task 10: Final Verification, Checklist Update, and Stage 2 Report

**Files:**
- Modify: `docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md`
- Create: `docs/paper_rebuild/STAGE2_SMOKE_ENVIRONMENT_FREEZE_REPORT.md`

**Interfaces:**
- Consumes all local/A100 validation JSON, monitor events, environment manifests, checksums, commands, and Git commits.
- Produces the user-facing Stage 2 handoff report and only evidence-supported checklist updates.

- [ ] **Step 1: Invoke the required completion verification workflow**

Use `superpowers:verification-before-completion` before any PASS or completion claim.

- [ ] **Step 2: Re-run fresh verification commands**

Run:

```powershell
.venv-stage2\Scripts\python.exe -m unittest discover -s tests -v
.venv-stage2\Scripts\python.exe -m compileall -q canonical configs data models training utils run_canonical.py run_stage2_smoke.py validate_stage2_smoke.py monitor_stage2_job.py freeze_stage2_environment.py
.venv-stage2\Scripts\python.exe validate_stage2_smoke.py --root ties_results/stage2_smoke/local_rtx5080 --canonical-dir ties_results/canonical_v1
.venv-stage2\Scripts\python.exe validate_stage2_smoke.py --root ties_results/stage2_smoke/colab_a100_run1 --canonical-dir ties_results/canonical_v1 --compare-repeat ties_results/stage2_smoke/colab_a100_repeat_full_sr
.venv-stage2\Scripts\python.exe freeze_stage2_environment.py --verify-only --output-dir ties_results/stage2_smoke/freeze_bundle
git diff --check
git status --short
```

Expected: all commands exit 0; status is clean before report edits; canonical_v1 is absent or empty.

- [ ] **Step 3: Update only proven Stage 2 checklist items**

Change every Stage 2 item to `[x]` only if its validator/report evidence exists. Change the current-status block to `[已完成，待用户验收] 阶段 2`; keep Stage 3 as waiting. If any gate fails, leave its checkbox unchecked and mark the stage partial.

- [ ] **Step 4: Write the Stage 2 experiment report**

The report must include:

```text
Material Passport and verification status
scope and non-goals
smoke-verified code commit and final documentation commit relationship
exact local and A100 commands
hardware/software/pip freeze tables
unit-test counts and exit statuses
local smoke artifact/check results
A100 primary and repeat metrics with absolute difference
official HANS access-order evidence
shared path/hash and class-prior restoration evidence
HANS prediction recomputation evidence
full data IDs/counts/checksums and freeze-bundle inventory
monitor drill and production thresholds
canonical_v1 absence/emptiness evidence
anomalies, fixes, reruns, and protocol-version assessment
remaining limitations and explicit Stage 3 authorization state
```

Use `Verification Status: VERIFIED` only if both real environments and the same-runtime A100 repeat passed. Otherwise use `PARTIAL` or `BLOCKED` and do not authorize Stage 3.

- [ ] **Step 5: Validate report claims against machine-readable evidence**

Search every numeric value, hash, version, test count, and path in the report against the source artifacts. Run:

```powershell
rg -n "TB[D]|TO[D]O|UNVERIFIED|placeholder" docs/paper_rebuild/STAGE2_SMOKE_ENVIRONMENT_FREEZE_REPORT.md
git diff --check
```

Expected: no placeholders or unintended unverified claims; diff check passes.

- [ ] **Step 6: Commit the report and approved checklist wording**

```powershell
git add docs/paper_rebuild/PAPER_REBUILD_MASTER_CHECKLIST.md docs/paper_rebuild/STAGE2_SMOKE_ENVIRONMENT_FREEZE_REPORT.md
git commit -m "docs: report stage 2 smoke environment freeze"
git status --short
git log -2 --oneline
```

Expected: worktree clean. The report records the smoke-verified code commit; the final docs commit is its documentation-only successor.

- [ ] **Step 7: Hand off and stop before Stage 3**

Provide the report link, final status, smoke-verified commit, freeze-bundle checksum, and any unresolved caveats. Explicitly request user acceptance before any formal canonical run.
