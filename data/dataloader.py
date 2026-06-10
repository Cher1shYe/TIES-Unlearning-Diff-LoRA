import random
import re
import numpy as np
import csv
import json
import urllib.request
import torch
from typing import Tuple
from torch.utils.data import DataLoader
from datasets import Dataset, Value, concatenate_datasets, load_dataset

from configs.config import TrainConfig

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _tokenize_pair(tok, batch, max_len: int):
    return tok(
        batch["premise"], batch["hypothesis"],
        truncation=True, padding="max_length", max_length=max_len,
    )

def _tokenize_hypothesis_only(tok, batch, max_len: int):
    # Hypothesis-only encoding for the bias model used by PoE / z-filtering baselines.
    return tok(
        batch["hypothesis"],
        truncation=True, padding="max_length", max_length=max_len,
    )

def _sample(ds: Dataset, n: int, seed: int) -> Dataset:
    return ds.shuffle(seed=seed).select(range(n)) if n and n < len(ds) else ds

def _load_hans_dataset(split: str = "eval"):
    # split="eval" -> heuristics_evaluation_set.txt  (held-out test only)
    # split="train" -> heuristics_train_set.txt      (shortcut capture + localization)
    # The two files are disjoint example sets with identical columns, so the choice
    # only affects *which* HANS examples are seen, never the downstream pipeline.
    fname = "heuristics_train_set.txt" if split == "train" else "heuristics_evaluation_set.txt"
    hans_url = f"https://raw.githubusercontent.com/tommccoy1/hans/master/{fname}"
    try:
        hans = load_dataset("csv", data_files=hans_url, delimiter="\t", split="train")
    except Exception:
        hans = load_dataset("csv", data_files=f"./{fname}", delimiter="\t", split="train")
    return hans

def _load_esnli_dataset():
    esnli_url = "https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/refs/heads/master/dataset/esnli_test.csv"

    # 1. Pull the raw text file directly over the network
    with urllib.request.urlopen(esnli_url) as response:
        lines = [line.decode('utf-8') for line in response.readlines()]

    # 2. Parse using standard comma delimiter (since the header is comma-separated)
    reader = csv.DictReader(lines, delimiter=",")

    # 3. Rebuild as a clean list of dictionaries
    data_list = [row for row in reader]

    # 4. Wrap it straight back into a Hugging Face Dataset format
    return Dataset.from_list(data_list)

def _prepare_hans_base_dataset(cfg: TrainConfig, tok) -> Dataset:
    # Evaluation set only — always the held-out HANS split, never used for training.
    hans = _load_hans_dataset("eval")

    label_map = {"entailment": 0, "non-entailment": 1}
    heuristic_map = {"lexical_overlap": 0, "subsequence": 1, "constituent": 2}

    hans = hans.map(lambda ex: {"label": label_map.get(ex["gold_label"], -1)})
    if "sentence1" in hans.column_names:
        hans = hans.rename_column("sentence1", "premise")
    if "sentence2" in hans.column_names:
        hans = hans.rename_column("sentence2", "hypothesis")
    hans = hans.filter(lambda ex: ex["label"] in (0, 1))

    def _tok_hans(batch):
        out = _tokenize_pair(tok, batch, cfg.max_seq_length)
        out["label"] = batch["label"]
        out["heuristic"] = [heuristic_map.get(h, -1) for h in batch["heuristic"]]
        return out

    return hans.map(_tok_hans, batched=True)

def _prepare_hans_analysis_base(cfg: TrainConfig) -> Dataset:
    # The analysis phase only needs premise/hypothesis/label, not the heuristic field.
    # Used for Phase-2 shortcut capture and Phase-2.5 layer localization. Pull from
    # the HANS *train* split when hans_clean_split is on, so the HANS evaluation set
    # (used by make_hans_loader) stays strictly held-out — no train/eval leakage.
    split = "train" if getattr(cfg, "hans_clean_split", True) else "eval"
    hans = _load_hans_dataset(split)
    label_map = {"entailment": 0, "non-entailment": 1}

    hans = hans.map(lambda ex: {"label": label_map.get(ex["gold_label"], -1)})
    if "sentence1" in hans.column_names:
        hans = hans.rename_column("sentence1", "premise")
    if "sentence2" in hans.column_names:
        hans = hans.rename_column("sentence2", "hypothesis")
    return hans.filter(lambda ex: ex["label"] in (0, 1))

def _prepare_esnli_test_dataset(cfg: TrainConfig, tok) -> Dataset:
    esnli = _load_esnli_dataset()
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    esnli = esnli.map(lambda ex: {"label": label_map.get(ex["gold_label"], -1)})
    if "Sentence1" in esnli.column_names:
        esnli = esnli.rename_column("Sentence1", "premise")
    if "Sentence2" in esnli.column_names:
        esnli = esnli.rename_column("Sentence2", "hypothesis")
    esnli = esnli.filter(lambda ex: ex["label"] in (0, 1, 2) and ex["premise"] is not None and ex["hypothesis"] is not None)

    def _tok_esnli(batch):
        out = _tokenize_pair(tok, batch, cfg.max_seq_length)
        out["label"] = batch["label"]
        return out

    return esnli.map(_tok_esnli, batched=True)

def make_mnli_loaders(cfg: TrainConfig, tok, return_dataset=False):
    # 1. Fetch and sample data
    ds = load_dataset("nyu-mll/glue", "mnli")
    train_ds = _sample(ds["train"], cfg.mnli_train_size, cfg.seed)
    val_ds = _sample(ds["validation_matched"], cfg.mnli_val_size, cfg.seed)

    # 2. Tokenize processing
    train_ds = train_ds.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)
    val_ds = val_ds.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)

    # 3. Set PyTorch format
    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)

    # 4. Create original Loaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # 5. [Added logic] JTT requires the underlying Dataset; return it together when return_dataset is set to True
    if return_dataset:
        return train_loader, val_loader, train_ds
        
    # Otherwise, keep the standard return format unchanged
    return train_loader, val_loader

def make_hans_loader(cfg: TrainConfig, tok):
    hans = _prepare_hans_base_dataset(cfg, tok)
    hans.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "heuristic"])
    return DataLoader(hans, batch_size=cfg.batch_size, shuffle=False)

def make_esnli_test_loader(cfg: TrainConfig, tok):
    test_ds = _prepare_esnli_test_dataset(cfg, tok)
    cols = ["input_ids", "attention_mask", "label"]
    test_ds.set_format(type="torch", columns=cols)

    return DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

def _prepare_anli_test_dataset(cfg: TrainConfig, tok) -> Dataset:
    # Adversarial NLI (Nie et al., 2020). A held-out OOD/adversarial benchmark the
    # method never touches during training or layer localization. The three test
    # rounds (R1/R2/R3) are concatenated into one test set. ANLI labels already use
    # the MNLI mapping (0=entailment, 1=neutral, 2=contradiction).
    ds = load_dataset("facebook/anli")
    anli = concatenate_datasets([ds["test_r1"], ds["test_r2"], ds["test_r3"]])
    anli = anli.filter(lambda ex: ex["label"] in (0, 1, 2)
                       and ex["premise"] is not None and ex["hypothesis"] is not None)

    def _tok_anli(batch):
        out = _tokenize_pair(tok, batch, cfg.max_seq_length)
        out["label"] = batch["label"]
        return out

    return anli.map(_tok_anli, batched=True)

def make_anli_test_loader(cfg: TrainConfig, tok):
    test_ds = _prepare_anli_test_dataset(cfg, tok)
    cols = ["input_ids", "attention_mask", "label"]
    test_ds.set_format(type="torch", columns=cols)
    return DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

def _load_snli_hard_dataset():
    # SNLI-hard test set (Gururangan et al., 2018): the SNLI test subset that a
    # hypothesis-only model gets wrong — a held-out shortcut stress test.
    url = "https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl"
    with urllib.request.urlopen(url) as response:
        lines = [line.decode("utf-8") for line in response.readlines()]
    data_list = [json.loads(line) for line in lines if line.strip()]
    return Dataset.from_list(data_list)

def _prepare_snli_hard_test_dataset(cfg: TrainConfig, tok) -> Dataset:
    snli = _load_snli_hard_dataset()
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    snli = snli.map(lambda ex: {"label": label_map.get(ex["gold_label"], -1)})
    if "sentence1" in snli.column_names:
        snli = snli.rename_column("sentence1", "premise")
    if "sentence2" in snli.column_names:
        snli = snli.rename_column("sentence2", "hypothesis")
    snli = snli.filter(lambda ex: ex["label"] in (0, 1, 2)
                       and ex["premise"] is not None and ex["hypothesis"] is not None)

    def _tok_snli(batch):
        out = _tokenize_pair(tok, batch, cfg.max_seq_length)
        out["label"] = batch["label"]
        return out

    return snli.map(_tok_snli, batched=True)

def make_snli_hard_test_loader(cfg: TrainConfig, tok):
    test_ds = _prepare_snli_hard_test_dataset(cfg, tok)
    cols = ["input_ids", "attention_mask", "label"]
    test_ds.set_format(type="torch", columns=cols)
    return DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

def _prepare_wanli_test_dataset(cfg: TrainConfig, tok) -> Dataset:
    # WANLI (Liu et al., 2022): worker-and-AI collaborative NLI with harder, more naturally
    # adversarial examples than MNLI. A held-out OOD generalization benchmark the method
    # never sees during training or layer localization. Uses the MNLI label mapping
    # (0=entailment, 1=neutral, 2=contradiction).
    ds = load_dataset("alisawuffles/WANLI")
    wanli = ds["test"]
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def _norm_label(ex):
        # WANLI stores the gold label as a string in "gold"; fall back to an int "label".
        if ex.get("gold") is not None:
            return {"label": label_map.get(str(ex["gold"]).lower(), -1)}
        return {"label": int(ex["label"]) if ex.get("label") is not None else -1}

    wanli = wanli.map(_norm_label)
    wanli = wanli.filter(lambda ex: ex["label"] in (0, 1, 2)
                         and ex["premise"] is not None and ex["hypothesis"] is not None)

    def _tok_wanli(batch):
        out = _tokenize_pair(tok, batch, cfg.max_seq_length)
        out["label"] = batch["label"]
        return out

    return wanli.map(_tok_wanli, batched=True)

def make_wanli_test_loader(cfg: TrainConfig, tok):
    test_ds = _prepare_wanli_test_dataset(cfg, tok)
    cols = ["input_ids", "attention_mask", "label"]
    test_ds.set_format(type="torch", columns=cols)
    return DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

def _select_phase2_train_columns(ds: Dataset) -> Dataset:
    # Unify the schema of Phase 2 training data: only keep input columns and force label to Value("int64").
    ds = ds.map(lambda ex: {"label": int(ex["label"])})
    keep = {"input_ids", "attention_mask", "label"}
    drop = [col for col in ds.column_names if col not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds.cast_column("label", Value("int64"))

def _sample_fixed_count(ds: Dataset, n: int, seed: int) -> Dataset:
    # Fixed sampling of n items; automatically sample with replacement when n exceeds the dataset size.
    if n <= len(ds):
        return ds.shuffle(seed=seed).select(range(n))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ds), size=n).tolist()
    return ds.select(idx)

def make_phase2_biased_mixed_loader(cfg: TrainConfig, tok):
    # Phase 2 mixes MNLI and HANS entailment proportionally, with a fixed number of batches per epoch.
    total = int(cfg.phase2_epoch_batches) * int(cfg.batch_size)
    mnli_n = int(round(total * cfg.phase2_mnli_mix_ratio))
    hans_n = total - mnli_n

    parts = []

    if mnli_n > 0:
        mnli = load_dataset("nyu-mll/glue", "mnli")["train"]
        mnli = _sample_fixed_count(mnli, mnli_n, seed=cfg.seed + 611)
        mnli = mnli.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)
        mnli = _select_phase2_train_columns(mnli)
        mnli.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        parts.append(mnli)

    if hans_n > 0:
        hans = _prepare_hans_analysis_base(cfg)
        hans_ent = hans.filter(lambda ex: int(ex["label"]) == 0)
        hans_ent = _sample_fixed_count(hans_ent, hans_n, seed=cfg.seed + 712)
        hans_ent = hans_ent.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)
        hans_ent = _select_phase2_train_columns(hans_ent)
        hans_ent.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        parts.append(hans_ent)

    if not parts:
        raise ValueError("Phase2 mixed loader is empty. Check phase2_epoch_batches and ratio.")

    mixed = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    mixed = mixed.shuffle(seed=cfg.seed + 813)
    mixed.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return DataLoader(mixed, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

# ── Self-discovered lexical-overlap shortcut data (MNLI-derived, HANS-free) ──────
_OV_WORD_RE = re.compile(r"[a-z0-9]+")
# Drop function words so "overlap" reflects content-word coverage, like HANS lexical_overlap.
_OV_STOP = set(
    "a an the of to in on at for and or but is are was were be been being do does did "
    "this that these those with as by from it its his her their our your my no not".split()
)

def _lexical_overlap_score(premise: str, hypothesis: str) -> float:
    """Fraction of hypothesis content-words that also appear in the premise.
    Mirrors the HANS 'lexical_overlap' heuristic but computed on MNLI, so we can
    self-discover overlap-biased examples WITHOUT ever touching HANS."""
    p = {w for w in _OV_WORD_RE.findall((premise or "").lower()) if w not in _OV_STOP}
    h = [w for w in _OV_WORD_RE.findall((hypothesis or "").lower()) if w not in _OV_STOP]
    if not h:
        return 0.0
    return sum(1 for w in h if w in p) / len(h)

def make_phase2_overlap_biased_loader(cfg: TrainConfig, tok):
    """Phase-2 N-branch training data, self-discovered from MNLI (HANS-free).

    Builds a LABEL-BALANCED overlap-biased set so the N branch must learn the
    lexical-overlap *feature* instead of collapsing to a constant 'always-entailment'
    predictor (the failure the old 90%-HANS-entailment loader caused):
      * high-overlap + entailment      (overlap -> entailment, shortcut-aligned)
      * low-overlap  + non-entailment  (neutral / contradiction)
    The two groups are 1:1 balanced, so a majority-class shortcut gets no traction:
    the only way to fit the data is to key on overlap. HANS is never used here
    (leakage-free: HANS stays a strictly unseen test set).
    """
    total = int(cfg.phase2_epoch_batches) * int(cfg.batch_size)
    half = total // 2
    hi, lo = cfg.overlap_high_thresh, cfg.overlap_low_thresh

    mnli = load_dataset("nyu-mll/glue", "mnli")["train"]
    # Compute overlap on a shuffled pool large enough to yield both groups
    # (no need to score all ~390k examples).
    pool_n = min(len(mnli), max(total * 8, 80_000))
    pool = mnli.shuffle(seed=cfg.seed + 901).select(range(pool_n))
    pool = pool.map(
        lambda b: {"_ov": [_lexical_overlap_score(p, h)
                           for p, h in zip(b["premise"], b["hypothesis"])]},
        batched=True,
    )

    aligned_ent = pool.filter(lambda ex: ex["label"] == 0 and ex["_ov"] >= hi)
    aligned_non = pool.filter(lambda ex: ex["label"] in (1, 2) and ex["_ov"] <= lo)

    k = min(half, len(aligned_ent), len(aligned_non))
    if k == 0:
        raise ValueError(
            f"overlap-biased loader is empty (high-ov entailment={len(aligned_ent)}, "
            f"low-ov non-entailment={len(aligned_non)}); loosen overlap thresholds "
            f"or enlarge the pool.")
    if k < half:
        print(f"[Phase2] WARNING: only {k} examples/group available (< {half}); "
              f"using {2*k} total. Enlarge pool or loosen thresholds for more.")

    ent = aligned_ent.shuffle(seed=cfg.seed + 902).select(range(k))
    non = aligned_non.shuffle(seed=cfg.seed + 903).select(range(k))
    print(f"[Phase2] overlap-biased (MNLI, HANS-free): {k} high-ov entailment + "
          f"{k} low-ov non-entailment, balanced (hi>={hi}, lo<={lo})")

    mixed = concatenate_datasets([ent, non])
    mixed = mixed.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)
    mixed = _select_phase2_train_columns(mixed)
    mixed = mixed.shuffle(seed=cfg.seed + 904)
    mixed.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    loader = DataLoader(mixed, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    # Surfaced in metrics: a small k means N sees few unique examples and may
    # memorize them instead of learning the overlap feature.
    loader.overlap_group_size = k
    return loader

def make_overlap_diagnostic_loader(cfg: TrainConfig, tok, n_per_group: int = 1000):
    """HANS-free shortcut diagnostic set, built from MNLI validation_matched.

    Takes the `n_per_group` examples with the HIGHEST lexical-overlap score and
    the `n_per_group` with the LOWEST, keeping gold labels. Used after Phase 2
    to verify the N branch keys on the overlap FEATURE: a shortcut-aligned N
    predicts entailment on the high group (even where the gold label is
    non-entailment) and not on the low group, giving a large entailment-rate
    gap between groups. validation_matched is disjoint from the MNLI train pool
    that builds the Phase-2 data, and HANS is never touched.
    """
    val = load_dataset("nyu-mll/glue", "mnli")["validation_matched"]
    val = val.map(
        lambda b: {"_ov": [_lexical_overlap_score(p, h)
                           for p, h in zip(b["premise"], b["hypothesis"])]},
        batched=True,
    )
    order = np.argsort(np.asarray(val["_ov"]))
    n = min(n_per_group, len(val) // 2)
    low = val.select(order[:n].tolist()).map(lambda ex: {"ov_group": 0})
    high = val.select(order[-n:].tolist()).map(lambda ex: {"ov_group": 1})
    print(f"[Phase2-diag] overlap diagnostic: {n}/group, "
          f"mean ov low={float(np.mean(low['_ov'])):.2f} high={float(np.mean(high['_ov'])):.2f}")
    diag = concatenate_datasets([low, high])
    diag = diag.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)
    diag = diag.map(lambda ex: {"label": int(ex["label"])})
    diag.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "ov_group"])
    return DataLoader(diag, batch_size=cfg.batch_size, shuffle=False)

def _split_for_reference_and_query(ds: Dataset, ref_n: int, query_n: int, seed: int) -> Tuple[Dataset, Dataset]:
    # Shuffle first then slice to ensure reference/query are clearly separated and avoid information leakage.
    required = ref_n + query_n
    if required > len(ds):
        raise ValueError(
            f"Requested {required} examples but only {len(ds)} are available for analysis sampling."
        )
    shuffled = ds.shuffle(seed=seed)
    ref_ds = shuffled.select(range(ref_n))
    query_ds = shuffled.select(range(ref_n, ref_n + query_n))
    return ref_ds, query_ds

def _attach_analysis_fields(ds: Dataset, group_id: int) -> Dataset:
    # kNN analysis is uniformly processed as binary classification: entailment=0, non-entailment=1.
    def _map_fn(ex):
        label = int(ex["label"])
        return {
            "analysis_label": 0 if label == 0 else 1,
            "group_id": group_id,
        }
    return ds.map(_map_fn)

def _tokenize_analysis_dataset(ds: Dataset, tok, cfg: TrainConfig) -> Dataset:
    # Only tokenize the sampled analysis samples to avoid processing the entire large dataset for Phase 2.5.
    return ds.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)

def _select_analysis_columns(ds: Dataset) -> Dataset:
    # Only keep the fields truly needed for kNN analysis to avoid inconsistencies in the original label schema of MNLI/HANS.
    return ds.remove_columns([col for col in ds.column_names if col not in {
        "input_ids", "attention_mask", "analysis_label", "group_id"
    }])

def make_phase2_5_analysis_loaders(cfg: TrainConfig, tok):
    # Construct reference bank and query set as needed: sample MNLI / HANS entail / HANS non-entail separately and then merge.
    mnli = load_dataset("nyu-mll/glue", "mnli")["train"]
    mnli_ref_raw, mnli_query_raw = _split_for_reference_and_query(
        mnli,
        cfg.knn_ref_mnli,
        cfg.knn_query_mnli,
        seed=cfg.seed + 101,
    )
    # MNLI is mapped to binary classification in the analysis phase; neutral/contradiction are both merged into non-entailment.
    mnli_ref = _attach_analysis_fields(_tokenize_analysis_dataset(mnli_ref_raw, tok, cfg), group_id=0)
    mnli_query = _attach_analysis_fields(_tokenize_analysis_dataset(mnli_query_raw, tok, cfg), group_id=0)

    hans = _prepare_hans_analysis_base(cfg)
    hans_ent_raw = hans.filter(lambda ex: int(ex["label"]) == 0)
    hans_non_raw = hans.filter(lambda ex: int(ex["label"]) == 1)

    hans_ent_ref_raw, hans_ent_query_raw = _split_for_reference_and_query(
        hans_ent_raw,
        cfg.knn_ref_hans_entail,
        cfg.knn_query_hans_entail,
        seed=cfg.seed + 202,
    )
    hans_non_ref_raw, hans_non_query_raw = _split_for_reference_and_query(
        hans_non_raw,
        cfg.knn_ref_hans_non_entail,
        cfg.knn_query_hans_non_entail,
        seed=cfg.seed + 303,
    )

    hans_ent_ref = _attach_analysis_fields(_tokenize_analysis_dataset(hans_ent_ref_raw, tok, cfg), group_id=1)
    hans_ent_query = _attach_analysis_fields(_tokenize_analysis_dataset(hans_ent_query_raw, tok, cfg), group_id=1)
    hans_non_ref = _attach_analysis_fields(_tokenize_analysis_dataset(hans_non_ref_raw, tok, cfg), group_id=2)
    hans_non_query = _attach_analysis_fields(_tokenize_analysis_dataset(hans_non_query_raw, tok, cfg), group_id=2)

    # Align fields before merging, otherwise MNLI's ClassLabel and HANS's int label will conflict.
    mnli_ref = _select_analysis_columns(mnli_ref)
    mnli_query = _select_analysis_columns(mnli_query)
    hans_ent_ref = _select_analysis_columns(hans_ent_ref)
    hans_ent_query = _select_analysis_columns(hans_ent_query)
    hans_non_ref = _select_analysis_columns(hans_non_ref)
    hans_non_query = _select_analysis_columns(hans_non_query)

    # Finally, piece the three parts together into a unified reference/query loader for P-only / N-only to share the same batch of samples.
    ref_ds = concatenate_datasets([mnli_ref, hans_ent_ref, hans_non_ref]).shuffle(seed=cfg.seed + 404)
    query_ds = concatenate_datasets([mnli_query, hans_ent_query, hans_non_query]).shuffle(seed=cfg.seed + 505)

    cols = ["input_ids", "attention_mask", "analysis_label", "group_id"]
    ref_ds.set_format(type="torch", columns=cols)
    query_ds.set_format(type="torch", columns=cols)

    return (
        DataLoader(ref_ds, batch_size=cfg.knn_batch_size, shuffle=False),
        DataLoader(query_ds, batch_size=cfg.knn_batch_size, shuffle=False),
    )

def make_debias_datasets(cfg: TrainConfig, tok):
    """Shared data for the PoE / z-filtering bias-model baselines.

    Returns three tokenized HuggingFace datasets that all share the same row order
    and an explicit ``idx`` column, so per-example bias log-probs (computed on the
    hypothesis-only encoding, in order) can be gathered back onto the full-input
    training batches by index:

      * ``train_full`` : MNLI train, premise+hypothesis encoding, carries ``idx``.
      * ``val_full``   : MNLI validation-matched, premise+hypothesis encoding.
      * ``train_hyp``  : MNLI train, hypothesis-only encoding, carries ``idx``.

    ``train_full`` and ``train_hyp`` are derived from the *same* sampled base, so
    ``idx`` is consistent across the two encodings.
    """
    ds = load_dataset("nyu-mll/glue", "mnli")
    # Identical sampling to make_mnli_loaders so the comparison stays fair.
    train_base = _sample(ds["train"], cfg.mnli_train_size, cfg.seed)
    val_base = _sample(ds["validation_matched"], cfg.mnli_val_size, cfg.seed)
    # GLUE MNLI ships a native (non-contiguous) ``idx`` column; drop it before
    # adding our own contiguous 0..N-1 index, otherwise Arrow raises
    # "columns['idx'] are duplicated". The contiguous index must match row order
    # so per-example bias log-probs can be gathered via ``bias_logprobs[idx]``.
    if "idx" in train_base.column_names:
        train_base = train_base.remove_columns("idx")
    train_base = train_base.add_column("idx", list(range(len(train_base))))

    train_full = train_base.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)
    val_full = val_base.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)
    train_hyp = train_base.map(lambda b: _tokenize_hypothesis_only(tok, b, cfg.max_seq_length), batched=True)

    full_cols = ["input_ids", "attention_mask", "label", "idx"]
    train_full.set_format(type="torch", columns=full_cols)
    train_hyp.set_format(type="torch", columns=full_cols)
    val_full.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return {"train_full": train_full, "val_full": val_full, "train_hyp": train_hyp}
