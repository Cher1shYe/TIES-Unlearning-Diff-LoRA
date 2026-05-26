import random
import numpy as np
import csv
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

def _sample(ds: Dataset, n: int, seed: int) -> Dataset:
    return ds.shuffle(seed=seed).select(range(n)) if n and n < len(ds) else ds

def _load_hans_dataset():
    hans_url = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
    try:
        hans = load_dataset("csv", data_files=hans_url, delimiter="\t", split="train")
    except Exception:
        hans = load_dataset("csv", data_files="./heuristics_evaluation_set.txt",
                            delimiter="\t", split="train")
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
    hans = _load_hans_dataset()

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

def _prepare_hans_analysis_base() -> Dataset:
    # The analysis phase only needs premise/hypothesis/label, not the heuristic field.
    hans = _load_hans_dataset()
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
    ds = load_dataset("glue", "mnli")
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
        mnli = load_dataset("glue", "mnli")["train"]
        mnli = _sample_fixed_count(mnli, mnli_n, seed=cfg.seed + 611)
        mnli = mnli.map(lambda b: _tokenize_pair(tok, b, cfg.max_seq_length), batched=True)
        mnli = _select_phase2_train_columns(mnli)
        mnli.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        parts.append(mnli)

    if hans_n > 0:
        hans = _prepare_hans_analysis_base()
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
    mnli = load_dataset("glue", "mnli")["train"]
    mnli_ref_raw, mnli_query_raw = _split_for_reference_and_query(
        mnli,
        cfg.knn_ref_mnli,
        cfg.knn_query_mnli,
        seed=cfg.seed + 101,
    )
    # MNLI is mapped to binary classification in the analysis phase; neutral/contradiction are both merged into non-entailment.
    mnli_ref = _attach_analysis_fields(_tokenize_analysis_dataset(mnli_ref_raw, tok, cfg), group_id=0)
    mnli_query = _attach_analysis_fields(_tokenize_analysis_dataset(mnli_query_raw, tok, cfg), group_id=0)

    hans = _prepare_hans_analysis_base()
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
