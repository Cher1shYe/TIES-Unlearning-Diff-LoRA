"""Data-centric debiasing baseline via bias-model example filtering ("z-filtering").

Reviewer item #7: "z-filtering or data-centric debiasing".

Pipeline:
  1. Train a hypothesis-only bias model.
  2. Score every training example by the bias model's probability of the *true*
     label, p_bias(y | x). High values mark "bias-aligned / easy" examples the
     shortcut already solves.
  3. Drop the top `zfilter_drop_ratio` fraction of the most bias-aligned examples
     and train a standard single-path LoRA model on the remaining (harder) data.
  4. Evaluate the filtered model on MNLI / HANS / e-SNLI.
"""
import os
import json
import torch
from torch.utils.data import DataLoader

from configs.config import TrainConfig
from data.dataloader import (
    set_seed, make_debias_datasets, make_hans_loader, make_esnli_test_loader,
    make_anli_test_loader, make_snli_hard_test_loader,
)
from training.bias_utils import (
    build_single_lora_model, train_bias_model, compute_logprobs,
    make_optimizer_and_schedule, run_training_loop,
)
from utils.optim_utils import _make_scaler
from training.evaluate import eval_mnli, eval_hans, eval_esnli, eval_anli, eval_snli_hard
from transformers import AutoTokenizer


def train_zfilter_baseline(cfg: TrainConfig):
    set_seed(cfg.training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[z-filter Baseline] device={device}")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    data = make_debias_datasets(cfg, tok)
    hans_loader = make_hans_loader(cfg, tok)
    esnli_loader = make_esnli_test_loader(cfg, tok)
    anli_loader = make_anli_test_loader(cfg, tok)
    snli_hard_loader = make_snli_hard_test_loader(cfg, tok)

    # --- 1) bias model ---
    hyp_train_loader = DataLoader(data["train_hyp"], batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    bias_model = train_bias_model(cfg, hyp_train_loader, device)

    # --- 2) score examples by bias prob of the true label ---
    hyp_ordered_loader = DataLoader(data["train_hyp"], batch_size=cfg.batch_size, shuffle=False)
    bias_logprobs = compute_logprobs(bias_model, hyp_ordered_loader, device)  # [N, num_labels] CPU
    del bias_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # datasets>=4.0 returns a lazy ``Column`` from ds["label"] (no .numel()); older
    # versions return a torch tensor. torch.as_tensor normalizes both to a 1-D long
    # tensor, aligned with bias_logprobs row order.
    labels = torch.as_tensor(data["train_hyp"]["label"]).long().view(-1)
    n_total = labels.numel()
    p_true = bias_logprobs.exp()[torch.arange(n_total), labels]  # [N]

    # --- 3) drop the most bias-aligned examples ---
    n_drop = int(round(cfg.zfilter_drop_ratio * n_total))
    if n_drop > 0:
        drop_idx = torch.topk(p_true, n_drop).indices
        keep_mask = torch.ones(n_total, dtype=torch.bool)
        keep_mask[drop_idx] = False
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).tolist()
    else:
        keep_indices = list(range(n_total))
    print(f"[z-filter] dropped {n_drop}/{n_total} bias-aligned examples; "
          f"training on {len(keep_indices)} examples.")

    filtered_train = data["train_full"].select(keep_indices)

    # --- 4) train standard LoRA on the filtered set ---
    model = build_single_lora_model(cfg, device)
    total_epochs = cfg.phase1_epochs + cfg.phase2_epochs + cfg.phase3_epochs
    train_loader = DataLoader(filtered_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    total_steps = total_epochs * len(train_loader)
    opt, sch = make_optimizer_and_schedule(model, cfg, total_steps)
    scaler = _make_scaler(cfg.fp16, device)

    print(f"\n[z-filter Baseline] Training filtered model ({total_epochs} epochs)")
    run_training_loop(model, opt, sch, scaler, train_loader, cfg, device,
                      total_epochs, "z-filter")

    val_loader = DataLoader(data["val_full"], batch_size=cfg.batch_size, shuffle=False)
    zf_mnli = eval_mnli(model, val_loader, device)
    zf_hans = eval_hans(model, hans_loader, device)
    zf_esnli = eval_esnli(model, esnli_loader, device)
    zf_anli = eval_anli(model, anli_loader, device)
    zf_snli_hard = eval_snli_hard(model, snli_hard_loader, device)

    metrics = {
        "method": "z-filtering (data-centric)",
        "mnli": zf_mnli,
        "hans": zf_hans,
        "esnli": zf_esnli,
        "anli": zf_anli,
        "snli_hard": zf_snli_hard,
        "n_dropped": n_drop,
        "n_kept": len(keep_indices),
    }
    run_dir = os.path.join(cfg.output_dir, "zfilter_baseline")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  MNLI Acc: {zf_mnli['mnli_accuracy']:.4f} | "
          f"HANS: {zf_hans['hans_overall']:.4f} (nent={zf_hans['hans_non_entailment']:.4f}) | "
          f"ESNLI: {zf_esnli['esnli_accuracy']:.4f}")
    return metrics
