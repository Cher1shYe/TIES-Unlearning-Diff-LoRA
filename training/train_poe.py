"""Product-of-Experts (PoE) debiasing baseline.

Reviewer item #7: "product-of-experts or bias-model methods".

Pipeline (Clark et al., 2019, "Don't Take the Easy Way Out"):
  1. Train a hypothesis-only bias model b(x) that captures dataset shortcuts.
  2. Pre-compute b's per-example log-probabilities over the MNLI train set.
  3. Train the main model f(x) on premise+hypothesis with the PoE objective
        loss = CE( log_softmax(f(x)) + scale * log b(x),  y )
     so f is encouraged to focus on examples the bias model gets wrong.
  4. At test time use f(x) alone (bias model discarded).

The main model is a standard single-path LoRA RoBERTa, matching the other
baselines for a fair parameter/compute comparison.
"""
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import TrainConfig
from data.dataloader import (
    set_seed, make_debias_datasets, make_hans_loader, make_esnli_test_loader,
    make_anli_test_loader, make_snli_hard_test_loader, make_wanli_test_loader,
)
from training.bias_utils import (
    build_single_lora_model, train_bias_model, compute_logprobs,
    make_optimizer_and_schedule, run_training_loop,
)
from utils.optim_utils import _make_scaler
from training.evaluate import eval_mnli, eval_hans, eval_esnli, eval_anli, eval_snli_hard, eval_wanli
from transformers import AutoTokenizer


def train_poe_baseline(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[PoE Baseline] device={device}")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    data = make_debias_datasets(cfg, tok)
    hans_loader = make_hans_loader(cfg, tok)
    esnli_loader = make_esnli_test_loader(cfg, tok)
    anli_loader = make_anli_test_loader(cfg, tok)
    snli_hard_loader = make_snli_hard_test_loader(cfg, tok)
    try:
        wanli_loader = make_wanli_test_loader(cfg, tok)
    except Exception as e:
        print(f"[PoE] WANLI load failed ({e}); WANLI skipped.")
        wanli_loader = None

    # --- 1) bias model on hypothesis-only inputs ---
    hyp_train_loader = DataLoader(data["train_hyp"], batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    bias_model = train_bias_model(cfg, hyp_train_loader, device)

    # --- 2) precompute ordered bias log-probs over the train set ---
    hyp_ordered_loader = DataLoader(data["train_hyp"], batch_size=cfg.batch_size, shuffle=False)
    bias_logprobs = compute_logprobs(bias_model, hyp_ordered_loader, device)  # [N, num_labels] on CPU
    bias_logprobs = bias_logprobs.to(device)
    del bias_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 3) train main model with the PoE objective ---
    main_model = build_single_lora_model(cfg, device)
    total_epochs = cfg.phase1_epochs + cfg.phase2_epochs + cfg.phase3_epochs
    train_loader = DataLoader(data["train_full"], batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    total_steps = total_epochs * len(train_loader)
    opt, sch = make_optimizer_and_schedule(main_model, cfg, total_steps)
    scaler = _make_scaler(cfg.fp16, device)

    bias_scale = cfg.poe_bias_scale

    def poe_loss_fn(model, batch):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        # Combine in log space: log f + scale * log b  (acts as PoE logits).
        combined = F.log_softmax(logits, dim=-1) + bias_scale * bias_logprobs[batch["idx"]]
        return F.cross_entropy(combined, batch["label"])

    print(f"\n[PoE Baseline] Training main model ({total_epochs} epochs, bias_scale={bias_scale})")
    run_training_loop(main_model, opt, sch, scaler, train_loader, cfg, device,
                      total_epochs, "PoE-main", loss_fn=poe_loss_fn)

    # --- 4) evaluate the main model alone ---
    val_loader = DataLoader(data["val_full"], batch_size=cfg.batch_size, shuffle=False)
    poe_mnli = eval_mnli(main_model, val_loader, device)
    poe_hans = eval_hans(main_model, hans_loader, device)
    poe_esnli = eval_esnli(main_model, esnli_loader, device)
    poe_anli = eval_anli(main_model, anli_loader, device)
    poe_snli_hard = eval_snli_hard(main_model, snli_hard_loader, device)
    poe_wanli = (eval_wanli(main_model, wanli_loader, device)
                 if wanli_loader is not None else {"wanli_accuracy": float("nan")})

    metrics = {
        "method": "PoE (hypothesis-only bias)",
        "mnli": poe_mnli,
        "hans": poe_hans,
        "esnli": poe_esnli,
        "anli": poe_anli,
        "snli_hard": poe_snli_hard,
        "wanli": poe_wanli,
    }
    run_dir = os.path.join(cfg.output_dir, "poe_baseline")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  MNLI Acc: {poe_mnli['mnli_accuracy']:.4f} | "
          f"HANS: {poe_hans['hans_overall']:.4f} (nent={poe_hans['hans_non_entailment']:.4f}) | "
          f"ESNLI: {poe_esnli['esnli_accuracy']:.4f}")
    return metrics
