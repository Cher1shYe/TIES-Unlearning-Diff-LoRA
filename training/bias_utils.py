"""Shared building blocks for the bias-model debiasing baselines (PoE, z-filtering).

Both PoE and z-filtering rely on a *bias model* — here a hypothesis-only RoBERTa —
that captures the dataset shortcuts a hypothesis-alone classifier can exploit. This
module provides:

  * build_single_lora_model : a single-path LoRA RoBERTa (no N branch, no subtraction),
                              reused for both the bias model and the main model.
  * run_training_loop       : a generic AMP training loop with an optional custom loss.
  * compute_logprobs        : ordered per-example log-probabilities (for PoE / z-filter).

Keeping these here avoids duplicating the Phase-1 training boilerplate across the
three baseline drivers.
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from configs.config import TrainConfig, LoRAConfig
from models.surgery import inject_ties_unlearn_lora
from utils.optim_utils import _split_params, _make_scaler, _amp_enabled

# Keys that may live in a batch dict but must not be forwarded to the model.
_NON_MODEL_KEYS = ("label", "idx")


def build_single_lora_model(cfg: TrainConfig, device):
    """Standard single-path LoRA model (beta=0 ⇒ N branch inert, only P + head train).

    Identical in spirit to training/baseline.py's setup; factored out so the bias
    model and the PoE / z-filter main models share one definition.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels,
    )
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "classifier" in name or "score" in name:
            p.requires_grad = True

    lora_cfg = LoRAConfig(
        pos_rank=cfg.pos_rank, neg_rank=cfg.neg_rank,
        alpha=cfg.alpha, beta=0.0,
        lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        trim_ratio=cfg.trim_ratio, merge_mode="p_only",
    )
    model, _ = inject_ties_unlearn_lora(model, list(cfg.target_modules), lora_cfg)
    model.to(device)
    for n, p in model.named_parameters():
        if "lora_N_" in n:
            p.requires_grad = False
    return model


def make_optimizer_and_schedule(model, cfg: TrainConfig, total_steps: int, lr: float = None):
    lr = cfg.learning_rate if lr is None else lr
    pos_params, _, head_params = _split_params(model)
    opt = AdamW([
        {"params": head_params, "lr": lr, "weight_decay": cfg.weight_decay},
        {"params": pos_params, "lr": lr, "weight_decay": cfg.weight_decay},
    ])
    sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
    return opt, sch


def run_training_loop(model, opt, sch, scaler, train_loader, cfg, device, epochs, desc, loss_fn=None):
    """Generic training loop.

    If ``loss_fn`` is None the standard ``model(**inputs).loss`` (cross-entropy) is used.
    Otherwise ``loss_fn(model, batch)`` must return a scalar loss; the batch tensors are
    already moved to ``device``. Used by PoE to inject the bias log-probs.
    """
    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"{desc} Ep {ep + 1}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=_amp_enabled(cfg.fp16, device)):
                if loss_fn is None:
                    inputs = {k: v for k, v in batch.items() if k not in _NON_MODEL_KEYS}
                    inputs["labels"] = batch["label"]
                    loss = model(**inputs).loss
                else:
                    loss = loss_fn(model, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sch.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")


def train_bias_model(cfg: TrainConfig, hyp_train_loader, device):
    """Train the hypothesis-only bias model for cfg.bias_model_epochs."""
    print(f"\n[Bias model] Training hypothesis-only RoBERTa ({cfg.bias_model_epochs} epochs)")
    model = build_single_lora_model(cfg, device)
    total_steps = cfg.bias_model_epochs * len(hyp_train_loader)
    opt, sch = make_optimizer_and_schedule(model, cfg, total_steps)
    scaler = _make_scaler(cfg.fp16, device)
    run_training_loop(model, opt, sch, scaler, hyp_train_loader, cfg, device,
                      cfg.bias_model_epochs, "BiasModel")
    return model


@torch.no_grad()
def compute_logprobs(model, ordered_loader, device):
    """Return per-example log-softmax probabilities [N, num_labels], in loader order.

    The loader must NOT be shuffled so row i corresponds to example idx i.
    """
    model.eval()
    chunks = []
    for batch in tqdm(ordered_loader, desc="Bias log-probs"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        chunks.append(F.log_softmax(logits, dim=-1).cpu())
    return torch.cat(chunks, dim=0)
