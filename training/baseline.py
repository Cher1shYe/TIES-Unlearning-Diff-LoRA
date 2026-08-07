import os
import json
import torch
from dataclasses import asdict
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Compatibility: torch.cuda.amp location changed in newer PyTorch
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from configs.config import TrainConfig, LoRAConfig
from data.dataloader import (
    set_seed, make_mnli_loaders, make_hans_evaluation_loader, make_esnli_test_loader,
    make_anli_test_loader, make_snli_hard_test_loader,
)
from models.surgery import inject_ties_unlearn_lora
from utils.optim_utils import _split_params, _make_scaler, _amp_enabled
from training.evaluate import eval_mnli, eval_hans, eval_esnli, eval_anli, eval_snli_hard
from canonical.artifacts import sha256_file, write_json, write_jsonl

def train_single_lora_baseline(cfg: TrainConfig, *, method_tag=None):
    """
    Standard single LoRA training (no N path, no TIES merge).
    Uses the same pos_rank so the comparison is fair.
    """
    set_seed(cfg.training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Baseline] device={device}")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    train_loader, val_loader = make_mnli_loaders(cfg, tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels,
    )

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        # 匹配 Roberta 的分类头 (通常叫 classifier)
        if "classifier" in name or "score" in name:
            param.requires_grad = True

    # Inject single-path LoRA (beta=0 ⇒ N path has zero effect)
    lora_cfg = LoRAConfig(
        pos_rank=cfg.pos_rank, neg_rank=cfg.neg_rank,
        alpha=cfg.alpha, beta=0.0,          # <-- no subtraction
        lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        trim_ratio=cfg.trim_ratio,
    )
    model, _replaced = inject_ties_unlearn_lora(model, list(cfg.target_modules), lora_cfg)
    model.to(device)

    # Freeze N entirely — only train P + head
    for n, p in model.named_parameters():
        if "lora_N_" in n:
            p.requires_grad = False

    pos_params, _, head_params = _split_params(model)
    total_epochs = cfg.phase1_epochs + cfg.phase2_epochs + cfg.phase3_epochs
    opt = AdamW([
        {"params": head_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        {"params": pos_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
    ])
    total_steps = total_epochs * len(train_loader)
    sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
    scaler = _make_scaler(cfg.fp16, device)

    for ep in range(total_epochs):
        model.train()
        ep_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Baseline Ep {ep+1}/{total_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            if "label" in batch:
                batch["labels"] = batch.pop("label")
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=_amp_enabled(cfg.fp16, device)):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sch.step()
            ep_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = ep_loss / max(n_batch, 1)
        val = eval_mnli(model, val_loader, device)["mnli_accuracy"]
        print(f"  Baseline Ep{ep+1}: loss={avg:.4f} val_acc={val:.4f}")

    run_dir = os.path.join(cfg.output_dir, cfg.experiment_name if method_tag else "baseline_single_lora")
    os.makedirs(run_dir, exist_ok=True)
    final_checkpoint_hash = None
    if method_tag is not None:
        state_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(state_dir, exist_ok=True)
        final_state_path = os.path.join(state_dir, "final_model_state.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "method_tag": method_tag,
                "training_seed": cfg.training_seed,
            },
            final_state_path,
        )
        final_checkpoint_hash = sha256_file(final_state_path)

    hans_loader = make_hans_evaluation_loader(cfg, tok)
    esnli_loader = make_esnli_test_loader(cfg, tok)
    anli_loader = make_anli_test_loader(cfg, tok)
    snli_hard_loader = make_snli_hard_test_loader(cfg, tok)
    bl_mnli = eval_mnli(model, val_loader, device)
    hans_predictions = None
    if method_tag is not None:
        bl_hans, hans_predictions = eval_hans(
            model,
            hans_loader,
            device,
            prediction_context={
                "training_seed": cfg.training_seed,
                "method_tag": method_tag,
                "checkpoint_hash": final_checkpoint_hash,
            },
        )
    else:
        bl_hans = eval_hans(model, hans_loader, device)
    bl_esnli = eval_esnli(model, esnli_loader, device)
    bl_anli = eval_anli(model, anli_loader, device)
    bl_snli_hard = eval_snli_hard(model, snli_hard_loader, device)

    metrics = {
        "method": "Single LoRA Baseline",
        "mnli": bl_mnli,
        "hans": bl_hans,
        "esnli": bl_esnli,
        "anli": bl_anli,
        "snli_hard": bl_snli_hard,
        "config": asdict(cfg),
        "checkpoint_provenance": {
            "source_phase2_checkpoint_hash": None,
            "final_checkpoint_hash": final_checkpoint_hash,
        },
    }

    if method_tag is not None:
        write_json(os.path.join(run_dir, "metrics.json"), metrics)
        write_jsonl(os.path.join(run_dir, "hans_predictions.jsonl"), hans_predictions)
        write_json(os.path.join(run_dir, "selected_layers.json"), {})
    else:
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics
