import os
import json
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

# Import project configurations and utilities
from configs.config import TrainConfig, LoRAConfig
from data.dataloader import (
    set_seed, make_mnli_loaders, make_hans_loader, make_esnli_test_loader,
    make_anli_test_loader, make_snli_hard_test_loader, make_wanli_test_loader,
)
from models.surgery import inject_ties_unlearn_lora
from utils.optim_utils import _split_params, _make_scaler, _amp_enabled
from training.evaluate import eval_mnli, eval_hans, eval_esnli, eval_anli, eval_snli_hard, eval_wanli

def setup_single_lora_model(cfg: TrainConfig, device):
    """Helper function: Initialize a standard single-LoRA model and its corresponding optimizer."""
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels,
    )

    # Freeze all base parameters; only unfreeze the classification head
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "classifier" in name or "score" in name:
            param.requires_grad = True

    # Inject single-path LoRA (beta=0 means the N-path is inactive, degrading to a standard LoRA)
    lora_cfg = LoRAConfig(
        pos_rank=cfg.pos_rank, neg_rank=cfg.neg_rank,
        alpha=cfg.alpha, beta=0.0,          
        lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        trim_ratio=cfg.trim_ratio,
    )
    model, _ = inject_ties_unlearn_lora(model, list(cfg.target_modules), lora_cfg)
    model.to(device)

    # Freeze the N-path to ensure only the P-path is trained, matching the standard baseline training process
    for n, p in model.named_parameters():
        if "lora_N_" in n:
            p.requires_grad = False

    # Configure optimizer, directly reusing baseline code structure
    pos_params, _, head_params = _split_params(model)
    opt = AdamW([
        {"params": head_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        {"params": pos_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
    ])
    
    return model, opt

def train_jtt_baseline(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[JTT Baseline] device={device}")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    # Modified make_mnli_loaders to return the dataset
    train_loader, val_loader, train_dataset = make_mnli_loaders(cfg, tok, return_dataset=True)
    hans_loader = make_hans_loader(cfg, tok)
    esnli_loader = make_esnli_test_loader(cfg, tok)
    anli_loader = make_anli_test_loader(cfg, tok)
    snli_hard_loader = make_snli_hard_test_loader(cfg, tok)
    try:
        wanli_loader = make_wanli_test_loader(cfg, tok)
    except Exception as e:
        print(f"[JTT] WANLI load failed ({e}); WANLI skipped.")
        wanli_loader = None

    # ──────────────────────────────────────────────────────────
    # [Phase 1] Train the temporary identification model, matched to phase1_epochs from unlearn config
    # ──────────────────────────────────────────────────────────
    print(f"\n>>> [Phase 1] ID-Model Training ({cfg.jtt_phase1_epochs} Epochs)")
    # Note: Added jtt_phase1_epochs to the parameter list. Ideally, this should be unified 
    # with the standard method's parameters. Manual adjustment may be needed for future JTT baseline reruns.

    model_id, opt_id = setup_single_lora_model(cfg, device)
    
    total_steps_id = cfg.jtt_phase1_epochs * len(train_loader)
    sch_id = get_linear_schedule_with_warmup(opt_id, int(cfg.warmup_ratio * total_steps_id), total_steps_id)
    scaler_id = _make_scaler(cfg.fp16, device)

    for ep in range(cfg.jtt_phase1_epochs):
        model_id.train()
        pbar = tqdm(train_loader, desc=f"P1 Ep {ep+1}/{cfg.jtt_phase1_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            if "label" in batch: batch["labels"] = batch.pop("label")
            opt_id.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=_amp_enabled(cfg.fp16, device)):
                loss = model_id(**batch).loss
            scaler_id.scale(loss).backward()
            scaler_id.unscale_(opt_id)
            torch.nn.utils.clip_grad_norm_(model_id.parameters(), cfg.max_grad_norm)
            scaler_id.step(opt_id)
            scaler_id.update()
            sch_id.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # ──────────────────────────────────────────────────────────
    # [Phase 2] Extract the error set (misclassified samples)
    # ──────────────────────────────────────────────────────────
    print("\n>>> [Phase 2] Extracting Error Set...")
    model_id.eval()
    eval_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False)
    error_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Finding Errors")):
            start_idx = batch_idx * cfg.batch_size
            batch = {k: v.to(device) for k, v in batch.items()}
            if "label" in batch: batch["labels"] = batch.pop("label")
            
            with autocast(device_type=device.type, enabled=_amp_enabled(cfg.fp16, device)):
                logits = model_id(**batch).logits
            preds = logits.argmax(dim=-1)
            wrong_mask = (preds != batch["labels"]).cpu()
            
            for i, is_wrong in enumerate(wrong_mask):
                if is_wrong:
                    error_indices.append(start_idx + i)
                    
    print(f"  --> Total train data: {len(train_dataset)}, Errors found: {len(error_indices)}")

    # ──────────────────────────────────────────────────────────
    # [Phase 3] Train the final robust model (e.g., 5 Epochs)
    # ──────────────────────────────────────────────────────────
    print(f"\n>>> [Phase 3] Robust Model Training (Upweight={cfg.jtt_upweight_factor})")
    error_dataset = Subset(train_dataset, error_indices)
    jtt_train_dataset = ConcatDataset([train_dataset] + [error_dataset] * cfg.jtt_upweight_factor)
    jtt_train_loader = DataLoader(jtt_train_dataset, batch_size=cfg.batch_size, shuffle=True)
    
    del model_id, opt_id, sch_id, scaler_id
    torch.cuda.empty_cache()
    
    # Initialize a brand new single-LoRA model
    model_robust, opt_robust = setup_single_lora_model(cfg, device)
    
    total_steps_robust = cfg.phase3_epochs * len(jtt_train_loader)
    sch_robust = get_linear_schedule_with_warmup(opt_robust, int(cfg.warmup_ratio * total_steps_robust), total_steps_robust)
    scaler_robust = _make_scaler(cfg.fp16, device)

    for ep in range(cfg.phase3_epochs):
        model_robust.train()
        pbar = tqdm(jtt_train_loader, desc=f"P3 Ep {ep+1}/{cfg.phase3_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            if "label" in batch: batch["labels"] = batch.pop("label")
            opt_robust.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=_amp_enabled(cfg.fp16, device)):
                loss = model_robust(**batch).loss
            scaler_robust.scale(loss).backward()
            scaler_robust.unscale_(opt_robust)
            torch.nn.utils.clip_grad_norm_(model_robust.parameters(), cfg.max_grad_norm)
            scaler_robust.step(opt_robust)
            scaler_robust.update()
            sch_robust.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # ──────────────────────────────────────────────────────────
    # [Final] Evaluate and save
    # ──────────────────────────────────────────────────────────
    print("\n>>> [Final] Evaluating JTT Model...")
    jtt_mnli = eval_mnli(model_robust, val_loader, device)
    jtt_hans = eval_hans(model_robust, hans_loader, device)
    jtt_esnli = eval_esnli(model_robust, esnli_loader, device)
    jtt_anli = eval_anli(model_robust, anli_loader, device)
    jtt_snli_hard = eval_snli_hard(model_robust, snli_hard_loader, device)
    jtt_wanli = (eval_wanli(model_robust, wanli_loader, device)
                 if wanli_loader is not None else {"wanli_accuracy": float("nan")})

    metrics = {
        "method": "JTT Baseline",
        "mnli": jtt_mnli,
        "hans": jtt_hans,
        "esnli": jtt_esnli,
        "anli": jtt_anli,
        "snli_hard": jtt_snli_hard,
        "wanli": jtt_wanli,
    }
    
    run_dir = os.path.join(cfg.output_dir, "jtt_baseline")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    print(f"  MNLI Acc: {jtt_mnli['mnli_accuracy']:.4f} | HANS: {jtt_hans['hans_overall']:.4f} | ESNLI: {jtt_esnli['esnli_accuracy']:.4f}")
    return metrics
