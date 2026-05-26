import os
import json
import torch
from dataclasses import asdict
from typing import Dict, List, Optional

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
    set_seed, make_mnli_loaders, make_hans_loader, 
    make_phase2_biased_mixed_loader, make_phase2_5_analysis_loaders,
    make_esnli_test_loader
)
from models.surgery import inject_ties_unlearn_lora, get_ties_modules_by_layer, configure_ties_layers, merge_and_unload
from models.ties_lora import set_model_forward_mode
from models.analyzer import analyze_shortcut_layers
from utils.optim_utils import _split_params, _make_scaler, _amp_enabled, _log_lora_norms, _save_checkpoint, _load_checkpoint
from training.evaluate import eval_mnli, eval_hans, eval_esnli


def train_ties_unlearn(cfg: TrainConfig, resume_from_checkpoint_path: Optional[str] = None):
    """
    Three-phase TIES-Unlearning Diff-LoRA training.
    Returns a dict of metrics for final comparison.
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TIES-Unlearn] device={device}")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    train_loader, val_loader = make_mnli_loaders(cfg, tok)
    hans_loader = make_hans_loader(cfg, tok)
    phase2_train_loader = make_phase2_biased_mixed_loader(cfg, tok)
    esnli_loader = make_esnli_test_loader(cfg, tok)

    # --- build model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels,
    )

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        # Match the classification head parameters.
        if "classifier" in name or "score" in name:
            param.requires_grad = True

    lora_cfg = LoRAConfig(
        pos_rank=cfg.pos_rank, neg_rank=cfg.neg_rank,
        alpha=cfg.alpha, beta=cfg.beta,
        lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        trim_ratio=cfg.trim_ratio,
    )
    model, replaced = inject_ties_unlearn_lora(model, list(cfg.target_modules), lora_cfg)
    model.to(device)
    print(f"[TIES-Unlearn] injected {len(replaced)} layers: {replaced[:4]}...")

    # Cache the classifier head parameter names so each phase can freeze/unfreeze precisely.
    initial_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    head_names = [n for n in initial_trainable if "lora_" not in n]

    history: Dict[str, List] = {"train_loss": [], "val_acc": [], "phase": []}
    metrics = {
        "method": "layerwise knn Diff-LoRA with biased N",
        "config": asdict(cfg),
        "history": history,
        "phase1": {}, "phase2": {}, "phase2_5": {}, "phase3": {}
    }

    start_phase = 1

    if resume_from_checkpoint_path:
        # Dummy optimizer and scheduler for loading state_dict. Actual ones will be recreated.
        dummy_opt = AdamW(model.parameters(), lr=1e-3)
        dummy_sch = get_linear_schedule_with_warmup(dummy_opt, 1, 2)

        loaded_epoch, loaded_phase, loaded_history, loaded_phase_metrics = _load_checkpoint(
            model, dummy_opt, dummy_sch, resume_from_checkpoint_path, device
        )
        start_phase = loaded_phase + 1
        history = loaded_history
        if "phase1" in loaded_phase_metrics:
            metrics["phase1"] = loaded_phase_metrics["phase1"]
        if "phase2" in loaded_phase_metrics:
            metrics["phase2"] = loaded_phase_metrics["phase2"]

        print(f"[TIES-Unlearn] Resuming from Phase {loaded_phase} (Epoch {loaded_epoch}). Starting training from Phase {start_phase}.")

    # │────────────────────────────────────────────────────────────────────│
    # Phase 1 — Learn Task: train head + P (N frozen)
    # │────────────────────────────────────────────────────────────────────│
    if start_phase <= 1:
        print("\n" + "=" * 60)
        print("Phase 1: Learn Task (head + P, N frozen)")
        print("=" * 60)
        set_model_forward_mode(model, "phase1")

        # Freeze N, unfreeze P and head.
        for n, p in model.named_parameters():
            if "lora_P_" in n:
                p.requires_grad = True
            elif "lora_N_" in n:
                p.requires_grad = False
            elif n in head_names:
                p.requires_grad = True

        pos_params, _, head_params = _split_params(model)
        opt = AdamW([
            {"params": head_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
            {"params": pos_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        ])
        total_steps = cfg.phase1_epochs * len(train_loader)
        sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
        scaler = _make_scaler(cfg.fp16, device)

        for ep in range(cfg.phase1_epochs):
            model.train()
            ep_loss, n_batch = 0.0, 0
            pbar = tqdm(train_loader, desc=f"P1 Ep {ep+1}/{cfg.phase1_epochs}")
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
            history["train_loss"].append(avg)
            history["val_acc"].append(val)
            history["phase"].append(1)
            print(f"  Phase1 Ep{ep+1}: loss={avg:.4f} val_acc={val:.4f}")

        p1_mnli = eval_mnli(model, val_loader, device)
        p1_hans = eval_hans(model, hans_loader, device)
        p1_esnli = eval_esnli(model, esnli_loader, device)
        print(f"  Phase1 HANS: overall={p1_hans['hans_overall']:.4f} "
              f"ent={p1_hans['hans_entailment']:.4f} "
              f"non-ent={p1_hans['hans_non_entailment']:.4f}")
        _log_lora_norms(model)
        metrics["phase1"] = {"mnli": p1_mnli, "hans": p1_hans, "esnli": p1_esnli}

        if cfg.save_checkpoints_per_phase:
            phase1_metrics_to_save = {"mnli": p1_mnli, "hans": p1_hans, "esnli": p1_esnli}
            if cfg.checkpoint_dir:
                checkpoint_path_p1 = os.path.join(cfg.checkpoint_dir, f"phase1_checkpoint_epoch{cfg.phase1_epochs}.pt")
            else:
                run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
                checkpoint_path_p1 = os.path.join(run_dir, "checkpoints", f"phase1_checkpoint_epoch{cfg.phase1_epochs}.pt")
            _save_checkpoint(model, opt, sch, cfg.phase1_epochs, 1, checkpoint_path_p1, history, phase1_metrics_to_save)

    # │────────────────────────────────────────────────────────────────────│
    # Phase 2 — Capture Shortcut: train N (P & head frozen)
    # │────────────────────────────────────────────────────────────────────│
    if start_phase <= 2:
        print("\n" + "=" * 60)
        print("Phase 2: Capture Shortcut on mixed data (N only, P & head frozen)")
        print("=" * 60)
        print(f"[Phase2] mix ratio -> MNLI: {cfg.phase2_mnli_mix_ratio:.2%}, HANS-entailment: {1.0-cfg.phase2_mnli_mix_ratio:.2%}; epoch_batches={cfg.phase2_epoch_batches}")
        set_model_forward_mode(model, "phase2")

        # Unfreeze N, freeze P and head so N must fit within the existing head space.
        for n, p in model.named_parameters():
            if "lora_P_" in n:
                p.requires_grad = False
            elif "lora_N_" in n:
                p.requires_grad = True
            elif n in head_names:
                p.requires_grad = False

        _, neg_params, _ = _split_params(model)
        opt = AdamW([
            {"params": neg_params, "lr": cfg.learning_rate * cfg.neg_lr_mult, "weight_decay": cfg.weight_decay},
        ])
        total_steps = cfg.phase2_epochs * len(phase2_train_loader)
        sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
        scaler = _make_scaler(cfg.fp16, device)

        for ep in range(cfg.phase2_epochs):
            model.train()
            ep_loss, n_batch = 0.0, 0
            pbar = tqdm(phase2_train_loader, desc=f"P2 Ep {ep+1}/{cfg.phase2_epochs}")
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
            history["train_loss"].append(avg)
            history["val_acc"].append(val)
            history["phase"].append(2)
            print(f"  Phase2 Ep{ep+1}: loss={avg:.4f} val_acc={val:.4f}")

        p2_mnli = eval_mnli(model, val_loader, device)
        p2_hans = eval_hans(model, hans_loader, device)
        p2_esnli = eval_esnli(model, esnli_loader, device)
        print(f"  Phase2 HANS: overall={p2_hans['hans_overall']:.4f} "
              f"ent={p2_hans['hans_entailment']:.4f} "
              f"non-ent={p2_hans['hans_non_entailment']:.4f}")
        _log_lora_norms(model)
        metrics["phase2"] = {"mnli": p2_mnli, "hans": p2_hans, "esnli": p2_esnli}

        if cfg.save_checkpoints_per_phase:
            phase2_metrics_to_save = {"mnli": p2_mnli, "hans": p2_hans, "esnli": p2_esnli}
            if cfg.checkpoint_dir:
                checkpoint_path_p2 = os.path.join(cfg.checkpoint_dir, f"phase2_checkpoint_epoch{cfg.phase2_epochs}.pt")
            else:
                run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
                checkpoint_path_p2 = os.path.join(run_dir, "checkpoints", f"phase2_checkpoint_epoch{cfg.phase2_epochs}.pt")
            _save_checkpoint(model, opt, sch, cfg.phase2_epochs, 2, checkpoint_path_p2, history, phase2_metrics_to_save)

    if start_phase <= 3:
        all_layer_tags = list(get_ties_modules_by_layer(model).keys())
        if cfg.no_ties_ablation:
            configure_ties_layers(model, [])
            metrics["phase2_5"] = {
                "mode": "no_ties",
                "shortcut_layers": [],
                "all_layer_tags": all_layer_tags,
            }
            print("[Phase2.5] No-TIES ablation enabled. All layers fall back to P-only in Phase 3.")
        elif cfg.enable_layerwise_analysis:
            # Phase 2.5 only performs analysis and layer selection; no parameters are updated here.
            print("\n" + "=" * 60)
            print(f"Phase 2.5: Layer-wise shortcut analysis (knn_mode={cfg.knn_mode})")
            print("=" * 60)
            analysis_ref_loader = None
            analysis_query_loader = None
            if cfg.knn_mode != "off":
                # Only prepare analysis samples if kNN is enabled; 'off' mode defaults to the original pure KL behavior.
                analysis_ref_loader, analysis_query_loader = make_phase2_5_analysis_loaders(cfg, tok)
            shortcut_layers = analyze_shortcut_layers(
                model,
                train_loader,
                analysis_ref_loader,
                analysis_query_loader,
                cfg,
            )
            # Apply the shortcut_layers selected during the analysis phase back to the model for selective TIES merging in Phase 3.
            configure_ties_layers(model, shortcut_layers)
            analysis_cache = getattr(model, "shortcut_analysis_cache", {})
            metrics["phase2_5"] = {
                "mode": "layerwise",
                **analysis_cache,
            }
        else:
            configure_ties_layers(model, all_layer_tags)
            metrics["phase2_5"] = {
                "mode": "global_ties",
                "shortcut_layers": all_layer_tags,
                "all_layer_tags": all_layer_tags,
            }
            print(f"[Phase2.5] Global TIES enabled on all {len(all_layer_tags)} layers.")

    # │────────────────────────────────────────────────────────────────────│
    # Phase 3 — TIES-Unlearning Fine-tuning (head + P, N frozen)
    # │────────────────────────────────────────────────────────────────────│
    if start_phase <= 3:
        print("\n" + "=" * 60)
        print("Phase 3: TIES-Unlearning Fine-tuning (head + P, N frozen)")
        print("=" * 60)
        set_model_forward_mode(model, "eval")

        # Freeze N, unfreeze P and head.
        for n, p in model.named_parameters():
            if "lora_P_" in n:
                p.requires_grad = True
            elif "lora_N_" in n:
                p.requires_grad = False
            elif n in head_names:
                p.requires_grad = True

        pos_params, _, head_params = _split_params(model)
        opt = AdamW([
            {"params": head_params, "lr": cfg.learning_rate * 0.1, "weight_decay": cfg.weight_decay},
            {"params": pos_params, "lr": cfg.learning_rate * 0.1, "weight_decay": cfg.weight_decay},
        ])
        total_steps = cfg.phase3_epochs * len(train_loader)
        sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
        scaler = _make_scaler(cfg.fp16, device)

        for ep in range(cfg.phase3_epochs):
            model.train()
            ep_loss, n_batch = 0.0, 0
            pbar = tqdm(train_loader, desc=f"P3 Ep {ep+1}/{cfg.phase3_epochs}")
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
            history["train_loss"].append(avg)
            history["val_acc"].append(val)
            history["phase"].append(3)
            print(f"  Phase3 Ep{ep+1}: loss={avg:.4f} val_acc={val:.4f}")

    p3_mnli = eval_mnli(model, val_loader, device)
    p3_hans = eval_hans(model, hans_loader, device)
    p3_esnli = eval_esnli(model, esnli_loader, device)
    print(f"  Phase3 HANS: overall={p3_hans['hans_overall']:.4f} "
          f"ent={p3_hans['hans_entailment']:.4f} "
          f"non-ent={p3_hans['hans_non_entailment']:.4f}")
    _log_lora_norms(model)
    metrics["phase3"] = {"mnli": p3_mnli, "hans": p3_hans, "esnli": p3_esnli}

    # --- save ---
    run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    metrics["history"] = history

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if cfg.save_checkpoints:
        ckpt_dir = os.path.join(run_dir, "model")
        os.makedirs(ckpt_dir, exist_ok=True)
        export = merge_and_unload(model)
        export.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)
        print(f"  Checkpoint saved to {ckpt_dir}")

    return metrics
