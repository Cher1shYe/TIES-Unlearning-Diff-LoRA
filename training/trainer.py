import os
import json
import random
import torch
from dataclasses import asdict
from typing import Dict, List, Optional

from torch.optim import AdamW
from torch.utils.data import DataLoader
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
    set_seed, make_mnli_loaders, make_hans_dev_loader, make_hans_evaluation_loader,
    make_phase2_biased_mixed_loader, make_phase2_5_analysis_loaders,
    make_esnli_test_loader, make_anli_test_loader, make_snli_hard_test_loader,
    make_wanli_test_loader,
)
from models.surgery import (
    inject_ties_unlearn_lora, get_ties_modules_by_layer, configure_ties_layers,
    merge_and_unload, _layer_index_from_tag,
)
from models.ties_lora import set_model_forward_mode
from models.analyzer import analyze_shortcut_layers
from utils.optim_utils import (
    _split_params, _make_scaler, _amp_enabled, _log_lora_norms,
    _save_checkpoint, _load_checkpoint, _load_model_checkpoint,
)
from training.evaluate import eval_mnli, eval_hans, eval_esnli, eval_anli, eval_snli_hard, eval_wanli
from training.weighting import compute_class_priors, resolve_weighting_mode, torch_phase3_weights
from canonical.artifacts import sha256_file, write_json, write_jsonl
from canonical.results import validate_final_metric_schema


@torch.no_grad()
def _estimate_class_priors(model, train_loader, device, cfg):
    """Estimate frozen class-only weights from the fixed MNLI train subset."""
    labels_all = []
    probabilities_all = []
    set_model_forward_mode(model, "phase2")
    model.eval()
    estimation_loader = DataLoader(
        train_loader.dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    for batch in estimation_loader:
        labels = batch["label"].to(device)
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        logits = model(**inputs).logits
        gold_probability = torch.softmax(logits.float(), dim=-1).gather(
            1, labels.view(-1, 1)
        ).squeeze(1)
        labels_all.extend(labels.cpu().tolist())
        probabilities_all.extend(gold_probability.cpu().tolist())
    return compute_class_priors(
        labels_all,
        probabilities_all,
        gamma=cfg.phase3_reweight_gamma,
        classes=tuple(range(cfg.num_labels)),
    )


def train_ties_unlearn(
    cfg: TrainConfig,
    resume_from_checkpoint_path: Optional[str] = None,
    *,
    stop_after_phase2: bool = False,
    shared_checkpoint_path: Optional[str] = None,
    method_tag: Optional[str] = None,
    checkpoint_hash: Optional[str] = None,
):
    """
    Three-phase TIES-Unlearning Diff-LoRA training.
    Returns a dict of metrics for final comparison.
    """
    if resume_from_checkpoint_path and shared_checkpoint_path:
        raise ValueError("Use either legacy resume or shared_checkpoint_path, not both.")
    if stop_after_phase2 and shared_checkpoint_path:
        raise ValueError("A shared branch cannot also prepare a shared checkpoint.")
    set_seed(cfg.training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TIES-Unlearn] device={device}")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    train_loader, val_loader = make_mnli_loaders(cfg, tok)
    if shared_checkpoint_path:
        hans_dev_loader = None
        phase2_train_loader = None
    else:
        hans_dev_loader = make_hans_dev_loader(cfg, tok)
        phase2_train_loader = make_phase2_biased_mixed_loader(cfg, tok)

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
        trim_ratio=cfg.trim_ratio, merge_mode=cfg.merge_mode,
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
    class_prior_weights = None

    if shared_checkpoint_path:
        shared = _load_model_checkpoint(model, shared_checkpoint_path, device)
        start_phase = 3
        history = shared.get("history", history)
        metrics["history"] = history
        loaded_metrics = shared.get("phase_metrics", {})
        metrics["phase1"] = loaded_metrics.get("phase1", {})
        metrics["phase2"] = loaded_metrics.get("phase2", {})
        raw_priors = shared.get("class_prior_weights")
        if raw_priors is not None:
            class_prior_weights = {int(key): float(value) for key, value in raw_priors.items()}
        actual_hash = sha256_file(shared_checkpoint_path)
        if checkpoint_hash is not None and checkpoint_hash != actual_hash:
            raise ValueError(
                f"Shared checkpoint hash mismatch: expected {checkpoint_hash}, got {actual_hash}."
            )
        checkpoint_hash = actual_hash
        print("[TIES-Unlearn] Starting canonical branch from shared Phase-2 checkpoint.")

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
        p1_hans_dev = eval_hans(model, hans_dev_loader, device)
        print(f"  Phase1 HANS-dev: overall={p1_hans_dev['hans_overall']:.4f} "
              f"ent={p1_hans_dev['hans_entailment']:.4f} "
              f"non-ent={p1_hans_dev['hans_non_entailment']:.4f}")
        _log_lora_norms(model)
        metrics["phase1"] = {"mnli": p1_mnli, "hans_dev": p1_hans_dev}

        if cfg.save_checkpoints_per_phase:
            phase1_metrics_to_save = {"mnli": p1_mnli, "hans_dev": p1_hans_dev}
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
        p2_hans_dev = eval_hans(model, hans_dev_loader, device)
        print(f"  Phase2 HANS-dev: overall={p2_hans_dev['hans_overall']:.4f} "
              f"ent={p2_hans_dev['hans_entailment']:.4f} "
              f"non-ent={p2_hans_dev['hans_non_entailment']:.4f}")
        _log_lora_norms(model)
        metrics["phase2"] = {"mnli": p2_mnli, "hans_dev": p2_hans_dev}

        if cfg.save_checkpoints_per_phase:
            phase2_metrics_to_save = {"mnli": p2_mnli, "hans_dev": p2_hans_dev}
            if cfg.checkpoint_dir:
                checkpoint_path_p2 = os.path.join(cfg.checkpoint_dir, f"phase2_checkpoint_epoch{cfg.phase2_epochs}.pt")
            else:
                run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
                checkpoint_path_p2 = os.path.join(run_dir, "checkpoints", f"phase2_checkpoint_epoch{cfg.phase2_epochs}.pt")
            _save_checkpoint(model, opt, sch, cfg.phase2_epochs, 2, checkpoint_path_p2, history, phase2_metrics_to_save)

    if stop_after_phase2:
        class_prior_weights = _estimate_class_priors(model, train_loader, device, cfg)
        if cfg.checkpoint_dir:
            shared_path = os.path.join(cfg.checkpoint_dir, "shared_phase2_checkpoint.pt")
        else:
            shared_path = os.path.join(
                cfg.output_dir,
                cfg.experiment_name,
                "checkpoints",
                "shared_phase2_checkpoint.pt",
            )
        phase_metrics = {"phase1": metrics["phase1"], "phase2": metrics["phase2"]}
        _save_checkpoint(
            model,
            opt,
            sch,
            cfg.phase2_epochs,
            2,
            shared_path,
            history,
            phase_metrics,
            metadata={
                "config": asdict(cfg),
                "class_prior_weights": {
                    str(label): value for label, value in class_prior_weights.items()
                },
                "checkpoint_role": "canonical_shared_phase2",
            },
        )
        shared_hash = sha256_file(shared_path)
        return {
            "method": "canonical_shared_phase2",
            "config": asdict(cfg),
            "phase1": metrics["phase1"],
            "phase2": metrics["phase2"],
            "class_prior_weights": class_prior_weights,
            "checkpoint_path": shared_path,
            "checkpoint_hash": shared_hash,
        }

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
        elif cfg.random_layer_selection:
            # Ablation: pick layer_selection_topk layers uniformly at random (seeded for reproducibility).
            rng = random.Random(cfg.training_seed)
            k = min(int(cfg.layer_selection_topk), len(all_layer_tags))
            shortcut_layers = sorted(rng.sample(all_layer_tags, k), key=_layer_index_from_tag)
            configure_ties_layers(model, shortcut_layers)
            metrics["phase2_5"] = {
                "mode": "random",
                "shortcut_layers": shortcut_layers,
                "all_layer_tags": all_layer_tags,
            }
            print(f"[Phase2.5] Random layer selection: {shortcut_layers}")
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

        # Fix (B): keep the P-branch of the subtracted layers frozen so Phase-3 cannot
        # rebuild the shortcut direction we just removed. _split_params filters by
        # requires_grad, so the optimizer below automatically excludes these.
        n_frozen_p = 0
        if cfg.phase3_freeze_subtracted_p:
            for layer_tag, modules in get_ties_modules_by_layer(model).items():
                for module in modules:
                    if module.enable_ties:
                        for p in module.get_pos_params():
                            p.requires_grad = False
                            n_frozen_p += 1
            metrics["phase2_5"]["phase3_frozen_p_tensors"] = n_frozen_p
            print(f"[Phase3] freeze_subtracted_p: froze P on subtracted layers "
                  f"({n_frozen_p} P tensors); head + non-subtracted P stay trainable.")

        pos_params, _, head_params = _split_params(model)
        opt = AdamW([
            {"params": head_params, "lr": cfg.learning_rate * 0.1, "weight_decay": cfg.weight_decay},
            {"params": pos_params, "lr": cfg.learning_rate * 0.1, "weight_decay": cfg.weight_decay},
        ])
        total_steps = cfg.phase3_epochs * len(train_loader)
        sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
        scaler = _make_scaler(cfg.fp16, device)
        weighting_mode = resolve_weighting_mode(cfg)
        if weighting_mode == "class_prior" and class_prior_weights is None:
            raise ValueError("class_prior weighting requires priors from the shared Phase-2 checkpoint.")
        if weighting_mode == "n_guided":
            print(f"[Phase3] debias reweighting ON (gamma={cfg.phase3_reweight_gamma}): "
                  f"down-weighting examples the frozen N (shortcut) path already solves, "
                  f"so Phase-3 cannot recover MNLI via the shortcut.")
        elif weighting_mode == "class_prior":
            print(f"[Phase3] class-prior reweighting ON: {class_prior_weights}")
        else:
            print("[Phase3] example reweighting OFF (ordinary cross entropy).")

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
                    if weighting_mode != "none":
                        inputs = {k: v for k, v in batch.items() if k != "labels"}
                        labels = batch["labels"]
                        n_gold_probabilities = None
                        if weighting_mode == "n_guided":
                            with torch.no_grad():
                                set_model_forward_mode(model, "phase2")
                                n_logits = model(**inputs).logits
                                set_model_forward_mode(model, "eval")
                                n_gold_probabilities = torch.softmax(
                                    n_logits.float(), dim=-1
                                ).gather(1, labels.view(-1, 1)).squeeze(1)
                        w = torch_phase3_weights(
                            weighting_mode,
                            labels,
                            n_gold_probabilities=n_gold_probabilities,
                            class_priors=class_prior_weights,
                            gamma=cfg.phase3_reweight_gamma,
                        )
                        logits = model(**inputs).logits
                        per = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                        loss = (per * w).mean()
                    else:
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

    run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    final_checkpoint_hash = checkpoint_hash
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
                "source_phase2_checkpoint_hash": checkpoint_hash,
            },
            final_state_path,
        )
        final_checkpoint_hash = sha256_file(final_state_path)

    # Official HANS and all other final-only OOD loaders are constructed only
    # after the frozen branch has finished training and its final state is saved.
    hans_loader = make_hans_evaluation_loader(cfg, tok)
    esnli_loader = make_esnli_test_loader(cfg, tok)
    anli_loader = make_anli_test_loader(cfg, tok)
    snli_hard_loader = make_snli_hard_test_loader(cfg, tok)
    try:
        wanli_loader = make_wanli_test_loader(cfg, tok)
    except Exception as e:
        print(f"[TIES-Unlearn] WARNING: could not load WANLI ({e}); WANLI eval skipped (n/a).")
        wanli_loader = None

    p3_mnli = eval_mnli(model, val_loader, device)
    hans_predictions = None
    if method_tag is not None:
        p3_hans, hans_predictions = eval_hans(
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
        p3_hans = eval_hans(model, hans_loader, device)
    p3_esnli = eval_esnli(model, esnli_loader, device)
    p3_anli = eval_anli(model, anli_loader, device)
    p3_snli_hard = eval_snli_hard(model, snli_hard_loader, device)
    p3_wanli = (eval_wanli(model, wanli_loader, device)
                if wanli_loader is not None else {
                    "wanli_accuracy": None if method_tag is not None else float("nan")
                })
    print(f"  Phase3 HANS: overall={p3_hans['hans_overall']:.4f} "
          f"ent={p3_hans['hans_entailment']:.4f} "
          f"non-ent={p3_hans['hans_non_entailment']:.4f}")
    wanli_display = (
        f"{p3_wanli['wanli_accuracy']:.4f}"
        if p3_wanli["wanli_accuracy"] is not None else "n/a"
    )
    print(f"  Phase3 OOD: ANLI={p3_anli['anli_accuracy']:.4f} "
          f"SNLI-hard={p3_snli_hard['snli_hard_accuracy']:.4f} "
          f"WANLI={wanli_display}")
    _log_lora_norms(model)
    metrics["phase3"] = {"mnli": p3_mnli, "hans": p3_hans, "esnli": p3_esnli,
                         "anli": p3_anli, "snli_hard": p3_snli_hard, "wanli": p3_wanli}
    if method_tag is not None:
        metrics["checkpoint_provenance"] = {
            "source_phase2_checkpoint_hash": checkpoint_hash,
            "final_checkpoint_hash": final_checkpoint_hash,
        }

    if cfg.record_branch_only_metrics:
        print("  Recording final branch-only evaluations (P-only and N-only).")

        set_model_forward_mode(model, "phase1")
        p_only_mnli = eval_mnli(model, val_loader, device)
        p_only_hans = eval_hans(model, hans_loader, device)
        p_only_esnli = eval_esnli(model, esnli_loader, device)
        p_only_anli = eval_anli(model, anli_loader, device)
        p_only_snli_hard = eval_snli_hard(model, snli_hard_loader, device)
        p_only_wanli = (eval_wanli(model, wanli_loader, device)
                        if wanli_loader is not None else {"wanli_accuracy": float("nan")})

        set_model_forward_mode(model, "phase2")
        n_only_mnli = eval_mnli(model, val_loader, device)
        n_only_hans = eval_hans(model, hans_loader, device)
        n_only_esnli = eval_esnli(model, esnli_loader, device)
        n_only_anli = eval_anli(model, anli_loader, device)
        n_only_snli_hard = eval_snli_hard(model, snli_hard_loader, device)
        n_only_wanli = (eval_wanli(model, wanli_loader, device)
                        if wanli_loader is not None else {"wanli_accuracy": float("nan")})

        set_model_forward_mode(model, "eval")
        metrics["branch_only"] = {
            "p_only": {"mnli": p_only_mnli, "hans": p_only_hans, "esnli": p_only_esnli,
                       "anli": p_only_anli, "snli_hard": p_only_snli_hard, "wanli": p_only_wanli},
            "n_only": {"mnli": n_only_mnli, "hans": n_only_hans, "esnli": n_only_esnli,
                       "anli": n_only_anli, "snli_hard": n_only_snli_hard, "wanli": n_only_wanli},
        }

    # --- save ---
    metrics["history"] = history
    if method_tag is not None:
        validate_final_metric_schema(metrics["phase3"])
        write_json(os.path.join(run_dir, "metrics.json"), metrics)
        write_jsonl(os.path.join(run_dir, "hans_predictions.jsonl"), hans_predictions)
        write_json(os.path.join(run_dir, "selected_layers.json"), metrics.get("phase2_5", {}))
    else:
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
