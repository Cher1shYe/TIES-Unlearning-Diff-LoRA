import os
import json
import random
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
    make_phase2_biased_mixed_loader, make_phase2_overlap_biased_loader,
    make_overlap_diagnostic_loader,
    make_phase2_5_analysis_loaders,
    make_esnli_test_loader, make_anli_test_loader, make_snli_hard_test_loader,
    make_wanli_test_loader,
)
from models.surgery import (
    inject_ties_unlearn_lora, get_ties_modules_by_layer, configure_ties_layers,
    merge_and_unload, _layer_index_from_tag,
)
from models.ties_lora import set_model_forward_mode
from models.analyzer import analyze_shortcut_layers
from utils.optim_utils import _split_params, _make_scaler, _amp_enabled, _log_lora_norms, _save_checkpoint, _load_checkpoint
from training.evaluate import (
    eval_mnli, eval_hans, eval_esnli, eval_anli, eval_snli_hard, eval_wanli,
    eval_pred_distribution, eval_overlap_shortcut,
)


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
    if cfg.phase2_shortcut_source == "mnli_overlap":
        phase2_train_loader = make_phase2_overlap_biased_loader(cfg, tok)
    else:
        phase2_train_loader = make_phase2_biased_mixed_loader(cfg, tok)
    esnli_loader = make_esnli_test_loader(cfg, tok)
    anli_loader = make_anli_test_loader(cfg, tok)
    snli_hard_loader = make_snli_hard_test_loader(cfg, tok)
    try:
        wanli_loader = make_wanli_test_loader(cfg, tok)
    except Exception as e:
        print(f"[TIES-Unlearn] WARNING: could not load WANLI ({e}); WANLI eval skipped (n/a).")
        wanli_loader = None

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

        # --- N-branch sanity checks (model is still in "phase2" mode = base + ΔN) ---
        # (1) Collapse check on MNLI val. NOT on HANS: every HANS example is
        #     high-overlap by construction, so a HEALTHY overlap-shortcut N also
        #     predicts ~100% entailment there — the old HANS-based check could not
        #     tell shortcut learning apart from collapse. Only on MNLI does a
        #     one-class histogram really mean a constant predictor.
        p2_ndist = eval_pred_distribution(model, val_loader, device)
        # (2) Shortcut-alignment check on MNLI-derived high/low-overlap groups
        #     (HANS-free): N learned the overlap feature iff it predicts
        #     entailment on high-overlap but not on low-overlap examples.
        overlap_diag_loader = make_overlap_diagnostic_loader(cfg, tok)
        p2_shortcut = eval_overlap_shortcut(model, overlap_diag_loader, device)
        # HANS pred dist kept for reference only (expected ~all-entailment for a
        # shortcut-aligned N; not used for the collapse verdict).
        p2_hans_ndist = eval_pred_distribution(model, hans_loader, device)
        print(f"  [N-branch check] MNLI pred dist: "
              f"ent={p2_ndist['pred_entailment']:.2%} "
              f"neu={p2_ndist['pred_neutral']:.2%} "
              f"con={p2_ndist['pred_contradiction']:.2%}")
        print(f"  [N-branch check] overlap alignment: "
              f"high-ov ent-rate={p2_shortcut['high_ov_pred_entail_rate']:.2%} "
              f"low-ov ent-rate={p2_shortcut['low_ov_pred_entail_rate']:.2%} "
              f"gap={p2_shortcut['ent_rate_gap']:.2%} | "
              f"high-ov gold-non-ent predicted entail: "
              f"{p2_shortcut['high_ov_nonent_pred_entail_rate']:.2%}")
        if p2_ndist["collapsed"]:
            print(f"  [N-branch check] ⚠️  COLLAPSED: {p2_ndist['max_class_frac']:.1%} "
                  f"of MNLI predictions are one class — N is a constant predictor. "
                  f"Lower neg_lr_mult or check phase2_shortcut_source.")
        elif not p2_shortcut["learned_shortcut"]:
            print(f"  [N-branch check] ⚠️  predictions spread but NOT overlap-aligned "
                  f"(gap {p2_shortcut['ent_rate_gap']:.1%} < 30%) — N may have learned "
                  f"task signal instead of the shortcut. Sharpen overlap thresholds "
                  f"or raise phase2_epochs.")
        else:
            print(f"  [N-branch check] ✅ N keys on the overlap feature "
                  f"(gap {p2_shortcut['ent_rate_gap']:.1%}), not a constant.")

        norm_summary = _log_lora_norms(model)
        metrics["phase2"] = {"mnli": p2_mnli, "hans": p2_hans, "esnli": p2_esnli,
                             "lora_norm_summary": norm_summary,
                             "n_branch_pred_dist": p2_ndist,
                             "n_branch_shortcut": p2_shortcut,
                             "n_branch_hans_pred_dist": p2_hans_ndist,
                             "overlap_group_size": getattr(phase2_train_loader, "overlap_group_size", None)}

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
        elif cfg.random_layer_selection:
            # Ablation: pick layer_selection_topk layers uniformly at random (seeded for reproducibility).
            rng = random.Random(cfg.seed)
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

        # Merged-model snapshot BEFORE any Phase-3 fine-tuning: separates what the
        # subtraction itself does to the model from what the debias FT does to it
        # (the HANS ent→0 flip can come from either; this disambiguates).
        pm_mnli = eval_mnli(model, val_loader, device)
        pm_hans = eval_hans(model, hans_loader, device)
        metrics["phase2_5"]["post_merge_pre_ft"] = {"mnli": pm_mnli, "hans": pm_hans}
        print(f"[Phase3] post-merge (pre-FT): MNLI={pm_mnli['mnli_accuracy']:.4f} "
              f"HANS ent={pm_hans['hans_entailment']:.4f} "
              f"non-ent={pm_hans['hans_non_entailment']:.4f}")

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
        if cfg.phase3_debias_reweight:
            print(f"[Phase3] debias reweighting ON (gamma={cfg.phase3_reweight_gamma}): "
                  f"down-weighting examples the frozen N (shortcut) path already solves, "
                  f"so Phase-3 cannot recover MNLI via the shortcut.")

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
                    if cfg.phase3_debias_reweight:
                        # Fix (A): score each example with the FROZEN N (shortcut) path.
                        # High N-confidence on the gold label => the shortcut alone solves it
                        # => down-weight, so recovering MNLI here can't re-learn the shortcut.
                        inputs = {k: v for k, v in batch.items() if k != "labels"}
                        labels = batch["labels"]
                        with torch.no_grad():
                            model.eval()                              # dropout off: stable N scores
                            set_model_forward_mode(model, "phase2")   # base + ΔN only
                            n_logits = model(**inputs).logits
                            set_model_forward_mode(model, "eval")      # back to merged
                            model.train()
                            p_n = torch.softmax(n_logits.float(), dim=-1).gather(
                                1, labels.view(-1, 1)).squeeze(1)
                            # Floor keeps shortcut-solvable examples visible instead of
                            # effectively deleting them from training.
                            w = (1.0 - p_n).clamp_min(0.05).pow(cfg.phase3_reweight_gamma)
                            # PER-CLASS mean-1 normalization. N solves far more entailment
                            # than non-entailment examples, so a global normalization
                            # shifts the effective label prior toward non-entailment and
                            # the model flips to 'never entailment' on HANS (ent→5%,
                            # non-ent→98%). Normalizing within each gold class keeps the
                            # label prior intact while still emphasizing, inside each
                            # class, the examples the shortcut cannot solve.
                            for c in labels.unique():
                                m = labels == c
                                w[m] = w[m] * (m.sum() / w[m].sum().clamp_min(1e-6))
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

    p3_mnli = eval_mnli(model, val_loader, device)
    p3_hans = eval_hans(model, hans_loader, device)
    p3_esnli = eval_esnli(model, esnli_loader, device)
    p3_anli = eval_anli(model, anli_loader, device)
    p3_snli_hard = eval_snli_hard(model, snli_hard_loader, device)
    p3_wanli = (eval_wanli(model, wanli_loader, device)
                if wanli_loader is not None else {"wanli_accuracy": float("nan")})
    print(f"  Phase3 HANS: overall={p3_hans['hans_overall']:.4f} "
          f"ent={p3_hans['hans_entailment']:.4f} "
          f"non-ent={p3_hans['hans_non_entailment']:.4f}")
    print(f"  Phase3 OOD: ANLI={p3_anli['anli_accuracy']:.4f} "
          f"SNLI-hard={p3_snli_hard['snli_hard_accuracy']:.4f} "
          f"WANLI={p3_wanli['wanli_accuracy']:.4f}")
    _log_lora_norms(model)
    metrics["phase3"] = {"mnli": p3_mnli, "hans": p3_hans, "esnli": p3_esnli,
                         "anli": p3_anli, "snli_hard": p3_snli_hard, "wanli": p3_wanli}

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
