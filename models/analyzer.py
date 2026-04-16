import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from models.ties_lora import TIESUnlearnLoRALinear, set_model_forward_mode
from models.surgery import get_ties_modules_by_layer, _layer_index_from_tag

ANALYSIS_GROUP_NAMES = {
    0: "mnli",
    1: "hans_entailment",
    2: "hans_non_entailment",
}

def _classifier_logits_from_hidden(model: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(1)

    if hasattr(model, "classifier"):
        classifier = model.classifier
        try:
            logits = classifier(hidden_states)
        except TypeError:
            pooled = hidden_states[:, 0, :]
            if hasattr(model, "dropout") and isinstance(model.dropout, nn.Module):
                pooled = model.dropout(pooled)
            logits = classifier(pooled)
        return logits if logits.dim() == 2 else logits[:, 0, :]

    if hasattr(model, "score"):
        pooled = hidden_states[:, 0, :] if hidden_states.dim() == 3 else hidden_states
        logits = model.score(pooled)
        return logits if logits.dim() == 2 else logits[:, 0, :]

    raise ValueError("Unsupported classifier head: expected `classifier` or `score`.")

def _normalise_layer_scores(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return {key: 0.0 for key in values}
    return {key: float((val - lo) / (hi - lo)) for key, val in values.items()}

def _pd_positions_to_summary(pd_positions: torch.Tensor, layer_tags: List[str]) -> Dict[str, float]:
    layer_indices = torch.tensor(
        [_layer_index_from_tag(layer_tags[int(pos)]) for pos in pd_positions.tolist()],
        dtype=torch.float32,
    )
    return {
        "mean": float(layer_indices.mean().item()),
        "q25": float(torch.quantile(layer_indices, 0.25).item()),
        "median": float(torch.quantile(layer_indices, 0.50).item()),
        "q75": float(torch.quantile(layer_indices, 0.75).item()),
    }

def _compute_stability_positions(layer_preds: torch.Tensor, stability_window: int = 2) -> torch.Tensor:
    # For each sample, find the earliest stable layer: from this layer to the final layer, the predictions match the final layer's prediction.
    num_layers, num_samples = layer_preds.shape
    final_pred = layer_preds[-1]
    depth = torch.full(
        (num_samples,),
        fill_value=num_layers - 1,
        dtype=torch.long,
        device=layer_preds.device,
    )
    assigned = torch.zeros(num_samples, dtype=torch.bool, device=layer_preds.device)

    for start in range(num_layers):
        suffix_match = (layer_preds[start:] == final_pred.unsqueeze(0)).all(dim=0)
        window_end = min(num_layers, start + max(int(stability_window), 1))
        window_match = (layer_preds[start:window_end] == final_pred.unsqueeze(0)).all(dim=0)
        stable_now = suffix_match & window_match & (~assigned)
        depth = torch.where(stable_now, torch.full_like(depth, start), depth)
        assigned = assigned | stable_now

    return depth

def _curve_from_positions(
    # Convert sample-level stable layer positions into a cumulative curve representing "how many samples are stable up to a certain layer".
    pd_positions: torch.Tensor,
    layer_count: int,
    mask: Optional[torch.Tensor] = None,
    wrong_mask: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    if mask is None:
        mask = torch.ones_like(pd_positions, dtype=torch.bool)
    selected = pd_positions[mask]
    denom = max(int(mask.sum().item()), 1)
    curves = {
        "cumulative": [],
    }

    for pos in range(layer_count):
        curves["cumulative"].append(float((selected <= pos).sum().item() / denom))

    if wrong_mask is not None:
        wrong_selected = wrong_mask[mask]
        wrong_curve = []
        for pos in range(layer_count):
            active = (selected <= pos) & wrong_selected
            wrong_curve.append(float(active.sum().item() / denom))
        curves["early_wrong_cumulative"] = wrong_curve

    return curves

@torch.no_grad()
def _collect_kl_layer_stats(model: nn.Module, calib_loader, layer_tags: List[str], cfg) -> List[Dict[str, float]]:
    # Stage 1: First, use the existing KL analysis to roughly filter all optional layers. The subsequent kNN will only run on the candidate layers.
    device = next(model.parameters()).device
    stats = {layer_tag: {"kl_sum": 0.0, "count": 0} for layer_tag in layer_tags}
    max_batches = max(int(cfg.kl_batches), 1)

    for batch_idx, batch in enumerate(calib_loader):
        if batch_idx >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items() if k != "label"}
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "output_hidden_states": True,
            "return_dict": True,
        }

        set_model_forward_mode(model, "phase1")
        p_outputs = model(**model_inputs)
        set_model_forward_mode(model, "phase2")
        n_outputs = model(**model_inputs)

        for layer_tag in layer_tags:
            hidden_idx = _layer_index_from_tag(layer_tag) + 1
            if hidden_idx >= len(p_outputs.hidden_states) or hidden_idx >= len(n_outputs.hidden_states):
                continue

            p_logits = _classifier_logits_from_hidden(model, p_outputs.hidden_states[hidden_idx]).float()
            n_logits = _classifier_logits_from_hidden(model, n_outputs.hidden_states[hidden_idx]).float()

            p_log_probs = torch.log_softmax(p_logits, dim=-1)
            n_log_probs = torch.log_softmax(n_logits, dim=-1)
            p_probs = p_log_probs.exp()
            n_probs = n_log_probs.exp()
            kl = 0.5 * (
                torch.nn.functional.kl_div(p_log_probs, n_probs, reduction="batchmean")
                + torch.nn.functional.kl_div(n_log_probs, p_probs, reduction="batchmean")
            )
            stats[layer_tag]["kl_sum"] += kl.item()
            stats[layer_tag]["count"] += 1

    ranked = []
    for layer_tag in layer_tags:
        denom = max(stats[layer_tag]["count"], 1)
        ranked.append({
            "layer_tag": layer_tag,
            "avg_kl": stats[layer_tag]["kl_sum"] / denom,
        })
    ranked.sort(key=lambda row: row["avg_kl"], reverse=True)
    return ranked

@torch.no_grad()
def _collect_mode_embeddings(
    # Extract the CLS representations of candidate layers under a given forward mode; P-only and N-only will each run once.
    model: nn.Module,
    loader,
    candidate_layers: List[str],
    mode: str,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    set_model_forward_mode(model, mode)
    buckets: Dict[str, List[torch.Tensor]] = {layer: [] for layer in candidate_layers}
    all_labels: List[torch.Tensor] = []
    all_groups: List[torch.Tensor] = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Take the [CLS] vector for each layer and apply L2 normalization, so dot product later is equivalent to cosine similarity.
        for layer_tag in candidate_layers:
            hidden_idx = _layer_index_from_tag(layer_tag) + 1
            cls = outputs.hidden_states[hidden_idx][:, 0, :].float()
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
            buckets[layer_tag].append(cls.cpu())

        all_labels.append(batch["analysis_label"].cpu())
        all_groups.append(batch["group_id"].cpu())

    embeddings = {
        layer_tag: torch.cat(layer_chunks, dim=0) if layer_chunks else torch.empty(0, 0)
        for layer_tag, layer_chunks in buckets.items()
    }
    labels = torch.cat(all_labels, dim=0)
    groups = torch.cat(all_groups, dim=0)
    return embeddings, labels, groups

def _knn_predict(
    # Simple brute-force kNN: The reference bank and query set are not large, so we can directly compute the similarity matrix.
    ref_emb: torch.Tensor,
    ref_labels: torch.Tensor,
    query_emb: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if ref_emb.numel() == 0 or query_emb.numel() == 0:
        return torch.empty(query_emb.size(0), dtype=torch.long)
    k = min(int(k), ref_emb.size(0))
    sim = query_emb @ ref_emb.t()
    topk_idx = sim.topk(k=k, dim=1).indices
    neighbor_labels = ref_labels[topk_idx]
    counts = torch.nn.functional.one_hot(neighbor_labels, num_classes=2).sum(dim=1)
    return counts.argmax(dim=-1)

def _build_knn_metric_rows(
    layer_preds: Dict[str, torch.Tensor],
    final_preds: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    candidate_layers: List[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[Dict[str, Any]]]:
    nested: Dict[str, Dict[str, Dict[str, float]]] = {}
    rows: List[Dict[str, Any]] = []

    for layer_tag in candidate_layers:
        layer_pred = layer_preds[layer_tag]
        nested[layer_tag] = {}
        for group_id, group_name in ANALYSIS_GROUP_NAMES.items():
            mask = groups == group_id
            if int(mask.sum().item()) == 0:
                continue
            acc = float((layer_pred[mask] == labels[mask]).float().mean().item())
            agree = float((layer_pred[mask] == final_preds[mask]).float().mean().item())
            nested[layer_tag][group_name] = {
                "accuracy": acc,
                "agreement_with_final": agree,
            }
            rows.append({
                "layer_tag": layer_tag,
                "group": group_name,
                "accuracy": acc,
                "agreement_with_final": agree,
            })

    return nested, rows

def _analyse_knn_outputs(
    # Organize the layer-wise kNN predictions into three types of signals: PD, accuracy difference, and early-wrong on HANS non-entailment.
    candidate_layers: List[str],
    query_labels: torch.Tensor,
    query_groups: torch.Tensor,
    preds_by_mode: Dict[str, Dict[str, torch.Tensor]],
    cfg,
) -> Dict[str, Any]:
    mode_metrics: Dict[str, Any] = {}
    signal_inputs = {
        "pd_signal": {layer_tag: 0.0 for layer_tag in candidate_layers},
        "acc_signal": {layer_tag: 0.0 for layer_tag in candidate_layers},
        "early_wrong_signal": {layer_tag: 0.0 for layer_tag in candidate_layers},
    }

    candidate_count = len(candidate_layers)
    group_masks = {
        name: (query_groups == group_id)
        for group_id, name in ANALYSIS_GROUP_NAMES.items()
    }

    pd_curves_by_mode: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for mode_name, layer_preds_dict in preds_by_mode.items():
        # Stack the predictions of candidate layers to facilitate calculating PD directly by "layer x sample".
        stacked_preds = torch.stack([layer_preds_dict[layer_tag] for layer_tag in candidate_layers], dim=0)
        final_preds = stacked_preds[-1]
        pd_positions = _compute_stability_positions(stacked_preds, cfg.knn_stability_window)
        metric_nested, metric_rows = _build_knn_metric_rows(
            layer_preds_dict,
            final_preds,
            query_labels,
            query_groups,
            candidate_layers,
        )

        pd_summary = {}
        pd_curves = {}
        for group_name, mask in group_masks.items():
            if int(mask.sum().item()) == 0:
                continue
            pd_summary[group_name] = _pd_positions_to_summary(pd_positions[mask], candidate_layers)
            wrong_mask = None
            # Only on HANS non-entailment do we additionally care about samples that "stabilized early but were ultimately wrong".
            if group_name == "hans_non_entailment":
                wrong_mask = final_preds != query_labels
            pd_curves[group_name] = _curve_from_positions(
                pd_positions,
                candidate_count,
                mask=mask,
                wrong_mask=wrong_mask,
            )

        mode_metrics[mode_name] = {
            "layer_metrics": metric_nested,
            "pd_summary": pd_summary,
        }
        pd_curves_by_mode[mode_name] = pd_curves

    n_curves = pd_curves_by_mode["phase2"]
    p_curves = pd_curves_by_mode["phase1"]
    non_name = "hans_non_entailment"

    n_cumulative = n_curves.get(non_name, {}).get("cumulative", [0.0] * candidate_count)
    p_cumulative = p_curves.get(non_name, {}).get("cumulative", [0.0] * candidate_count)
    early_wrong_curve = n_curves.get(non_name, {}).get("early_wrong_cumulative", [0.0] * candidate_count)

    phase1_non = mode_metrics["phase1"]["layer_metrics"]
    phase2_non = mode_metrics["phase2"]["layer_metrics"]

    # Here, sample-level statistics are compressed into layer-level signals for the final candidate layer scoring.
    for idx, layer_tag in enumerate(candidate_layers):
        signal_inputs["pd_signal"][layer_tag] = float(n_cumulative[idx] - p_cumulative[idx])
        p_acc = phase1_non.get(layer_tag, {}).get(non_name, {}).get("accuracy", 0.0)
        n_acc = phase2_non.get(layer_tag, {}).get(non_name, {}).get("accuracy", 0.0)
        signal_inputs["acc_signal"][layer_tag] = float(p_acc - n_acc)
        signal_inputs["early_wrong_signal"][layer_tag] = float(early_wrong_curve[idx])

    return {
        "by_mode": mode_metrics,
        "signal_inputs": signal_inputs,
    }

def _combine_candidate_scores(
    # Stage 2: Within the KL candidate layers, normalize the KL and kNN signals and perform a linear combination.
    candidate_layers: List[str],
    candidate_kl: Dict[str, float],
    knn_cache: Optional[Dict[str, Any]],
    cfg,
) -> List[Dict[str, Any]]:
    kl_norm = _normalise_layer_scores(candidate_kl)
    rows: List[Dict[str, Any]] = []

    if cfg.knn_mode == "off" or not knn_cache:
        for layer_tag in candidate_layers:
            rows.append({
                "layer_tag": layer_tag,
                "score": float(candidate_kl[layer_tag]),
                "components": {
                    "kl_raw": float(candidate_kl[layer_tag]),
                    "kl_norm": float(kl_norm[layer_tag]),
                },
            })
        rows.sort(key=lambda row: row["score"], reverse=True)
        return rows

    signals = knn_cache["signal_inputs"]
    if cfg.knn_mode == "pd":
        pd_norm = _normalise_layer_scores(signals["pd_signal"])
        total = max(cfg.kl_weight + cfg.knn_weight, 1e-8)
        for layer_tag in candidate_layers:
            score = (
                cfg.kl_weight * kl_norm[layer_tag]
                + cfg.knn_weight * pd_norm[layer_tag]
            ) / total
            rows.append({
                "layer_tag": layer_tag,
                "score": float(score),
                "components": {
                    "kl_norm": float(kl_norm[layer_tag]),
                    "pd_norm": float(pd_norm[layer_tag]),
                    "pd_raw": float(signals["pd_signal"][layer_tag]),
                },
            })
    elif cfg.knn_mode == "acc":
        acc_norm = _normalise_layer_scores(signals["acc_signal"])
        total = max(cfg.kl_weight + cfg.knn_weight, 1e-8)
        for layer_tag in candidate_layers:
            score = (
                cfg.kl_weight * kl_norm[layer_tag]
                + cfg.knn_weight * acc_norm[layer_tag]
            ) / total
            rows.append({
                "layer_tag": layer_tag,
                "score": float(score),
                "components": {
                    "kl_norm": float(kl_norm[layer_tag]),
                    "acc_norm": float(acc_norm[layer_tag]),
                    "acc_raw": float(signals["acc_signal"][layer_tag]),
                },
            })
    elif cfg.knn_mode == "pd_and_early_wrong":
        pd_norm = _normalise_layer_scores(signals["pd_signal"])
        early_wrong_norm = _normalise_layer_scores(signals["early_wrong_signal"])
        total = max(cfg.kl_weight + cfg.pd_weight + cfg.early_wrong_weight, 1e-8)
        for layer_tag in candidate_layers:
            score = (
                cfg.kl_weight * kl_norm[layer_tag]
                + cfg.pd_weight * pd_norm[layer_tag]
                + cfg.early_wrong_weight * early_wrong_norm[layer_tag]
            ) / total
            rows.append({
                "layer_tag": layer_tag,
                "score": float(score),
                "components": {
                    "kl_norm": float(kl_norm[layer_tag]),
                    "pd_norm": float(pd_norm[layer_tag]),
                    "early_wrong_norm": float(early_wrong_norm[layer_tag]),
                    "pd_raw": float(signals["pd_signal"][layer_tag]),
                    "early_wrong_raw": float(signals["early_wrong_signal"][layer_tag]),
                },
            })
    else:
        raise ValueError(f"Unsupported knn_mode: {cfg.knn_mode}")

    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows

@torch.no_grad()
def analyze_shortcut_layers(
    # Complete Phase 2.5: First perform a rough KL filtering, then run kNN analysis according to knn_mode and output the final shortcut_layers.
    model: nn.Module,
    calib_loader,
    analysis_ref_loader,
    analysis_query_loader,
    cfg,
) -> List[str]:
    layer_groups = get_ties_modules_by_layer(model)
    if not layer_groups:
        raise ValueError("No TIES modules were found for layer-wise analysis.")

    device = next(model.parameters()).device
    all_layer_tags = list(layer_groups.keys())

    was_training = model.training
    prev_mode = "eval"
    for module in model.modules():
        if isinstance(module, TIESUnlearnLoRALinear):
            prev_mode = module.forward_mode
            break

    model.eval()

    ranked_layers = _collect_kl_layer_stats(model, calib_loader, all_layer_tags, cfg)
    candidate_rows = ranked_layers[: min(int(cfg.kl_topk_candidates), len(ranked_layers))]
    candidate_layers = sorted(
        [row["layer_tag"] for row in candidate_rows],
        key=_layer_index_from_tag,
    )
    candidate_kl = {
        row["layer_tag"]: float(row["avg_kl"])
        for row in candidate_rows
    }

    print("[Phase2.5] KL top candidates:")
    for row in candidate_rows:
        print(f"  - {row['layer_tag']}: avg_kl={row['avg_kl']:.6f}")

    knn_cache: Optional[Dict[str, Any]] = None
    if cfg.knn_mode != "off":
        # P-only and N-only build their respective reference bank / query representations separately, without mixing them.
        if analysis_ref_loader is None or analysis_query_loader is None:
            raise ValueError("kNN analysis requires both reference and query loaders.")
        ref_phase1, ref_labels, _ = _collect_mode_embeddings(
            model, analysis_ref_loader, candidate_layers, "phase1", device
        )
        query_phase1, query_labels, query_groups = _collect_mode_embeddings(
            model, analysis_query_loader, candidate_layers, "phase1", device
        )
        ref_phase2, ref_labels_n, _ = _collect_mode_embeddings(
            model, analysis_ref_loader, candidate_layers, "phase2", device
        )
        query_phase2, query_labels_n, query_groups_n = _collect_mode_embeddings(
            model, analysis_query_loader, candidate_layers, "phase2", device
        )

        if not torch.equal(ref_labels, ref_labels_n):
            raise ValueError("Reference label order mismatch between P-only and N-only kNN banks.")
        if not torch.equal(query_labels, query_labels_n):
            raise ValueError("Query label order mismatch between P-only and N-only kNN banks.")
        if not torch.equal(query_groups, query_groups_n):
            raise ValueError("Query group order mismatch between P-only and N-only kNN runs.")

        # Run kNN majority voting for P-only and N-only respectively on each candidate layer.
        preds_by_mode = {
            "phase1": {},
            "phase2": {},
        }
        for layer_tag in candidate_layers:
            preds_by_mode["phase1"][layer_tag] = _knn_predict(
                ref_phase1[layer_tag], ref_labels, query_phase1[layer_tag], cfg.knn_k
            )
            preds_by_mode["phase2"][layer_tag] = _knn_predict(
                ref_phase2[layer_tag], ref_labels, query_phase2[layer_tag], cfg.knn_k
            )

        knn_cache = _analyse_knn_outputs(
            candidate_layers,
            query_labels,
            query_groups,
            preds_by_mode,
            cfg,
        )

        print("[Phase2.5] Candidate-layer kNN metrics:")
        for layer_tag in candidate_layers:
            print(f"  * {layer_tag}")
            for mode_name in ("phase1", "phase2"):
                mode_metrics = knn_cache["by_mode"][mode_name]["layer_metrics"].get(layer_tag, {})
                for group_name in ("mnli", "hans_entailment", "hans_non_entailment"):
                    if group_name not in mode_metrics:
                        continue
                    stats = mode_metrics[group_name]
                    print(
                        f"      [{mode_name}] {group_name}: "
                        f"acc={stats['accuracy']:.4f} "
                        f"agree_final={stats['agreement_with_final']:.4f}"
                    )

    # Sort the comprehensive scores of the candidate layers and take the top-k as the actual shortcut_layers fed into TIES.
    final_scores = _combine_candidate_scores(candidate_layers, candidate_kl, knn_cache, cfg)
    shortcut_topk = min(int(cfg.layer_selection_topk), len(final_scores))
    shortcut_layers = [
        row["layer_tag"] for row in final_scores[:shortcut_topk]
    ]
    shortcut_layers.sort(key=_layer_index_from_tag)

    print("[Phase2.5] Final shortcut layer scores:")
    for row in final_scores:
        print(f"  - {row['layer_tag']}: score={row['score']:.6f} components={row['components']}")
    print(f"[Phase2.5] shortcut_layers={shortcut_layers}")

    model.shortcut_analysis_cache = {
        "knn_mode": cfg.knn_mode,
        "ranked_layers": ranked_layers,
        "candidate_layers": candidate_rows,
        "knn_analysis": knn_cache,
        "final_layer_scores": final_scores,
        "shortcut_layers": shortcut_layers,
    }

    set_model_forward_mode(model, prev_mode)
    if was_training:
        model.train()
    else:
        model.eval()

    return shortcut_layers