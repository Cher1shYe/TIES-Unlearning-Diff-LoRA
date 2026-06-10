import numpy as np
import torch
from typing import Dict

@torch.no_grad()
def eval_mnli(model, loader, device) -> Dict[str, float]:
    model.eval()
    correct = total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return {"mnli_accuracy": correct / max(total, 1)}


@torch.no_grad()
def eval_hans(model, loader, device) -> Dict:
    model.eval()
    all_pred, all_label, all_heur = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=att).logits
        pred_3 = logits.argmax(-1).cpu()
        # Map 3-class → 2-class: entailment (0) vs non-entailment (1/2 → 1)
        pred_bin = (pred_3 != 0).long()
        all_pred.extend(pred_bin.tolist())
        all_label.extend(batch["label"].tolist())
        if isinstance(batch["heuristic"], torch.Tensor):
            all_heur.extend(batch["heuristic"].tolist())
        else:
            all_heur.extend(batch["heuristic"])

    pred = np.array(all_pred)
    label = np.array(all_label)
    heur = np.array(all_heur)

    ent_m = label == 0
    nent_m = label == 1
    overall = float((pred == label).mean()) if len(label) else 0.0
    ent_acc = float((pred[ent_m] == label[ent_m]).mean()) if ent_m.sum() else 0.0
    nent_acc = float((pred[nent_m] == label[nent_m]).mean()) if nent_m.sum() else 0.0

    heur_names = {0: "lexical_overlap", 1: "subsequence", 2: "constituent"}
    heur_acc = {}
    for hid, hname in heur_names.items():
        m = heur == hid
        if m.sum():
            heur_acc[hname] = float((pred[m] == label[m]).mean())

    return {
        "hans_overall": overall,
        "hans_entailment": ent_acc,
        "hans_non_entailment": nent_acc,
        "heuristic_breakdown": heur_acc,
    }

@torch.no_grad()
def eval_pred_distribution(model, loader, device) -> Dict:
    """3-class predicted-label histogram for whatever forward mode the model is
    currently in. Used after Phase 2 (model in 'phase2' mode = base + ΔN only) to
    check the N branch actually learned the shortcut FEATURE instead of collapsing
    into a constant 'always-entailment' predictor.

    A healthy N branch spreads its predictions across classes (and, on HANS, gets
    a non-trivial non-entailment accuracy). A collapsed N puts ~all mass on one
    class -> `max_class_frac` ≈ 1.0 and `collapsed` is True.

    Also breaks predictions down by the TRUE label, so you can see e.g. that on
    true non-entailment examples N still predicts entailment (the collapse symptom).
    """
    model.eval()
    counts = torch.zeros(3, dtype=torch.long)              # pred histogram
    by_true = {0: torch.zeros(3, dtype=torch.long),        # pred hist per true label
               1: torch.zeros(3, dtype=torch.long),
               2: torch.zeros(3, dtype=torch.long)}
    total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"]
        preds = model(input_ids=ids, attention_mask=mask).logits.argmax(-1).cpu()
        for c in range(3):
            counts[c] += (preds == c).sum().item()
        for t in range(3):
            tm = labels == t
            if tm.any():
                pt = preds[tm]
                for c in range(3):
                    by_true[t][c] += (pt == c).sum().item()
        total += labels.numel()

    total = max(total, 1)
    frac = (counts.float() / total).tolist()
    max_frac = max(frac)
    out = {
        "n": int(total),
        "pred_entailment": frac[0],
        "pred_neutral": frac[1],
        "pred_contradiction": frac[2],
        "max_class_frac": max_frac,
        "collapsed": bool(max_frac >= 0.90),   # ~one-class predictor
    }
    # Per-true-label predicted-entailment rate: on true non-entailment (labels 1/2),
    # a high value == the shortcut collapse symptom (N says "entailment" anyway).
    for t, name in {0: "true_entailment", 1: "true_neutral", 2: "true_contradiction"}.items():
        tt = int(by_true[t].sum().item())
        out[f"{name}_pred_entail_rate"] = (by_true[t][0].item() / tt) if tt else None
    return out


@torch.no_grad()
def eval_overlap_shortcut(model, loader, device) -> Dict:
    """Shortcut-alignment check for the N branch, on MNLI-derived data (HANS-free).

    Expects a loader from make_overlap_diagnostic_loader (high / low lexical-overlap
    groups with gold labels). An N branch that keys on the overlap feature predicts
    entailment far more often on the high-overlap group than on the low-overlap
    group (large `ent_rate_gap`), and keeps predicting entailment on high-overlap
    examples whose gold label is non-entailment
    (`high_ov_nonent_pred_entail_rate`) — the heuristic-breaking signature that
    neither a constant predictor nor a task-only predictor shows.
    """
    model.eval()
    stats = {0: [0, 0], 1: [0, 0]}   # ov_group -> [pred_entailment, total]
    hi_nonent = [0, 0]               # gold non-entailment within the high group
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"]
        groups = batch["ov_group"]
        preds = model(input_ids=ids, attention_mask=mask).logits.argmax(-1).cpu()
        for g in (0, 1):
            gm = groups == g
            stats[g][0] += (preds[gm] == 0).sum().item()
            stats[g][1] += int(gm.sum().item())
        hm = (groups == 1) & (labels != 0)
        hi_nonent[0] += (preds[hm] == 0).sum().item()
        hi_nonent[1] += int(hm.sum().item())
    low_rate = stats[0][0] / max(stats[0][1], 1)
    high_rate = stats[1][0] / max(stats[1][1], 1)
    return {
        "low_ov_pred_entail_rate": low_rate,
        "high_ov_pred_entail_rate": high_rate,
        "ent_rate_gap": high_rate - low_rate,
        "high_ov_nonent_pred_entail_rate": hi_nonent[0] / max(hi_nonent[1], 1),
        "learned_shortcut": bool((high_rate - low_rate) >= 0.30),
    }


@torch.no_grad()
def _eval_pair_accuracy(model, loader, device, key: str) -> Dict[str, float]:
    """3-class premise+hypothesis accuracy. Shared by e-SNLI / ANLI / SNLI-hard."""
    model.eval()
    correct = total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return {key: correct / max(total, 1)}


def eval_esnli(model, loader, device) -> Dict[str, float]:
    return _eval_pair_accuracy(model, loader, device, "esnli_accuracy")


def eval_anli(model, loader, device) -> Dict[str, float]:
    return _eval_pair_accuracy(model, loader, device, "anli_accuracy")


def eval_snli_hard(model, loader, device) -> Dict[str, float]:
    return _eval_pair_accuracy(model, loader, device, "snli_hard_accuracy")


def eval_wanli(model, loader, device) -> Dict[str, float]:
    return _eval_pair_accuracy(model, loader, device, "wanli_accuracy")
