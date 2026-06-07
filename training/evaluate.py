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
