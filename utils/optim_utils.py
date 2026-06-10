import os
import torch
import torch.nn as nn
from typing import Dict, Tuple
from torch.optim import AdamW

from models.ties_lora import TIESUnlearnLoRALinear

# Compatibility: torch.cuda.amp location changed in newer PyTorch
try:
    from torch.amp import GradScaler          # PyTorch >= 2.4
except ImportError:
    from torch.cuda.amp import GradScaler      # PyTorch < 2.4

def _split_params(model):
    """Split trainable params into (pos, neg, head) groups."""
    pos, neg, head = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_N_" in n:
            neg.append(p)
        elif "lora_P_" in n:
            pos.append(p)
        else:
            head.append(p)
    return pos, neg, head

def _neg_l2(model) -> torch.Tensor:
    """L2 regularization on the negative LoRA parameters."""
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if p.requires_grad and "lora_N_" in n:
            reg = reg + p.pow(2).mean()
    return reg

def _make_scaler(fp16: bool, device):
    if not fp16 or device.type != "cuda":
        return GradScaler(enabled=False)
    return GradScaler(enabled=True)

def _amp_enabled(fp16: bool, device):
    return fp16 and device.type == "cuda"

def _log_lora_norms(model):
    """Print weight norms for all TIES-Unlearn layers (debug).
    Returns a mean-norm summary so callers can persist it in metrics."""
    dp, dn = [], []
    for name, module in model.named_modules():
        if isinstance(module, TIESUnlearnLoRALinear):
            norms = module.weight_norms()
            dp.append(norms["dP_norm"])
            dn.append(norms["dN_norm"])
            print(f"  {name}: dP={norms['dP_norm']:.4f}  dN={norms['dN_norm']:.4f}")
    return {
        "dP_norm_mean": sum(dp) / max(len(dp), 1),
        "dN_norm_mean": sum(dn) / max(len(dn), 1),
    }

def _save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    phase: int,
    path: str,
    history: Dict,
    phase_metrics: Dict,
):
    """Saves the training state to a checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "history": history,
        "phase_metrics": phase_metrics,
    }
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved checkpoint for Phase {phase}, Epoch {epoch} to {path}")

def _load_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    path: str,
    device: torch.device,
) -> Tuple[int, int, Dict, Dict]:
    """Loads the training state from a checkpoint file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Optimizer and scheduler state dicts will be loaded into newly created instances
    # within the training loop for the current phase.
    # For now, we only load them to ensure consistency in the returned data, but
    # they will be re-initialized for the *next* phase.
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"[Checkpoint] Loaded checkpoint from {path}")
    return (
        checkpoint["epoch"],
        checkpoint["phase"],
        checkpoint["history"],
        checkpoint["phase_metrics"],
    )