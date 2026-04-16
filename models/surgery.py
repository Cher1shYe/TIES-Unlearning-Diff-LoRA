import copy
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

from configs.config import LoRAConfig
from models.ties_lora import TIESUnlearnLoRALinear

def _resolve_parent(root: nn.Module, parts: List[str]) -> nn.Module:
    parent = root
    for p in parts:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent

def _set_child(parent: nn.Module, name: str, module: nn.Module):
    if name.isdigit():
        parent[int(name)] = module
    else:
        setattr(parent, name, module)

def _extract_layer_tag(full_name: str) -> Optional[str]:
    parts = full_name.split(".")
    for idx in range(len(parts) - 2):
        if parts[idx] == "encoder" and parts[idx + 1] == "layer" and parts[idx + 2].isdigit():
            return ".".join(parts[idx:idx + 3])
    for idx in range(len(parts) - 1):
        if parts[idx] == "layer" and parts[idx + 1].isdigit():
            start = idx - 1 if idx > 0 else idx
            return ".".join(parts[start:idx + 2])
    return None

def _layer_index_from_tag(layer_tag: str) -> int:
    try:
        return int(layer_tag.split(".")[-1])
    except (AttributeError, TypeError, ValueError):
        return 10 ** 9

def inject_ties_unlearn_lora(
    model: nn.Module,
    target_keywords: List[str],
    cfg: LoRAConfig,
) -> Tuple[nn.Module, List[str]]:
    """Replace matching nn.Linear layers with TIESUnlearnLoRALinear."""
    replaced: List[str] = []
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(k.lower() in full_name.lower() for k in target_keywords):
            continue

        parts = full_name.split(".")
        parent = _resolve_parent(model, parts[:-1])

        new_layer = TIESUnlearnLoRALinear(
            base_linear=module,
            pos_rank=cfg.pos_rank,
            neg_rank=cfg.neg_rank,
            alpha=cfg.alpha,
            beta=cfg.beta,
            lora_alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            trim_ratio=cfg.trim_ratio,
        )
        new_layer.layer_tag = _extract_layer_tag(full_name) or full_name
        _set_child(parent, parts[-1], new_layer)
        replaced.append(full_name)

    return model, replaced

def get_ties_modules_by_layer(model: nn.Module) -> Dict[str, List[TIESUnlearnLoRALinear]]:
    grouped: Dict[str, List[TIESUnlearnLoRALinear]] = {}
    for _name, module in model.named_modules():
        if not isinstance(module, TIESUnlearnLoRALinear):
            continue
        layer_tag = module.layer_tag or "unassigned"
        grouped.setdefault(layer_tag, []).append(module)
    return dict(sorted(grouped.items(), key=lambda item: _layer_index_from_tag(item[0])))

def configure_ties_layers(model: nn.Module, shortcut_layers: List[str]):
    selected = set(shortcut_layers)
    for layer_tag, modules in get_ties_modules_by_layer(model).items():
        enable = layer_tag in selected
        for module in modules:
            module.enable_ties = enable

@torch.no_grad()
def merge_and_unload(model: nn.Module) -> nn.Module:
    """Materialize all TIES-Unlearn LoRA layers into plain nn.Linear."""
    merged = copy.deepcopy(model)
    for full_name, module in list(merged.named_modules()):
        if not isinstance(module, TIESUnlearnLoRALinear):
            continue
        dP, dN = module._deltas()
        if module.enable_ties:
            dEff = module._ties_unlearn_merge(dP, dN)
        else:
            dEff = module.alpha * dP

        base = module.base_linear
        lin = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)
        lin = lin.to(device=base.weight.device, dtype=base.weight.dtype)
        lin.weight.data.copy_(base.weight.data + dEff.to(dtype=base.weight.dtype))
        if base.bias is not None:
            lin.bias.data.copy_(base.bias.data)

        parts = full_name.split(".")
        parent = _resolve_parent(merged, parts[:-1])
        _set_child(parent, parts[-1], lin)
    return merged