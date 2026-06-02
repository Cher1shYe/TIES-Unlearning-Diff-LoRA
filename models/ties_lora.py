import math
import torch
import torch.nn as nn
from typing import Tuple, List, Dict

class TIESUnlearnLoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear.

    Contains a frozen base weight W plus two LoRA paths:
      P (positive / high-rank) — learns the full task
      N (negative / low-rank)  — captures shortcuts due to limited capacity

    The TIES-Unlearning merge in the forward pass ensures only the
    *sign-consensual, high-magnitude* shortcut components are subtracted,
    preventing the catastrophic interference of naive subtraction.
    """

    # Supported merge modes for the effective update on a TIES-enabled layer.
    #   full      : alpha*dP - beta*(trim_mask * sign_mask)*dN   (paper default)
    #   naive     : alpha*dP - beta*dN                           (no masks)
    #   sign_only : alpha*dP - beta*(sign_mask)*dN               (sign consensus only)
    #   trim_only : alpha*dP - beta*(trim_mask)*dN               (magnitude trimming only)
    #   p_only    : alpha*dP                                     (no subtraction)
    VALID_MERGE_MODES = ("full", "naive", "sign_only", "trim_only", "p_only")

    def __init__(
        self,
        base_linear: nn.Linear,
        pos_rank: int = 32,
        neg_rank: int = 4,
        alpha: float = 1.0,
        beta: float = 0.5,
        lora_alpha: int = 16,
        dropout: float = 0.1,
        trim_ratio: float = 0.2,
        merge_mode: str = "full",
    ):
        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        self.pos_rank = pos_rank
        self.neg_rank = neg_rank
        self.alpha = alpha
        self.beta = beta
        self.trim_ratio = trim_ratio
        if merge_mode not in self.VALID_MERGE_MODES:
            raise ValueError(f"merge_mode must be one of {self.VALID_MERGE_MODES}, got {merge_mode!r}.")
        self.merge_mode = merge_mode
        self.forward_mode = "eval"
        self.enable_ties = True
        self.layer_tag = None

        self.pos_scaling = lora_alpha / pos_rank
        self.neg_scaling = lora_alpha / neg_rank

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f = base_linear.in_features
        out_f = base_linear.out_features

        # High-rank path  (P — positive)
        self.lora_P_A = nn.Linear(in_f, pos_rank, bias=False)
        self.lora_P_B = nn.Linear(pos_rank, out_f, bias=False)
        # Low-rank path   (N — negative / shortcut)
        self.lora_N_A = nn.Linear(in_f, neg_rank, bias=False)
        self.lora_N_B = nn.Linear(neg_rank, out_f, bias=False)

        # Standard LoRA initialisation: A ~ Kaiming, B = 0
        nn.init.kaiming_uniform_(self.lora_P_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_P_B.weight)
        nn.init.kaiming_uniform_(self.lora_N_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_N_B.weight)

    # ----- helpers -----

    def _deltas(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return scaled (ΔP, ΔN), each shape [out_features, in_features]."""
        dP = self.lora_P_B.weight @ self.lora_P_A.weight
        dN = self.lora_N_B.weight @ self.lora_N_A.weight
        return self.pos_scaling * dP, self.neg_scaling * dN

    def _trim_mask(self, dN: torch.Tensor) -> torch.Tensor:
        """Keep only the top-`trim_ratio` fraction of ΔN by magnitude (TIES trimming)."""
        if self.trim_ratio >= 1.0:
            return torch.ones_like(dN)
        dN_abs = dN.abs()
        k = max(1, int(dN.numel() * self.trim_ratio))
        threshold = torch.topk(dN_abs.view(-1), k).values[-1]
        return (dN_abs >= threshold).float()

    def _sign_mask(self, dP: torch.Tensor, dN: torch.Tensor) -> torch.Tensor:
        """1 where ΔP and ΔN agree in sign (TIES sign-consensus / Elect-Sign)."""
        return (dP.sign() == dN.sign()).float()

    def _merge_delta(self, dP: torch.Tensor, dN: torch.Tensor) -> torch.Tensor:
        """Build the effective weight update according to ``self.merge_mode``.

        ``full`` is the paper default (trim + sign). The other modes exist so the
        ablation / baseline drivers can isolate the contribution of each mask, and
        so that a *true* naive task-vector subtraction (no masks) can be produced —
        something the original single ``_ties_unlearn_merge`` could not express.
        """
        mode = self.merge_mode
        if mode == "p_only":
            return self.alpha * dP
        if mode == "naive":
            mask = torch.ones_like(dN)
        elif mode == "sign_only":
            mask = self._sign_mask(dP, dN)
        elif mode == "trim_only":
            mask = self._trim_mask(dN)
        elif mode == "full":
            mask = self._trim_mask(dN) * self._sign_mask(dP, dN)
        else:
            raise ValueError(f"Unknown merge_mode: {mode!r}")
        return self.alpha * dP - self.beta * mask.detach() * dN

    # Backwards-compatible alias (older call sites / external scripts).
    def _ties_unlearn_merge(self, dP: torch.Tensor, dN: torch.Tensor) -> torch.Tensor:
        return self._merge_delta(dP, dN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        x_drop = self.dropout(x)

        dP, dN = self._deltas()

        if self.forward_mode == "phase1":
            return base_out + torch.matmul(x_drop, dP.t())
        if self.forward_mode == "phase2":
            return base_out + torch.matmul(x_drop, dN.t())

        # Layer not selected for TIES → fall back to P-only (used by layer localization).
        if not self.enable_ties:
            dEff = self.alpha * dP
        else:
            dEff = self._merge_delta(dP, dN)
        return base_out + torch.matmul(x_drop, dEff.t())

    # ----- introspection -----

    def get_pos_params(self) -> List[nn.Parameter]:
        return list(self.lora_P_A.parameters()) + list(self.lora_P_B.parameters())

    def get_neg_params(self) -> List[nn.Parameter]:
        return list(self.lora_N_A.parameters()) + list(self.lora_N_B.parameters())

    def weight_norms(self) -> Dict[str, float]:
        with torch.no_grad():
            dP, dN = self._deltas()
            return {
                "dP_norm": dP.norm().item(),
                "dN_norm": dN.norm().item(),
            }

def set_model_forward_mode(model: nn.Module, mode: str):
    """Switch all TIESUnlearnLoRALinear modules to a given forward mode."""
    for module in model.modules():
        if isinstance(module, TIESUnlearnLoRALinear):
            module.forward_mode = mode

def set_model_merge_mode(model: nn.Module, merge_mode: str):
    """Switch the TIES merge mode (full / naive / sign_only / trim_only / p_only)
    on every TIESUnlearnLoRALinear module."""
    for module in model.modules():
        if isinstance(module, TIESUnlearnLoRALinear):
            if merge_mode not in module.VALID_MERGE_MODES:
                raise ValueError(
                    f"merge_mode must be one of {module.VALID_MERGE_MODES}, got {merge_mode!r}."
                )
            module.merge_mode = merge_mode