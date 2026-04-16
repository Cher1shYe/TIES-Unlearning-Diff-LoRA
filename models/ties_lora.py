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

    def _ties_unlearn_merge(
        self, dP: torch.Tensor, dN: torch.Tensor
    ) -> torch.Tensor:
        # 1) Magnitude trimming on ΔN
        dN_abs = dN.abs()
        if self.trim_ratio < 1.0:
            k = max(1, int(dN.numel() * self.trim_ratio))
            threshold = torch.topk(dN_abs.view(-1), k).values[-1]
            trim_mask = (dN_abs >= threshold).float()
        else:
            trim_mask = torch.ones_like(dN)

        # 2) ELECT SIGN
        sign_mask = (dP.sign() == dN.sign()).float()

        mask = (trim_mask * sign_mask).detach()
        return self.alpha * dP - self.beta * mask * dN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        x_drop = self.dropout(x)

        dP, dN = self._deltas()

        if self.forward_mode == "phase1":
            return base_out + torch.matmul(x_drop, dP.t())
        if self.forward_mode == "phase2":
            return base_out + torch.matmul(x_drop, dN.t())

        if not self.enable_ties:
            dEff = self.alpha * dP
        else:
            dEff = self._ties_unlearn_merge(dP, dN)
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