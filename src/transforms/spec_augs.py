import torch
import random
from torch import nn
from typing import Optional, Literal

class LogClamp:
    def __init__(self, eps: float = 1e-5):
        self.eps = eps
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(self.eps).log()
    
class Normalize:
    def __init__(self, eps: float = 1e-5):
        self.eps = eps
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std

class FreqCutout(nn.Module):
    def __init__(
        self,
        max_width: int = 8,
        num_masks: int = 2,
        p: float = 1.0,
    ):
        super().__init__()
        assert max_width >= 1
        assert num_masks >= 1
        assert 0.0 <= p <= 1.0
        self.max_width = max_width
        self.num_masks = num_masks
        self.p = p

    def _apply_one(self, spec: torch.Tensor) -> torch.Tensor:
        n_mels, _ = spec.shape
        out = spec.clone()

        fill_val = out.mean(dim=1, keepdim=True)
        for _ in range(self.num_masks):
            w = random.randint(0, min(self.max_width, n_mels))
            if w == 0:
                continue
            f0 = random.randint(0, n_mels - w)
            if isinstance(fill_val, torch.Tensor):
                out[f0:f0 + w, :] = fill_val[f0:f0 + w]
            else:
                out[f0:f0 + w, :] = fill_val
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x

        if x.dim() == 2:
            return self._apply_one(x)
        elif x.dim() == 3:
            return torch.stack([self._apply_one(s) for s in x], dim=0)
        else:
            raise ValueError(f"Expected [n_mels, T] or [B, n_mels, T], got {tuple(x.shape)}")
