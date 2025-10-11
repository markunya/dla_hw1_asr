import torch

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
