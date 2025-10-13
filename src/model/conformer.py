from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv_out_len(l: torch.Tensor, k: int, s: int, p: int = 1, d: int = 1):
    return torch.floor((l + 2 * p - d * (k - 1) - 1) / s + 1)

def lengths_after_subsampling(lens: torch.Tensor,
                              kernel_stride_pairs=((3, 2), (3, 2)),
                              padding: int = 1) -> torch.Tensor:
    out = lens.to(torch.float32).clone()
    for k, s in kernel_stride_pairs:
        out = _conv_out_len(out, k, s, p=padding)
    return out.to(torch.long)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2dSubsampling4(nn.Module):
    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear((n_mels // 4) * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, W, C * H)
        x = self.out(x)
        return x

class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        x_ln = self.ln(x)
        out, _ = self.attn(x_ln, x_ln, x_ln, key_padding_mask=key_padding_mask, need_weights=False)
        return out

class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1, padding=0)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                   padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ln = self.ln(x)
        y = x_ln.transpose(1, 2)
        y = self.pointwise1(y)
        y = self.glu(y)
        y = self.depthwise(y)
        y = self.bn(y)
        y = self.swish(y)
        y = self.pointwise2(y)
        y = y.transpose(1, 2)
        return self.dropout(y)

class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        macaron_scale: float = 0.5,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mhsa = MHSA(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.final_ln = nn.LayerNorm(d_model)
        self.macaron_scale = macaron_scale

    def forward(self, x, key_padding_mask=None):
        x = x + self.macaron_scale * self.ff1(x)
        x = x + self.dropout(self.mhsa(x, key_padding_mask=key_padding_mask))
        x = x + self.conv(x)
        x = x + self.macaron_scale * self.ff2(x)
        return self.final_ln(x)

class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 12,
        d_ff_mult: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        subsample: bool = True,
    ):
        super().__init__()
        self.subsample = subsample
        if subsample:
            self.sub = Conv2dSubsampling4(n_mels, d_model)
        else:
            self.in_proj = nn.Linear(n_mels, d_model)

        d_ff = d_model * d_ff_mult
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, d_ff, conv_kernel, dropout)
            for _ in range(num_layers)
        ])

        self.pos = SinusoidalPositionalEncoding(d_model)

    def forward(self, feats: torch.Tensor, feats_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.subsample:
            x = self.sub(feats)
            x = self.pos(x)
            out_len = lengths_after_subsampling(feats_len)
        else:
            x = feats.transpose(1, 2)
            x = self.in_proj(x)
            x = self.pos(x)
            out_len = feats_len

        out_len = out_len.to(x.device)

        max_t = x.size(1)
        device = x.device
        rng = torch.arange(max_t, device=device)[None, :]
        key_padding_mask = rng >= out_len[:, None]

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return x, out_len

class CTCProjection(nn.Module):
    def __init__(self, d_model: int, n_tokens: int):
        super().__init__()
        self.lin = nn.Linear(d_model, n_tokens)

    def forward(self, x):
        return F.log_softmax(self.lin(x), dim=-1)

class ConformerCTC(nn.Module):
    def __init__(
        self,
        n_feats: int = 80,
        n_tokens: int = 40,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 12,
        d_ff_mult: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        subsample: bool = True
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            n_mels=n_feats,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            d_ff_mult=d_ff_mult,
            conv_kernel=conv_kernel,
            dropout=dropout,
            subsample=subsample,
        )
        self.ctc = CTCProjection(d_model, n_tokens)

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch):
        enc, out_len = self.encoder(spectrogram, spectrogram_length)
        log_probs = self.ctc(enc)
        return {
            "log_probs": log_probs,
            "log_probs_length": out_len,
        }
    