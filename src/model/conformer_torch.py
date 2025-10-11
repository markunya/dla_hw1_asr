# src/model/ta_conformer.py
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchaudio.models import Conformer as TAConformer
except Exception as e:
    raise ImportError(
        "torchaudio.models.Conformer недоступен. "
        "Убедись, что установлен совместимый torchaudio."
    ) from e


def lengths_after_subsampling(lens: torch.Tensor,
                              kernel_stride_pairs=((3, 2), (3, 2))) -> torch.Tensor:
    out = lens.clone().to(torch.float32)
    for k, s in kernel_stride_pairs:
        out = torch.floor((out + 2 * 1 - (k - 1) - 1) / s + 1)
    return out.to(torch.long)


class Conv2dSubsampling4(nn.Module):
    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear((n_mels // 4) * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, W, C * H)
        x = self.proj(x)
        return x


class CTCProjection(nn.Module):
    def __init__(self, d_model: int, n_tokens: int):
        super().__init__()
        self.lin = nn.Linear(d_model, n_tokens)

    def forward(self, x):
        return F.log_softmax(self.lin(x), dim=-1)


class TorchaudioConformerCTC(nn.Module):
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
        subsample: bool = True,
    ):
        super().__init__()
        self.subsample = subsample
        if subsample:
            self.sub = Conv2dSubsampling4(n_feats, d_model)
            conformer_input_dim = d_model
        else:
            self.in_proj = nn.Linear(n_feats, d_model)
            conformer_input_dim = d_model

        self.encoder = TAConformer(
            input_dim=conformer_input_dim,
            num_heads=n_heads,
            ffn_dim=d_model * d_ff_mult,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel,
            dropout=dropout,
            use_group_norm=True,
            convolution_first=False,
        )

        self.ctc = CTCProjection(d_model, n_tokens)

    def _make_pad_mask(self, out_len: torch.Tensor, max_t: int) -> torch.Tensor:
        rng = torch.arange(max_t, device=out_len.device)[None, :]
        return rng >= out_len[:, None]

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch):
        """
        spectrogram: [B, n_mels, T]
        spectrogram_length: [B]
        """
        if self.subsample:
            x = self.sub(spectrogram)                          # [B, T', d_model]
            out_len = lengths_after_subsampling(spectrogram_length)
        else:
            x = spectrogram.transpose(1, 2)                    # [B, T, n_mels]
            x = self.in_proj(x)                                 # [B, T, d_model]
            out_len = spectrogram_length

        # длины должны быть int64 на том же устройстве
        out_len = out_len.to(device=x.device, dtype=torch.int64)

        # Маска на всякий случай (для новых версий API)
        pad_mask = self._make_pad_mask(out_len, x.size(1))     # [B, T'], dtype=bool

        # Разные версии torchaudio имеют разный forward; пробуем по порядку:
        enc = None
        # 1) Частая сигнатура: forward(x, lengths) -> (enc, out_lengths)
        try:
            enc_out = self.encoder(x, out_len)
            if isinstance(enc_out, tuple):
                enc, _ = enc_out
            else:
                enc = enc_out
        except TypeError:
            pass

        # 2) Новая сигнатура: forward(x, src_key_padding_mask=pad_mask) -> enc
        if enc is None:
            try:
                enc = self.encoder(x, src_key_padding_mask=pad_mask)
            except TypeError:
                # 3) Ещё один вариант: forward(x, lengths=out_len)
                enc_out = self.encoder(x, lengths=out_len)
                if isinstance(enc_out, tuple):
                    enc, _ = enc_out
                else:
                    enc = enc_out

        log_probs = self.ctc(enc)                              # [B, T', vocab]
        return {
            "log_probs": log_probs,
            "log_probs_length": out_len,                        # уже на нужном девайсе и int64
        }
