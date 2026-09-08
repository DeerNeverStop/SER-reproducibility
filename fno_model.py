"""
fno_model.py
============
A 1-D Fourier Neural Operator classifier for speech emotion recognition.

Input  : log-mel spectrogram, shape (batch, n_mels, time)
         -> the n_mels frequency bins are treated as input channels,
            and the Fourier transform is taken along the time axis.
Output : emotion logits, shape (batch, n_classes)

The input has already passed through STFT and mel filtering. SpectralConv1d
then applies a learned Fourier operator along the log-mel *time* axis: it keeps
low temporal modulation modes, multiplies them by learned complex weights, and
returns to the time domain with an inverse FFT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Learnable Fourier layer (truncated to `modes` low-frequency modes)."""

    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        # complex weights R: one (in_ch x out_ch) matrix per kept Fourier mode
        self.weight = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):                                  # (B, in_ch, L)
        B, _, L = x.shape
        x_ft = torch.fft.rfft(x)                           # (B, in_ch, L//2+1)
        m = min(self.modes, x_ft.size(-1))
        out_ft = torch.zeros(B, self.out_ch, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        # multiply the kept low modes by the learnable weights
        out_ft[:, :, :m] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=L)                # (B, out_ch, L)


class FourierBlock(nn.Module):
    """One FNO layer: spectral conv + pointwise residual + GELU."""

    def __init__(self, width: int, modes: int, dropout: float = 0.1):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.bypass = nn.Conv1d(width, width, 1)           # the W (skip) term
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.gelu(self.spectral(x) + self.bypass(x)))


class FNOClassifier(nn.Module):
    """mel-spectrogram -> lift -> FNO blocks -> global pool -> emotion logits."""

    def __init__(self, n_mels: int = 64, n_classes: int = 8, width: int = 32,
                 modes: int = 16, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.lift = nn.Conv1d(n_mels, width, 1)            # lift channels -> width
        self.blocks = nn.ModuleList(
            [FourierBlock(width, modes, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, n_classes))

    def forward(self, x):                                  # (B, n_mels, T)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=-1)                                 # global average pool over time
        return self.head(x)
