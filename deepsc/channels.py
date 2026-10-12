"""Physical channel layers (Sec. III-A, eq. (2)): y = h x + n.

Symbols are complex tensors. All channels are differentiable so that the
transmitter can be trained end-to-end through them. For the fading channels
perfect CSI is assumed (as in Sec. V) and the receiver applies zero-forcing
equalisation y / h. The fading coefficient is constant over one sentence
(block fading, coherence time = M symbols).
"""
import math

import torch


def snr_to_noise_std(snr_db):
    """Per-real-dimension noise std for unit-power complex symbols."""
    return 1.0 / math.sqrt(2.0 * 10 ** (snr_db / 10.0))


def power_normalize(x, mask=None):
    """Scale every sentence's symbols to unit average power E|x|^2 = 1.

    x: complex (B, N); mask: bool (B, N) of symbols that are really sent.
    """
    if mask is None:
        mask = torch.ones_like(x.real, dtype=torch.bool)
    m = mask.to(x.real.dtype)
    p = (x.abs() ** 2 * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
    return x / torch.sqrt(p + 1e-12) * m


def _noise(x, snr_db):
    std = snr_to_noise_std(snr_db)
    return std * torch.complex(torch.randn_like(x.real), torch.randn_like(x.real))


def awgn(x, snr_db):
    return x + _noise(x, snr_db)


def _cn(shape, device):
    return torch.complex(torch.randn(shape, device=device), torch.randn(shape, device=device)) / math.sqrt(2)


def rayleigh(x, snr_db):
    h = _cn((x.shape[0], 1), x.device)  # h ~ CN(0, 1), one per sentence
    return (h * x + _noise(x, snr_db)) / h


def rician(x, snr_db, k=1.0):
    h = math.sqrt(k / (k + 1)) + math.sqrt(1 / (k + 1)) * _cn((x.shape[0], 1), x.device)
    return (h * x + _noise(x, snr_db)) / h


def erasure(x, snr_db, p=0.1):
    """AWGN followed by symbol erasures with probability p (erased -> 0)."""
    keep = (torch.rand_like(x.real) >= p).to(x.real.dtype)
    return awgn(x, snr_db) * keep


CHANNELS = {"awgn": awgn, "rayleigh": rayleigh, "rician": rician, "erasure": erasure}


def apply_channel(name, x, snr_db, **kw):
    return CHANNELS[name.lower()](x, snr_db, **kw)
