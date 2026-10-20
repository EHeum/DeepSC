"""Physical channel layers (Sec. III-A, eq. (2)): y = h x + n.

Symbols are complex tensors. All channels are differentiable so that the
transmitter can be trained end-to-end through them. For the fading channels
perfect CSI is assumed (as in Sec. V) and the receiver applies zero-forcing
equalisation y / h. The fading coefficient is constant over one sentence
(block fading, coherence time = M symbols).

``ofdm`` puts the symbols on the subcarriers of CP-OFDM and sends them through a
frequency-selective multipath channel (see its docstring).
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


def rayleigh_iid(x, snr_db):
    """Fast (i.i.d.) Rayleigh fading: every symbol gets its own h ~ CN(0, 1), ZF with perfect CSI.

    Each OFDM subcarrier gain is also CN(0, 1), so this channel has the same per-symbol statistics
    as ``ofdm`` and differs only in that the gains are uncorrelated.
    """
    h = _cn(x.shape, x.device)
    return (h * x + _noise(x, snr_db)) / h


def rician(x, snr_db, k=1.0):
    h = math.sqrt(k / (k + 1)) + math.sqrt(1 / (k + 1)) * _cn((x.shape[0], 1), x.device)
    return (h * x + _noise(x, snr_db)) / h


def erasure(x, snr_db, p=0.1):
    """AWGN followed by symbol erasures with probability p (erased -> 0)."""
    keep = (torch.rand_like(x.real) >= p).to(x.real.dtype)
    return awgn(x, snr_db) * keep


def _bit_reversal(n):
    bits = int(math.log2(n))
    assert 2 ** bits == n, "number of subcarriers must be a power of two"
    return [int(format(i, f"0{bits}b")[::-1], 2) for i in range(n)]


def multipath_taps(batch, n_taps=8, decay=2.0, device="cpu"):
    """Rayleigh taps with an exponential power-delay profile, E[sum |h_l|^2] = 1."""
    pdp = torch.exp(-torch.arange(n_taps, dtype=torch.float32, device=device) / decay)
    pdp = pdp / pdp.sum()
    return _cn((batch, n_taps), device) * pdp.sqrt()


def ofdm(x, snr_db, n_sc=64, cp=16, n_taps=8, decay=2.0, eq="zf", mapping="contiguous", taps=None):
    """CP-OFDM over a multipath channel, simulated in the time domain.

    x (B, N) complex -> equalised subcarrier symbols (B, N), in the same order.

    * The N symbols fill ceil(N / n_sc) OFDM symbols of n_sc subcarriers (unused ones carry 0).
      ``mapping="contiguous"`` puts consecutive symbols (e.g. one word's) on adjacent subcarriers,
      ``"interleaved"`` spreads them with a bit-reversal permutation (frequency diversity).
    * IFFT (unitary) + cyclic prefix, linear convolution with ``n_taps`` Rayleigh taps
      (exponential power-delay profile, constant over the sentence), AWGN in the time domain,
      CP removal + FFT. With n_taps - 1 <= cp there is no inter-symbol interference, so each
      subcarrier k sees Y = H_k X + N, N ~ CN(0, N0) and the average subcarrier SNR is 1 / N0.
    * Perfect CSI; ``eq="zf"`` gives Y / H_k, ``eq="mmse"`` gives conj(H_k) Y / (|H_k|^2 + N0).
    """
    assert n_taps - 1 <= cp, "the cyclic prefix must cover the channel delay spread"
    b, n = x.shape
    n_ofdm = -(-n // n_sc)
    grid = torch.cat([x, x.new_zeros(b, n_ofdm * n_sc - n)], 1).reshape(b, n_ofdm, n_sc)
    perm = None
    if mapping == "interleaved":
        perm = torch.tensor(_bit_reversal(n_sc), device=x.device)
        g = torch.empty_like(grid)
        g[..., perm] = grid
        grid = g
    s = torch.fft.ifft(grid, norm="ortho")
    s = torch.cat([s[..., -cp:], s], -1).reshape(b, -1)  # add CP, serialise
    h = multipath_taps(b, n_taps, decay, x.device) if taps is None else taps
    r = h[:, :1] * s
    for tap in range(1, n_taps):
        r = r + h[:, tap:tap + 1] * torch.cat([s.new_zeros(b, tap), s[:, :-tap]], 1)
    r = r + _noise(r, snr_db)
    y = torch.fft.fft(r.reshape(b, n_ofdm, n_sc + cp)[..., cp:], norm="ortho")
    hk = torch.fft.fft(h, n=n_sc)[:, None, :]  # frequency response per subcarrier
    if eq == "mmse":
        n0 = 10 ** (-snr_db / 10)
        y = y * hk.conj() / (hk.abs() ** 2 + n0)
    else:
        y = y / hk
    if perm is not None:
        y = y[..., perm]
    return y.reshape(b, -1)[:, :n]


CHANNELS = {"awgn": awgn, "ofdm": ofdm, "rayleigh_iid": rayleigh_iid, "rayleigh": rayleigh, "rician": rician, "erasure": erasure}


def apply_channel(name, x, snr_db, **kw):
    return CHANNELS[name.lower()](x, snr_db, **kw)
