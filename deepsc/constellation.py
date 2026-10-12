"""Constellation-constrained (digital) channel input, following DeepJSCC-Q.

T.-Y. Tung, D. B. Kurka, M. Jankowski, D. Gunduz, "DeepJSCC-Q: Constellation
constrained deep joint source-channel coding," IEEE JSAIT, 2022.

The channel encoder output z (complex, unit average power) is mapped onto a
fixed M-ary constellation C = {c_1..c_M}:

    soft  : x~ = sum_k softmax_k(-|z - c_k|^2 / (T d_min^2)) c_k     (differentiable)
    hard  : x^ = c_{argmin_k |z - c_k|^2}                             (what is transmitted)
    train : x  = x~ + stop_grad(x^ - x~)   (forward = hard, backward = soft)

so the transmitted symbols are always exact constellation points (a standard
modem can send them as log2(M)-bit labels), while the encoder still receives
gradients. Optionally the receiver makes hard symbol decisions (detect) before
the channel decoder, which makes the link fully digital.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from .baselines import QAM


def constellation_points(name):
    """'qam4' | 'qam16' | 'qam64' | 'qam256' | 'psk8' ... -> unit average power complex points."""
    name = name.lower()
    if name.startswith("qam"):
        pts = QAM(int(name[3:])).points
    elif name.startswith("psk"):
        m = int(name[3:])
        pts = np.exp(1j * (2 * np.pi * np.arange(m) / m + (np.pi / 4 if m == 4 else 0)))
    else:
        raise ValueError(f"unknown constellation {name}")
    return torch.tensor(pts, dtype=torch.complex64)


class ConstellationMapper(nn.Module):
    def __init__(self, name="qam16", temperature=0.5, ste="soft"):
        super().__init__()
        pts = constellation_points(name)
        # non-persistent: checkpoints of the analog model load unchanged
        self.register_buffer("points", pts, persistent=False)
        d = (pts[:, None] - pts[None]).abs()
        self.d_min2 = float(d[d > 0].min() ** 2)
        self.name, self.temperature, self.ste = name, temperature, ste
        self.bits = math.log2(len(pts))

    def nearest(self, z):
        return (z[..., None] - self.points).abs().argmin(-1)

    def forward(self, z):
        idx = self.nearest(z)
        hard = self.points[idx]
        if not self.training:
            return hard
        if self.ste == "identity":  # straight-through with identity gradient
            return z + (hard - z).detach()
        d2 = (z[..., None] - self.points).abs() ** 2
        p = torch.softmax(-d2 / (self.temperature * self.d_min2), -1)
        soft = (p * self.points).sum(-1)
        return soft + (hard - soft).detach()

    def detect(self, y):
        """Hard-decision demodulation at the receiver (ML detection for AWGN)."""
        return self.points[self.nearest(y)]
