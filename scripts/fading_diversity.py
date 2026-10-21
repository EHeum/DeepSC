"""Channel-only view of the fading / correlation losses (docs/03_ofdm.md).

For the 8 complex symbols that carry one word, look at their channel power gains |h|^2 under
block Rayleigh, i.i.d. Rayleigh and CP-OFDM with contiguous or interleaved mapping. Every
channel has E|h|^2 = 1 per symbol; they differ only in how the 8 gains are correlated, i.e.
in how much diversity one word gets.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepsc.channels import _bit_reversal, _cn, multipath_taps  # noqa: E402


def word_gains(kind, n_words, n_sym, n_sc, n_taps, decay):
    if kind == "block Rayleigh":
        return (_cn((n_words, 1), "cpu").abs() ** 2).expand(n_words, n_sym)
    if kind == "i.i.d. Rayleigh":
        return _cn((n_words, n_sym), "cpu").abs() ** 2
    hk = torch.fft.fft(multipath_taps(n_words, n_taps, decay), n=n_sc).abs() ** 2  # (W, n_sc)
    start = torch.randint(0, n_sc // n_sym, (n_words,)) * n_sym  # word-aligned slot in the OFDM symbol
    pos = start[:, None] + torch.arange(n_sym)
    if kind == "OFDM interleaved":
        pos = torch.tensor(_bit_reversal(n_sc))[pos]
    return hk.gather(1, pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=200000)
    ap.add_argument("--symbols-per-word", type=int, default=8)
    ap.add_argument("--out", default="results/ofdm/word_diversity.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    res = {}
    for kind in ["i.i.d. Rayleigh", "OFDM interleaved", "OFDM contiguous", "block Rayleigh"]:
        g = word_gains(kind, a.words, a.symbols_per_word, 64, 8, 2.0)
        mean = g.mean(1)
        c = torch.corrcoef(torch.stack([g[:, 0], g[:, 1]]))[0, 1].item()
        # diversity order of the word's mean gain: 1 / (normalised variance), = n_sym for i.i.d. gains
        div = (mean.mean() ** 2 / mean.var()).item()
        res[kind] = {
            "corr_adjacent_symbols": c,
            "effective_diversity": div,
            "p_word_mean_gain_below_0.1": (mean < 0.1).float().mean().item(),
            "p_word_mean_gain_below_0.3": (mean < 0.3).float().mean().item(),
            "p_symbol_gain_below_0.1": (g < 0.1).float().mean().item(),
        }
        r = res[kind]
        print(f"{kind:17s} corr(adjacent)={c:5.2f}  diversity={div:5.2f}  "
              f"P(word mean|h|^2<0.1)={r['p_word_mean_gain_below_0.1']:.4f}  "
              f"P(<0.3)={r['p_word_mean_gain_below_0.3']:.4f}  P(symbol<0.1)={r['p_symbol_gain_below_0.1']:.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    assert math.isfinite(res["i.i.d. Rayleigh"]["effective_diversity"])


if __name__ == "__main__":
    main()
