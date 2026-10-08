"""Mutual information I(X;Y) of a trained transmitter versus SNR (Fig. 9).

For every SNR a fresh MI estimation model T is trained on the frozen
transmitter output with the bound of eq. (9); the reported value is the
average lower bound of the last steps, in bits per complex symbol.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepsc.channels import apply_channel  # noqa: E402
from deepsc.data import SentenceDataset, load_corpus  # noqa: E402
from deepsc.engine import load_checkpoint  # noqa: E402
from deepsc.models import MINE, complex_pairs, mi_lower_bound  # noqa: E402


def estimate(model, batches, channel, snr, steps, lr, kw):
    mine = MINE().to(batches[0].device)
    opt = torch.optim.Adam(mine.parameters(), lr=lr)
    vals = []
    for i in range(steps):
        src = batches[i % len(batches)]
        with torch.no_grad():
            x, mask = model.transmit(src)
            y = apply_channel(channel, x, snr, **kw) * mask
        xs, ys = complex_pairs(x, y, mask, max_samples=8192)
        lb = mi_lower_bound(mine, xs, ys)
        opt.zero_grad()
        (-lb).backward()
        opt.step()
        if i >= steps - steps // 5:
            vals.append(lb.item())
    return sum(vals) / len(vals) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/europarl_en.json")
    ap.add_argument("--ckpts", required=True, help="comma separated name=path")
    ap.add_argument("--channel", default="awgn")
    ap.add_argument("--snrs", default="0,3,6,9,12,15,18")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="results/mutual_information.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.threads:
        torch.set_num_threads(a.threads)
    _, test, vocab = load_corpus(a.data)
    ds = SentenceDataset(test, vocab)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = [ds.collate([ds[j] for j in range(i, min(i + 128, len(ds)))]) for i in range(0, len(ds), 128)]
    batches = [b.to(dev) for b in batches]
    snrs = [float(s) for s in a.snrs.split(",")]
    res = {"snrs": snrs, "gaussian_capacity": [math.log2(1 + 10 ** (s / 10)) for s in snrs]}
    for item in a.ckpts.split(","):
        name, path = item.split("=")
        model, _, _ = load_checkpoint(path, vocab)
        model = model.to(dev)
        model.eval()
        res[name] = []
        for snr in snrs:
            mi = estimate(model, batches, a.channel, snr, a.steps, a.lr, {})
            res[name].append(mi)
            print(f"{name:20s} SNR {snr:4.1f} dB  I(X;Y) = {mi:.3f} bit/symbol "
                  f"(Gaussian capacity {math.log2(1 + 10 ** (snr / 10)):.3f})", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
