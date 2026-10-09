"""BLEU (1-4 gram) versus SNR of trained DeepSC models (Fig. 6) + sample sentences (Table II) + runtime (Table III)."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepsc.data import SentenceDataset, load_corpus  # noqa: E402
from deepsc.engine import evaluate_bleu, load_checkpoint, transmit_sentences  # noqa: E402


@torch.no_grad()
def symbol_stats(model, sentences, vocab):
    """Average power and, for a digital model, how the constellation is used."""
    model.eval()
    dev = next(model.parameters()).device
    ds = SentenceDataset(sentences, vocab)
    xs = []
    for i in range(0, len(ds), 256):
        src = ds.collate([ds[j] for j in range(i, min(i + 256, len(ds)))]).to(dev)
        x, mask = model.transmit(src)
        xs.append(x[mask].cpu())
    x = torch.cat(xs)
    st = {"avg_power": float((x.abs() ** 2).mean()), "peak_to_avg_power": float((x.abs() ** 2).max() / (x.abs() ** 2).mean())}
    if model.mapper is not None:
        pts = model.mapper.points.cpu()
        dist = (x[:, None] - pts).abs()
        idx = dist.argmin(1)
        st["on_constellation"] = float((dist.min(1).values < 1e-5).float().mean())
        p = torch.bincount(idx, minlength=len(pts)).float() / len(idx)
        st["used_points"] = int((p > 0).sum())
        st["label_entropy_bits"] = float(-(p[p > 0] * p[p > 0].log2()).sum())
        st["bits_per_symbol"] = model.mapper.bits
        st["coded_bits_per_sentence"] = None  # filled by caller
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/europarl_en.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--channel", default=None, help="test channel (default: training channel)")
    ap.add_argument("--snrs", default="0,3,6,9,12,15,18")
    ap.add_argument("--erasure-p", type=float, default=None)
    ap.add_argument("--n-test", type=int, default=0)
    ap.add_argument("--rx-hard", action="store_true", help="force hard symbol decisions at the receiver")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)

    _, test, vocab = load_corpus(a.data)
    if a.n_test:
        test = test[: a.n_test]
    model, _, ck = load_checkpoint(a.ckpt, vocab)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if a.rx_hard:
        model.rx_hard = True
    ch = a.channel or ck["cfg"]["channel"]
    kw = dict(ck["cfg"].get("channel_kwargs", {})) if ch == ck["cfg"]["channel"] else {}
    if a.erasure_p is not None:
        kw["p"] = a.erasure_p
    snrs = [float(s) for s in a.snrs.split(",")]

    res = evaluate_bleu(model, test, vocab, ch, snrs, **kw)
    t = time.time()
    transmit_sentences(model, test[:256], vocab, ch, 12.0, batch_size=1, **kw)
    runtime_ms = (time.time() - t) / 256 * 1000
    n_sym = sum(len(s.split()) + 2 for s in test) / len(test) * model.n_sym
    digital = symbol_stats(model, test, vocab)
    if model.mapper is not None:
        digital["coded_bits_per_sentence"] = n_sym * model.mapper.bits
    out = {"ckpt": a.ckpt, "constellation": ck["cfg"].get("constellation"), "rx_hard": model.rx_hard,
           "digital": digital, "channel": ch, "channel_kwargs": kw, "snrs": snrs,
           "symbols_per_sentence": n_sym, "runtime_ms_per_sentence": runtime_ms,
           "bleu": {str(s): res[s]["bleu"] for s in snrs},
           "samples": {str(s): res[s]["samples"] for s in snrs}}
    for s in snrs:
        b = res[s]["bleu"]
        print(f"{ch} SNR {s:5.1f} dB | BLEU-1 {b[1]:.3f} 2 {b[2]:.3f} 3 {b[3]:.3f} 4 {b[4]:.3f}")
    print(f"avg complex symbols / sentence = {n_sym:.1f}, runtime {runtime_ms:.2f} ms / sentence")
    print("transmitted symbols:", json.dumps(digital))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
