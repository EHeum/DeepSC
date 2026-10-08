"""BLEU (1-4 gram) versus SNR of trained DeepSC models (Fig. 6) + sample sentences (Table II) + runtime (Table III)."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepsc.data import load_corpus  # noqa: E402
from deepsc.engine import evaluate_bleu, load_checkpoint, transmit_sentences  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/europarl_en.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--channel", default=None, help="test channel (default: training channel)")
    ap.add_argument("--snrs", default="0,3,6,9,12,15,18")
    ap.add_argument("--erasure-p", type=float, default=None)
    ap.add_argument("--n-test", type=int, default=0)
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
    out = {"ckpt": a.ckpt, "channel": ch, "channel_kwargs": kw, "snrs": snrs,
           "symbols_per_sentence": n_sym, "runtime_ms_per_sentence": runtime_ms,
           "bleu": {str(s): res[s]["bleu"] for s in snrs},
           "samples": {str(s): res[s]["samples"] for s in snrs}}
    for s in snrs:
        b = res[s]["bleu"]
        print(f"{ch} SNR {s:5.1f} dB | BLEU-1 {b[1]:.3f} 2 {b[2]:.3f} 3 {b[3]:.3f} 4 {b[4]:.3f}")
    print(f"avg complex symbols / sentence = {n_sym:.1f}, runtime {runtime_ms:.2f} ms / sentence")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
