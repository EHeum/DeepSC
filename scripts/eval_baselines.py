"""BLEU versus SNR of the traditional separate source/channel coding schemes (Fig. 6 baselines)."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepsc.baselines import FixedLengthCoder, HuffmanCoder, ReedSolomon, TraditionalSystem, TurboCode  # noqa
from deepsc.data import load_corpus  # noqa: E402
from deepsc.metrics import bleu_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/europarl_en.json")
    ap.add_argument("--channels", default="awgn,rayleigh")
    ap.add_argument("--ofdm-map", default="contiguous", choices=["contiguous", "interleaved"])
    ap.add_argument("--snrs", default="0,3,6,9,12,15,18")
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--out", default="results/paper/baselines.json")
    a = ap.parse_args()

    train, test, _ = load_corpus(a.data)
    test = test[: a.n_test]
    huff, fixed = HuffmanCoder(train), FixedLengthCoder()
    print(f"Huffman: {huff.avg_bits_per_char(test):.3f} bit/char, fixed: {fixed.bits} bit/char")
    # (name, source coder, channel code, QAM order) as in Fig. 6 (128-QAM is replaced by 64-QAM,
    # since only square constellations are implemented)
    schemes = [
        ("Huffman + RS(42,30)", huff, lambda: ReedSolomon(42, 30), 64),
        ("5-bit + RS(54,42)", fixed, lambda: ReedSolomon(54, 42), 64),
        ("Huffman + Turbo", huff, lambda: TurboCode(5), 64),
        ("5-bit + Turbo", fixed, lambda: TurboCode(5), 64),
    ]
    snrs = [float(s) for s in a.snrs.split(",")]
    res = {}
    for ch in a.channels.split(","):
        res[ch] = {}
        for name, src, code, m in schemes:
            sysm = TraditionalSystem(src, code(), m, seed=0, ofdm_mapping=a.ofdm_map)
            res[ch][name] = {"bleu": {}, "samples": {}}
            for snr in snrs:
                t, hyp, nsym = time.time(), [], []
                for i in range(0, len(test), a.batch):
                    h, n = sysm.transmit(test[i:i + a.batch], snr, ch)
                    hyp += h
                    nsym += n
                b = bleu_scores(test, hyp)
                res[ch][name]["bleu"][str(snr)] = b
                res[ch][name]["samples"][str(snr)] = list(zip(test[:5], hyp[:5]))
                res[ch][name]["symbols_per_sentence"] = float(np.mean(nsym))
                res[ch][name]["runtime_ms_per_sentence"] = (time.time() - t) / len(test) * 1000
                print(f"{ch:8s} {name:20s} SNR {snr:4.1f} | BLEU-1 {b[1]:.3f} 2 {b[2]:.3f} 3 {b[3]:.3f} "
                      f"4 {b[4]:.3f} | {np.mean(nsym):.1f} sym/sent | {time.time() - t:.1f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
