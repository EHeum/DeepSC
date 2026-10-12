"""Figures and cliff metrics for the constellation-constrained (digital) DeepSC (README section 5)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import MARKERS, SERIES, plt  # noqa: E402  (shared style)

D = "results/digital"


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def curve(bleu, n=1):
    snrs = sorted(float(s) for s in bleu)
    return np.array(snrs), np.array([bleu[f"{s:.1f}" if f"{s:.1f}" in bleu else str(s)][str(n)] for s in snrs])


def transition(snr, y):
    """SNR at which the curve reaches 10 % / 90 % of its own maximum and the steepest 3 dB step."""
    top = y.max()
    def cross(level):
        above = np.nonzero(y >= level)[0]
        if len(above) == 0:
            return None
        i = above[0]
        if i == 0:
            return None  # already above the level at the lowest tested SNR
        return float(np.interp(level, [y[i - 1], y[i]], [snr[i - 1], snr[i]]))
    s10, s90 = cross(0.1 * top), cross(0.9 * top)
    steps = np.diff(y) / np.diff(snr) * 3.0
    k = int(np.argmax(steps))
    lo = s10 if s10 is not None else float(snr[0])
    return {"max": float(top), "snr_10": s10, "snr_90": s90, "width_10_90_db": s90 - lo,
            "width_is_lower_bound": s10 is None,
            "max_gain_per_3db": float(steps[k]), "at_snr": f"{snr[k]:g}->{snr[k + 1]:g}"}


def models():
    base = load("baselines_awgn.json") or {}
    out = [("Analog DeepSC", load("analog_ft.json"))]
    out += [(f"DeepSC {m.upper()[:3]}-{m[3:]}", load(f"{m}_ft.json")) for m in ("qam4", "qam16", "qam64")]
    out += [(f"DeepSC {m.upper()[:3]}-{m[3:]}, hard Rx", load(f"{m}_ft_hardrx.json")) for m in ("qam4", "qam16", "qam64")]
    out += [("Analog, trained @12 dB", load("analog_snr12.json")),
            ("QAM-16, trained @12 dB", load("qam16_snr12.json")),
            ("QAM-16 hard Rx, trained @12 dB", load("qam16_snr12_hardrx.json"))]
    for name in ("Huffman + Turbo", "5-bit + RS(54,42)"):
        if name in base.get("awgn", {}):
            out.append((name, {"bleu": base["awgn"][name]["bleu"],
                               "symbols_per_sentence": base["awgn"][name]["symbols_per_sentence"]}))
    return [(n, d) for n, d in out if d]


def plot(fname, names, title, n=1, styles=None, vline=None, labels=None):
    ms = dict(models())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, name in enumerate(names):
        if name not in ms:
            continue
        x, y = curve(ms[name]["bleu"], n)
        color, ls = (styles or {}).get(name, (SERIES[i], "-"))
        ax.plot(x, y, color=color, ls=ls, marker=MARKERS[i % len(MARKERS)], label=(labels or {}).get(name, name))
    if vline is not None:
        ax.axvline(vline, color="#898781", lw=1, ls=":")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(f"BLEU ({n}-gram)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(-6, 25, 3))
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(D, fname), dpi=130)
    plt.close(fig)


def plot_mi():
    d = load("mutual_information.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = [k for k in d if k not in ("snrs", "gaussian_capacity")]
    for i, k in enumerate(names):
        ax.plot(d["snrs"], d[k], color=SERIES[i], marker=MARKERS[i], label=k)
        if "QAM" in k:
            ax.axhline(np.log2(int(k.split("-")[0])), color=SERIES[i], lw=1, ls=":")
    ax.plot(d["snrs"], d["gaussian_capacity"], color="#898781", ls="--", lw=1.5, label="log2(1+SNR)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("I(X;Y) (bit / complex symbol)")
    ax.set_title("Mutual information of the learned channel input (dotted: log2 M)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "mi_digital.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    q = {"DeepSC QAM-4": SERIES[1], "DeepSC QAM-16": SERIES[2], "DeepSC QAM-64": SERIES[3]}
    styles = {"Analog DeepSC": (SERIES[0], "-"), "Huffman + Turbo": ("#52514e", "--"),
              "5-bit + RS(54,42)": ("#b5b4ad", "--")}
    styles.update({k: (c, "-") for k, c in q.items()})
    styles.update({k + ", hard Rx": (c, "--") for k, c in q.items()})
    main = ["Analog DeepSC", "DeepSC QAM-4", "DeepSC QAM-16", "DeepSC QAM-64", "Huffman + Turbo", "5-bit + RS(54,42)"]
    for n in (1, 4):
        plot(f"bleu{n}_digital_vs_analog.png", main, f"Analog vs constellation-constrained DeepSC (AWGN), BLEU-{n}",
             n, styles)
    plot("bleu1_soft_vs_hard_rx.png", ["DeepSC QAM-4", "DeepSC QAM-16", "DeepSC QAM-64", "DeepSC QAM-4, hard Rx",
                                       "DeepSC QAM-16, hard Rx", "DeepSC QAM-64, hard Rx"],
         "Soft (y) vs hard-decision receiver", 1, styles)
    plot("bleu1_snr_mismatch.png", ["Analog DeepSC", "Analog, trained @12 dB", "QAM-16, trained @12 dB",
                                    "QAM-16 hard Rx, trained @12 dB", "Huffman + Turbo"],
         "SNR mismatch: trained at 12 dB only", 1,
         {"Analog DeepSC": (SERIES[0], ":"), "Analog, trained @12 dB": (SERIES[0], "-"),
          "QAM-16, trained @12 dB": (SERIES[2], "-"), "QAM-16 hard Rx, trained @12 dB": (SERIES[2], "--"),
          "Huffman + Turbo": ("#52514e", "--")}, vline=12,
         labels={"Analog DeepSC": "Analog DeepSC (trained 0-18 dB, fine-tuned)"})
    plot_mi()

    rows = {}
    for name, d in models():
        x, y = curve(d["bleu"], 1)
        rows[name] = transition(x, y)
        x4, y4 = curve(d["bleu"], 4)
        t4 = transition(x4, y4)
        rows[name]["bleu4_max_gain_per_3db"], rows[name]["bleu4_at_snr"] = t4["max_gain_per_3db"], t4["at_snr"]
        rows[name]["symbols_per_sentence"] = d.get("symbols_per_sentence")
        rows[name]["coded_bits_per_sentence"] = (d.get("digital") or {}).get("coded_bits_per_sentence")
    with open(os.path.join(D, "cliff_metrics.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("| model | max BLEU-1 | SNR @10% of max | SNR @90% of max | 10-90% width (dB) "
          "| steepest BLEU-1 rise / 3 dB | steepest BLEU-4 rise / 3 dB |")
    print("|---|---|---|---|---|---|---|")
    for k, r in rows.items():
        s10 = "< -6" if r["snr_10"] is None else f"{r['snr_10']:.1f}"
        w = ("> " if r["width_is_lower_bound"] else "") + f"{r['width_10_90_db']:.1f}"
        print(f"| {k} | {r['max']:.2f} | {s10} | {r['snr_90']:.1f} | {w} "
              f"| {r['max_gain_per_3db']:.2f} ({r['at_snr']} dB) "
              f"| {r['bleu4_max_gain_per_3db']:.2f} ({r['bleu4_at_snr']} dB) |")
