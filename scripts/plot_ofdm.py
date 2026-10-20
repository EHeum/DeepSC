"""Figures for the OFDM experiments (docs/03_ofdm.md)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import MARKERS, SERIES, plt  # noqa: E402  (shared style)

D = "results/ofdm"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def curve(bleu, n):
    snrs = sorted(float(s) for s in bleu)
    return snrs, [bleu[f"{s:.1f}"][str(n)] for s in snrs]


def main():
    base = (load(os.path.join(D, "baselines_ofdm.json")) or {}).get("ofdm", {})
    lines = [
        ("Analog DeepSC, OFDM fine-tuned", load(os.path.join(D, "analog_ofdm.json")), SERIES[0], "-"),
        ("16-QAM DeepSC, OFDM fine-tuned", load(os.path.join(D, "qam16_ofdm.json")), SERIES[2], "-"),
        ("16-QAM DeepSC, OFDM fine-tuned, hard Rx", load(os.path.join(D, "qam16_ofdm_hardrx.json")), SERIES[2], "--"),
        ("Analog DeepSC, AWGN model (no retraining)", load(os.path.join(D, "analog_awgn_model.json")), SERIES[1], "-"),
        ("Analog DeepSC on AWGN (reference)", load("results/digital/analog_ft.json"), "#898781", ":"),
    ]
    for name, color in (("Huffman + Turbo", SERIES[4]), ("Huffman + RS(42,30)", SERIES[5])):
        if name in base:
            lines.append((name + " (64-QAM)", base[name], color, "--"))
    for n in (1, 4):
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        for i, (name, d, color, ls) in enumerate(lines):
            if not d:
                continue
            x, y = curve(d["bleu"], n)
            ax.plot(x, y, color=color, ls=ls, marker=MARKERS[i % len(MARKERS)], label=name)
        ax.set_xlabel("average subcarrier SNR (dB)")
        ax.set_ylabel(f"BLEU ({n}-gram)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(range(-6, 25, 3))
        ax.set_title(f"DeepSC over CP-OFDM (64 SC, 8-tap multipath, ZF), BLEU-{n}")
        h, lab = ax.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(os.path.join(D, f"bleu{n}_ofdm.png"), dpi=130)
        plt.close(fig)
    decomposition()
    print("figures written to", D)


def decomposition():
    """Fading vs correlation loss: one fixed model on every channel, and models fine-tuned per channel."""
    panels = [
        ("40-epoch AWGN model, no retraining", [
            ("AWGN", "awgnmodel_on_awgn", "#898781", ":"),
            ("i.i.d. Rayleigh", "awgnmodel_on_riid", SERIES[3], "-"),
            ("OFDM interleaved", "analog_awgn_model_il", SERIES[1], "-"),
            ("OFDM contiguous", "analog_awgn_model", SERIES[0], "-"),
            ("block Rayleigh", "awgnmodel_on_block", SERIES[6], "-"),
        ]),
        ("fine-tuned 12 epochs on each channel", [
            ("AWGN (fine-tuned 15 epochs)", "../digital/analog_ft", "#898781", ":"),
            ("i.i.d. Rayleigh", "analog_riid", SERIES[3], "-"),
            ("OFDM interleaved", "analog_ofdm_il", SERIES[1], "-"),
            ("OFDM contiguous", "analog_ofdm", SERIES[0], "-"),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (title, lines) in zip(axes, panels):
        for i, (name, f, color, ls) in enumerate(lines):
            d = load(os.path.join(D, f + ".json"))
            if not d:
                continue
            x, y = curve(d["bleu"], 1)
            ax.plot(x, y, color=color, ls=ls, marker=MARKERS[i], label=name)
        ax.set_title(title)
        ax.set_xlabel("average SNR per symbol (dB)")
        ax.set_xticks(range(-6, 25, 3))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("BLEU (1-gram)")
    fig.suptitle("Analog DeepSC: fading loss (AWGN -> i.i.d. Rayleigh) and the effect of correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fading_vs_correlation.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
