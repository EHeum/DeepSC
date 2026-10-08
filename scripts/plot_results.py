"""Draw the reproduced figures from the JSON results in results/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = "results"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def series_bleu(d, n):
    snrs = sorted(float(s) for s in d)
    return snrs, [d[str(s)][str(n)] if str(n) in d[str(s)] else d[str(s)][n] for s in snrs]


def fig_bleu_vs_snr():
    base = load("baselines.json") or {}
    for ch in ("awgn", "rayleigh"):
        curves = []
        ds = load(f"deepsc_{ch}.json")
        if ds:
            curves.append(("DeepSC", ds["bleu"]))
        for name, v in base.get(ch, {}).items():
            curves.append((name, v["bleu"]))
        if not curves:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
        for n, ax in enumerate(axes, 1):
            for i, (name, b) in enumerate(curves):
                x, y = series_bleu(b, n)
                ax.plot(x, y, color=SERIES[i], marker=MARKERS[i], label=name)
            ax.set_xlabel("SNR (dB)")
            ax.set_title(f"BLEU ({n}-gram)")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([0, 3, 6, 9, 12, 15, 18])
        axes[0].set_ylabel("BLEU score")
        axes[0].legend(loc="center right", fontsize=8)
        fig.suptitle(f"BLEU vs SNR - {'AWGN' if ch == 'awgn' else 'Rayleigh fading'} channel")
        fig.tight_layout()
        fig.savefig(os.path.join(R, f"fig6_bleu_{ch}.png"), dpi=130)
        plt.close(fig)


def fig_mi():
    d = load("mutual_information.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    names = [k for k in d if k not in ("snrs", "gaussian_capacity")]
    for i, k in enumerate(names):
        ax.plot(d["snrs"], d[k], color=SERIES[i], marker=MARKERS[i], label=k)
    ax.plot(d["snrs"], d["gaussian_capacity"], color="#898781", ls="--", lw=1.5, label="log2(1+SNR)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("I(X;Y) (bit / complex symbol)")
    ax.set_title("Mutual information vs SNR (AWGN)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig9_mutual_information.png"), dpi=130)
    plt.close(fig)


def fig_training_curves(fname, runs, title, key="train_ce", ylabel="Training CE loss"):
    runs = [(n, load(p)) for n, p in runs]
    runs = [(n, d) for n, d in runs if d]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, (n, d) in enumerate(runs):
        h = d["history"]
        ax.plot([r["epoch"] for r in h], [r[key] for r in h], color=SERIES[i], label=n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(R, fname), dpi=130)
    plt.close(fig)


def fig_simple_bleu(fname, runs, title):
    runs = [(n, load(p)) for n, p in runs]
    runs = [(n, d) for n, d in runs if d]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, (n, d) in enumerate(runs):
        x, y = series_bleu(d["bleu"], 1)
        ax.plot(x, y, color=SERIES[i], marker=MARKERS[i], label=n)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BLEU (1-gram)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(R, fname), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_bleu_vs_snr()
    fig_mi()
    fig_training_curves("fig10_mi_during_training.png",
                        [("with MI term (lambda=0.01)", "deepsc_awgn_history.json"),
                         ("without MI term", "deepsc_awgn_nomi_history.json")],
                        "MI estimate during training (AWGN)", key="mi_bits", ylabel="I(X;Y) (bit / symbol)")
    fig_training_curves("fig13_tl_rayleigh_loss.png",
                        [("with TL (from AWGN model)", "deepsc_rayleigh_tl_history.json"),
                         ("without TL (from scratch)", "deepsc_rayleigh_history.json")],
                        "Transfer learning: re-training for Rayleigh")
    fig_training_curves("fig13_tl_erasure_loss.png",
                        [("with TL (from AWGN model)", "deepsc_erasure_tl_history.json"),
                         ("without TL (from scratch)", "deepsc_erasure_history.json")],
                        "Transfer learning: re-training for erasure channel")
    fig_simple_bleu("fig8_symbols_per_word.png",
                    [("4 symbols / word", "deepsc_awgn_sym4.json"),
                     ("8 symbols / word", "deepsc_awgn.json")],
                    "DeepSC (AWGN): symbols per word")
    fig_simple_bleu("fig_mi_ablation_bleu.png",
                    [("with MI term", "deepsc_awgn.json"), ("without MI term", "deepsc_awgn_nomi.json")],
                    "DeepSC (AWGN): effect of the MI loss")
    fig_simple_bleu("fig13_tl_rayleigh_bleu.png",
                    [("with TL", "deepsc_rayleigh_tl.json"), ("without TL", "deepsc_rayleigh.json")],
                    "Transfer learning (Rayleigh)")
    print("figures written to", R)
