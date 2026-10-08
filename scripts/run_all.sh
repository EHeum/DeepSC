#!/usr/bin/env bash
# Full experiment pipeline used for the results in README.md (CPU, ~6 h on 4 cores).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p checkpoints results logs
T=${THREADS:-2}

python3 scripts/prepare_data.py
python3 scripts/eval_baselines.py --out results/baselines.json                       # Fig. 6 baselines

# --- basic model (Fig. 6, Table II/III) ---
python3 scripts/train.py --channel awgn     --epochs 40 --threads $T --out checkpoints/deepsc_awgn.pt
python3 scripts/train.py --channel rayleigh --epochs 40 --threads $T --out checkpoints/deepsc_rayleigh.pt
# --- ablations: MI loss (Fig. 9/10) and symbols per word (Fig. 8) ---
python3 scripts/train.py --channel awgn --epochs 40 --threads $T --mi-lambda 0 --out checkpoints/deepsc_awgn_nomi.pt
python3 scripts/train.py --channel awgn --epochs 40 --threads $T --symbols-per-word 4 --out checkpoints/deepsc_awgn_sym4.pt
# --- transfer learning for a new channel (Alg. 4, Fig. 5(a), Fig. 13) ---
python3 scripts/train.py --channel rayleigh --epochs 12 --threads $T --init checkpoints/deepsc_awgn.pt \
    --freeze semantic --unfreeze-epoch 9 --out checkpoints/deepsc_rayleigh_tl.pt
python3 scripts/train.py --channel erasure --erasure-p 0.1 --epochs 12 --threads $T --out checkpoints/deepsc_erasure.pt
python3 scripts/train.py --channel erasure --erasure-p 0.1 --epochs 12 --threads $T --init checkpoints/deepsc_awgn.pt \
    --freeze semantic --unfreeze-epoch 9 --out checkpoints/deepsc_erasure_tl.pt

# --- evaluation ---
for m in awgn rayleigh awgn_nomi awgn_sym4 rayleigh_tl; do
  python3 scripts/evaluate.py --ckpt checkpoints/deepsc_$m.pt --threads $T --out results/deepsc_$m.json
done
for m in erasure erasure_tl; do
  for p in 0.05 0.1 0.15 0.2; do
    python3 scripts/evaluate.py --ckpt checkpoints/deepsc_$m.pt --snrs 12 --erasure-p $p --threads $T \
        --out results/deepsc_${m}_p$p.json
  done
done
python3 scripts/estimate_mi.py --threads $T \
    --ckpts "with MI term=checkpoints/deepsc_awgn.pt,without MI term=checkpoints/deepsc_awgn_nomi.pt"
cp checkpoints/*_history.json results/
python3 scripts/plot_results.py
