#!/usr/bin/env bash
# Split the OFDM gap into fading loss (AWGN -> i.i.d. Rayleigh) and correlation loss
# (i.i.d. Rayleigh -> OFDM). Same per-symbol gain statistics, only the correlation differs.
# Re-running skips finished steps and resumes interrupted training.
cd "$(dirname "$0")/.."
mkdir -p results/ofdm logs
SNRS=-6,-3,0,3,6,9,12,15,18,21,24
[ -f checkpoints/dsc_analog_riid.pt ] || python3 scripts/train.py --channel rayleigh_iid --epochs 12 --lr 5e-4 \
    --threads ${THREADS:-2} --init checkpoints/deepsc_awgn.pt --out checkpoints/dsc_analog_riid.pt \
    >> logs/train_dsc_analog_riid.log 2>&1 &
ev() { out=$1; shift; [ -f results/ofdm/$out.json ] || python3 scripts/evaluate.py --snrs=$SNRS --threads 1 "$@" \
    --out results/ofdm/$out.json > logs/eval_ofdm_$out.log 2>&1; }
# one fixed model (the 40-epoch AWGN model) on every channel: a pure channel effect
ev awgnmodel_on_awgn --ckpt checkpoints/deepsc_awgn.pt --channel awgn &
ev awgnmodel_on_riid --ckpt checkpoints/deepsc_awgn.pt --channel rayleigh_iid &
wait
ev awgnmodel_on_block --ckpt checkpoints/deepsc_awgn.pt --channel rayleigh &
ev ofdmmodel_on_riid --ckpt checkpoints/dsc_analog_ofdm.pt --channel rayleigh_iid &
[ -f results/ofdm/baselines_riid.json ] || python3 scripts/eval_baselines.py --channels rayleigh_iid --snrs=$SNRS \
    --out results/ofdm/baselines_riid.json > logs/baselines_riid.log 2>&1 &
wait
# models fine-tuned on each channel (same 12-epoch budget)
ev analog_riid --ckpt checkpoints/dsc_analog_riid.pt --channel rayleigh_iid &
ev riidmodel_on_ofdm --ckpt checkpoints/dsc_analog_riid.pt --channel ofdm &
wait
cp checkpoints/dsc_analog_riid_history.json results/ofdm/ 2>/dev/null
echo decomp done
