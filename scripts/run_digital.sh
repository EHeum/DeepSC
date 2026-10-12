#!/usr/bin/env bash
# Constellation-constrained (DeepJSCC-Q style) DeepSC experiments; results in README section 5.
set -euo pipefail
cd "$(dirname "$0")/.."
T=${THREADS:-1}
FT="--init checkpoints/deepsc_awgn.pt --epochs 15 --lr 5e-4 --threads $T"
# (a) fine-tune the analog AWGN model with / without a constellation (same extra budget)
python3 scripts/train.py $FT --out checkpoints/dsc_analog_ft.pt > logs/train_dsc_analog_ft.log 2>&1 &
for m in qam4 qam16 qam64; do
  python3 scripts/train.py $FT --constellation $m --out checkpoints/dsc_${m}_ft.pt > logs/train_dsc_${m}_ft.log 2>&1 &
done
# (b) trained from scratch at a single SNR (12 dB) -> behaviour under SNR mismatch (cliff test)
for c in "analog" "qam16 --constellation qam16"; do
  set -- $c; n=$1; shift
  python3 scripts/train.py --epochs 25 --snr-min 12 --snr-max 12 --eval-snr 12 --threads $T "$@" \
      --out checkpoints/dsc_${n}_snr12.pt > logs/train_dsc_${n}_snr12.log 2>&1 &
done

SNRS=-6,-3,0,3,6,9,12,15,18,21,24
ev() { python3 scripts/evaluate.py --snrs=$SNRS --threads $T "$@"; }
wait_for() { until [ -f "$1" ]; do sleep 20; done; }
# evaluate the fine-tuned models as soon as they exist (in parallel with the 12 dB runs)
for m in analog qam4 qam16 qam64; do wait_for checkpoints/dsc_${m}_ft.pt; done
ev --ckpt checkpoints/dsc_analog_ft.pt --out results/digital/analog_ft.json > logs/eval_dsc_analog_ft.log 2>&1 &
for m in qam4 qam16 qam64; do
  ev --ckpt checkpoints/dsc_${m}_ft.pt --out results/digital/${m}_ft.json > logs/eval_dsc_${m}_ft.log 2>&1 &
done
wait
for m in qam4 qam16 qam64; do
  ev --ckpt checkpoints/dsc_${m}_ft.pt --rx-hard --out results/digital/${m}_ft_hardrx.json > logs/eval_dsc_${m}_ft_hardrx.log 2>&1 &
done
wait_for checkpoints/dsc_analog_snr12.pt
ev --ckpt checkpoints/dsc_analog_snr12.pt --out results/digital/analog_snr12.json > logs/eval_dsc_analog_snr12.log 2>&1 &
wait_for checkpoints/dsc_qam16_snr12.pt
ev --ckpt checkpoints/dsc_qam16_snr12.pt --out results/digital/qam16_snr12.json > logs/eval_dsc_qam16_snr12.log 2>&1 &
ev --ckpt checkpoints/dsc_qam16_snr12.pt --rx-hard --out results/digital/qam16_snr12_hardrx.json > logs/eval_dsc_qam16_snr12_hardrx.log 2>&1 &
python3 scripts/eval_baselines.py --channels awgn --snrs=$SNRS --out results/digital/baselines_awgn.json > logs/baselines_wide.log 2>&1 &
wait
python3 scripts/estimate_mi.py --threads 4 --snrs 0,6,12,18,24 --steps 1000 --out results/digital/mutual_information.json \
    --ckpts "analog=checkpoints/dsc_analog_ft.pt,4-QAM=checkpoints/dsc_qam4_ft.pt,16-QAM=checkpoints/dsc_qam16_ft.pt,64-QAM=checkpoints/dsc_qam64_ft.pt" \
    > logs/mi_digital.log 2>&1
cp checkpoints/dsc_*_history.json results/digital/
echo digital done
