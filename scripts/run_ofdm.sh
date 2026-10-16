#!/usr/bin/env bash
# DeepSC over CP-OFDM (64 subcarriers, CP 16, 8-tap exponential multipath, ZF, perfect CSI).
# Re-running the script skips finished steps and resumes interrupted training.
cd "$(dirname "$0")/.."
mkdir -p results/ofdm logs
SNRS=-6,-3,0,3,6,9,12,15,18,21,24
FT="--channel ofdm --epochs 12 --lr 5e-4 --threads ${THREADS:-2}"
[ -f checkpoints/dsc_analog_ofdm.pt ] || python3 scripts/train.py $FT --init checkpoints/deepsc_awgn.pt \
    --out checkpoints/dsc_analog_ofdm.pt >> logs/train_dsc_analog_ofdm.log 2>&1 &
[ -f checkpoints/dsc_qam16_ofdm.pt ] || python3 scripts/train.py $FT --init checkpoints/dsc_qam16_ft.pt \
    --constellation qam16 --out checkpoints/dsc_qam16_ofdm.pt >> logs/train_dsc_qam16_ofdm.log 2>&1 &
wait
ev() { out=$1; shift; [ -f results/ofdm/$out.json ] || python3 scripts/evaluate.py --snrs=$SNRS --threads 1 --channel ofdm "$@" \
    --out results/ofdm/$out.json > logs/eval_ofdm_$out.log 2>&1; }
ev analog_ofdm --ckpt checkpoints/dsc_analog_ofdm.pt &
ev qam16_ofdm --ckpt checkpoints/dsc_qam16_ofdm.pt &
ev qam16_ofdm_hardrx --ckpt checkpoints/dsc_qam16_ofdm.pt --rx-hard &
ev analog_awgn_model --ckpt checkpoints/deepsc_awgn.pt &
wait
ev qam16_awgn_model --ckpt checkpoints/dsc_qam16_ft.pt &
[ -f results/ofdm/baselines_ofdm.json ] || python3 scripts/eval_baselines.py --channels ofdm --snrs=$SNRS \
    --out results/ofdm/baselines_ofdm.json > logs/baselines_ofdm.log 2>&1 &
wait
cp checkpoints/dsc_*_ofdm_history.json results/ofdm/ 2>/dev/null
echo ofdm done
