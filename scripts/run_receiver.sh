#!/usr/bin/env bash
# Receiver study for fading channels: ZF (done in run_ofdm.sh / run_fading_decomp.sh) vs MMSE vs
# ZF + CSI-aware decoder vs MMSE + CSI-aware decoder, on i.i.d. Rayleigh and OFDM (contiguous /
# interleaved). All fine-tuned for 12 epochs from the 40-epoch AWGN model.
# Re-running skips finished steps and resumes interrupted training.
cd "$(dirname "$0")/.."
mkdir -p results/receiver logs
SNRS=-6,-3,0,3,6,9,12,15,18,21,24
T=${THREADS:-2}
declare -A CH=([riid]="--channel rayleigh_iid" [ofdm]="--channel ofdm" [ofdmil]="--channel ofdm --ofdm-map interleaved")
declare -A RX=([mmse]="--eq mmse" [csi]="--eq zf --csi-rx" [mmsecsi]="--eq mmse --csi-rx")
tr() { [ -f checkpoints/rx_$1_$2.pt ] || python3 scripts/train.py ${CH[$2]} ${RX[$1]} --epochs 12 --lr 5e-4 --threads $T \
    --init checkpoints/deepsc_awgn.pt --out checkpoints/rx_$1_$2.pt >> logs/train_rx_$1_$2.log 2>&1; }
ev() { [ -f results/receiver/$1_$2.json ] || python3 scripts/evaluate.py --ckpt checkpoints/rx_$1_$2.pt --snrs=$SNRS \
    --threads 1 --out results/receiver/$1_$2.json > logs/eval_rx_$1_$2.log 2>&1; }
for rx in mmse csi mmsecsi; do
  tr $rx riid & tr $rx ofdm & wait
  tr $rx ofdmil & ev $rx riid & ev $rx ofdm & wait
  ev $rx ofdmil
done
cp checkpoints/rx_*_history.json results/receiver/ 2>/dev/null
echo receiver done
