#!/bin/bash
# LAMNr Glow 3D normative model of wild-type E15.5 embryos (P01, 60 um rigid-aligned, masked crops).
# Settings follow the authors' whole-head example (LAMNrFlows examples/lamnr_glow/3D/train_3d_dlbs_t1_whole_head.sh),
# single view, no alignment term. Usage: bash train_e15.sh <crop dir> <out dir> <iterations> [batch]
set -euo pipefail
CROPS=${1:?crop dir}; OUT=${2:?out dir}; ITERS=${3:-60000}; BATCH=${4:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
read H W D < <(python -c "import ants,glob; print(*ants.image_read(sorted(glob.glob('$HERE/$CROPS/sub-*.nii.gz'))[0]).shape)")
AUG_STOP=$(( ITERS * 5 / 6 ))
AUG="noise_std:cos:0.02->0.001@${AUG_STOP},sd_affine:cos:0.05->0.01@${AUG_STOP},sd_deformation:linear:12.0->2.0@${AUG_STOP},sd_simulated_bias_field:cos:0.20->0.0@${AUG_STOP},sd_histogram_warping:cos:0.04->0.0@${AUG_STOP}"
export OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
python -m antstorch.lamnr_flows.scripts.train_lamnr_glow_3d \
  --view "$HERE/$CROPS/sub-*_e15.nii.gz" --split-manifest "$HERE/$CROPS/split.csv" \
  --H $H --W $W --D $D --spatial-dims 3 \
  --L 4 --K 12 16 24 32 --hidden 48 64 96 128 \
  --batch $BATCH --max-iter $ITERS --precision mixed --amp-dtype bf16 \
  --ema --ema-decay 0.9997 --aug-schedules "$AUG" \
  --lr 2.5e-5 --warmup-iters 2000 --weight-decay 1e-5 \
  --grad-clip 0.1 --scale-cap 0.1 --glowbase-min-log -1.0 --glowbase-max-log 1.0 --scale-map tanh \
  --glowbase-logscale-factor 1.0 --actnorm-scale-cap 0.1 --legacy-conv-cap 2.5 \
  --plateau-factor 0.999999 --plateau-patience 100000 \
  --eval-interval 1000 --plot-interval 5000 --sample-temp 0.8 --num-workers 6 \
  --devices cuda:0 --out-dir "$HERE/$OUT" --auto-resume
