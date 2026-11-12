#!/bin/bash
#SBATCH --job-name=e18gpu-12um-4gpu
#SBATCH --partition=gpu-core
#SBATCH --account=cdbrm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=08:00:00
#SBATCH --output=/data/hps/home/amaga/magalab/user/amaga/E18_amtc/logs/%x_%j.out
# One E18 subject at 12 um (template grid 852x1182x1965 = 1.98 G voxels, ~235 GB per registration):
# coarse SyN levels on one A100, finest level split across the node's 4 A100s
# (antstorch/syn/core/distributed.py). Template = round-11 12 um template (build12e); settings as the
# 12 um rounds (SyN[0.25,2,0], CC r4, 12x6x4x2x1 / 6x3x2x1x0 / 2000x600x400x300x200) with the
# antsRegistration convergence test (1e-9, window 10). Subject 0 of inputs12 (131123).
set -euo pipefail
A=/data/hps/home/amaga/magalab/user/amaga/E18_amtc
G=/data/hps/home/amaga/magalab/user/amaga/antstorch_gpu
OUT=$A/build12gpu_test
mkdir -p $OUT
INPUTS=( $A/inputs12/*.nii.gz )
S=${INPUTS[0]}
echo "subject $S on $(hostname) $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
export PATH=$A/antsbin:$PATH
module load python312 >/dev/null 2>&1
. $G/venv/bin/activate
export PYTHONPATH=$G/ANTsTorch_dist
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUMA=""; command -v numactl >/dev/null && NUMA="numactl --interleave=all"
/usr/bin/time -v $NUMA python $G/register_one_gpu_dist.py \
  --template $A/build12e/E18_template0.nii.gz --subject "$S" --index 0 \
  --outdir $OUT --outputname E18_ --ants-bin $A/antsbin \
  --iterations 2000x600x400x300x200 --shrink 12x6x4x2x1 --sigmas 6x3x2x1x0 \
  --grad-step 0.25 --update-variance 2 --cc-radius 4 --metric-chunks 8 \
  --convergence-threshold 1e-9 --convergence-window 10 \
  --final-level-devices cuda:0,cuda:1,cuda:2,cuda:3
rc=$?
echo "exit $rc $(date)"
exit $rc
