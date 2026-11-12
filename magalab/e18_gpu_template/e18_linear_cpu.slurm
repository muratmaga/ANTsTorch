#!/bin/bash
#SBATCH --job-name=e18-linear
#SBATCH --partition=cpu-cdbrm
#SBATCH --account=cdbrm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/data/hps/home/amaga/magalab/user/amaga/E18_amtc/logs/%x_%A_%a.out
# Linear stages of a GPU template round, on CPU: aMTC2's own antsRegistration rigid + affine call
# (MI 32 Regular 0.25, 1000x500x250x0, 6x4x2x1 / 4x2x1x0, -u 1 -w [0.01,0.99] -z 1, -r [T,S,1]),
# writing <OUT>/E18_input<jjjj>-<id>-0GenericAffine.mat. The GPU array then runs with --skip-linear,
# so GPU nodes only do the SyN stage. Usage:
#   sbatch --array=0-49 --export=ALL,OUT=<round dir>,TEMPLATE=<template>,INPUTDIR=<inputs> e18_linear_cpu.slurm
set -euo pipefail
A=/data/hps/home/amaga/magalab/user/amaga/E18_amtc
G=/data/hps/home/amaga/magalab/user/amaga/antstorch_gpu
: "${OUT:?}" "${TEMPLATE:?}" "${INPUTDIR:?}"
INPUTS=( $INPUTDIR/*.nii.gz )
S=${INPUTS[$SLURM_ARRAY_TASK_ID]}
mkdir -p $OUT
export PATH=$A/antsbin:$PATH
module load python312 >/dev/null 2>&1
. $G/venv/bin/activate
export PYTHONPATH=${CODE:-$G/ANTsTorch_dist}
NUMA=""; command -v numactl >/dev/null && NUMA="numactl --interleave=all"
$NUMA python $G/register_one_gpu_dist.py --linear-only \
  --template $TEMPLATE --subject "$S" --index $SLURM_ARRAY_TASK_ID \
  --outdir $OUT --outputname E18_ --ants-bin $A/antsbin
