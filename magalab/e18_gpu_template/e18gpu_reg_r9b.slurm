#!/bin/bash
#SBATCH --job-name=e18gpu-r9b
#SBATCH --partition=gpu-core
#SBATCH --account=cdbrm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/data/hps/home/amaga/magalab/user/amaga/E18_amtc/logs/%x_%A_%a.out
# E18 round 9 replicated again with GPU SyN, from the round-8 template (build18h), CPU round-9 caps.
# Uses ANTsTorch's ants_parity SyN (ITK update: level-voxel step, no midpoint averaging, per-iteration
# double inversion of both half-warps) and antsRegistration's convergence test (1e-9, window 10).
# The first GPU round 9 (build18gpu_r9, default ANTsTorch SyN, no convergence test) folded and is discarded.
# Same inputs and starting template as the CPU round 9 (build18i: -z build18h/E18_template0.nii.gz).
# Linear stages and the warped image use the same ANTs 2.5.4 container wrappers (antsbin/) as round 9;
# the SyN stage (SyN[0.25,2,0], CC r4, 12x6x4x2x1 / 6x3x2x1x0 / 2000x600x400x300x200) runs on the GPU
# with ANTsTorch (patched, mem-opt branch). Outputs use the script's own file names in $OUT.
# Array index = position in the script's input glob ($A/inputs18/*.nii.gz), as in aMTC2.
set -euo pipefail
A=/data/hps/home/amaga/magalab/user/amaga/E18_amtc
G=/data/hps/home/amaga/magalab/user/amaga/antstorch_gpu
OUT=$A/build18gpu_r9b
INPUTS=( $A/inputs18/*.nii.gz )
S=${INPUTS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID subject $S on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
export PATH=$A/antsbin:$PATH
module load python312 >/dev/null 2>&1
. $G/venv/bin/activate
export PYTHONPATH=$G/ANTsTorch_r9b   # code copy for this round
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUMA=""; command -v numactl >/dev/null && NUMA="numactl --interleave=all"
$NUMA python $G/register_one_gpu_r9b.py \
  --template $OUT/E18_template0.nii.gz --subject "$S" --index $SLURM_ARRAY_TASK_ID \
  --outdir $OUT --outputname E18_ --ants-bin $A/antsbin \
  --iterations 2000x600x400x300x200 --shrink 12x6x4x2x1 --sigmas 6x3x2x1x0 \
  --grad-step 0.25 --update-variance 2 --cc-radius 4 --metric-chunks 8 \
  --convergence-threshold 1e-9 --convergence-window 10 --ants-parity
rc=$?
echo "exit $rc $(date)"
exit $rc
