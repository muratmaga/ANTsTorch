#!/bin/bash
#SBATCH --job-name=e18gpu-r10-prep
#SBATCH --partition=cpu-core
#SBATCH --account=cdbrm
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/hps/home/amaga/magalab/user/amaga/E18_amtc/logs/%x_%j.out
# Round 10 starts from the GPU round-9 template (after its shape update + sharpening).
set -euo pipefail
A=/data/hps/home/amaga/magalab/user/amaga/E18_amtc
mkdir -p $A/build18gpu_r10/intermediateTemplates
cp $A/build18gpu_r9/E18_template0.nii.gz $A/build18gpu_r10/E18_template0.nii.gz
md5sum $A/build18gpu_r9/E18_template0.nii.gz $A/build18gpu_r10/E18_template0.nii.gz
