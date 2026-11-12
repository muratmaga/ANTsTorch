#!/bin/bash
#SBATCH --job-name=e18gpu-update
#SBATCH --partition=cpu-cdbrm
#SBATCH --account=cdbrm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/data/hps/home/amaga/magalab/user/amaga/E18_amtc/logs/%x_%j.out
# Template update for the GPU round 9, on CPU: the script's own summarizeimageset() and
# shapeupdatetotemplate(), sourced verbatim from the round-9 script copy and called with round 9's
# arguments: -a 0 (plain mean), Laplacian sharpening (-A 1 default), gradient step 0.25,
# AverageAffineTransformNoRigid (-y 0 default), float. Then the script's intermediate-template copies.
set -o pipefail   # no -u: the sourced script functions rely on unset-variable defaults
A=/data/hps/home/amaga/magalab/user/amaga/E18_amtc
OUT=${OUT:?set OUT to the round directory}
T=$A/antsbin/antsMultivariateTemplateConstruction2_64c.sh
export PATH=$A/antsbin:$PATH
cd $OUT
n_mat=$(ls E18_input*-0GenericAffine.mat | wc -l)
n_warp=$(ls E18_input*-1Warp.nii.gz | wc -l)
n_def=$(ls E18_input*-modality0-*-WarpedToTemplate.nii.gz | wc -l)
echo "affines $n_mat  warps $n_warp  warped images $n_def  ($(date), $(hostname))"
[[ $n_mat -eq 50 && $n_warp -eq 50 && $n_def -eq 50 ]] || { echo "expected 50 of each; not updating"; exit 1; }
IFS=$'\n\t'                                   # same as the script
source <(sed -n '/^function summarizeimageset() {/,/^function jobfnamepadding {/p' "$T" | sed '$d')
declare -F shapeupdatetotemplate summarizeimageset >/dev/null || { echo "functions not found in $T"; exit 1; }
IMAGESPERMODALITY=50
AVERAGE_AFFINE_PROGRAM=AverageAffineTransformNoRigid
USEFLOAT=1
WARP=antsApplyTransforms
cp E18_template0.nii.gz intermediateTemplates/initial_E18_template0.nii.gz
shapeupdatetotemplate 3 $OUT/E18_template0.nii.gz $OUT/E18_template $OUT/E18_ 0.25 0 0 1
rc=$?
cp E18_template0.nii.gz intermediateTemplates/SyN_iteration0_E18_template0.nii.gz
cp E18_template0warp.nii.gz intermediateTemplates/SyN_iteration0_shapeUpdateWarp.nii.gz
echo "shape update exit $rc $(date)"
exit $rc
