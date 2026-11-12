# LAMNr normative model of E15.5 embryos

A Glow 3D flow (`antstorch.lamnr_flows`) is trained on wild-type embryos only. Each embryo is then
scored by its distance from the wild-type latent distribution at each latent level (low-rank plus
diagonal Mahalanobis, leave-one-out for training embryos), standardised per level and averaged
(`z_combined`). Bits per dimension is computed too but does not detect abnormal embryos.

## Run order

Run from this directory with the ANTsTorch virtual environment active.

1. `prep_crops.py`: rigid-aligned 60 µm volumes, Otsu mask, 64×128×80 crops at 144 µm, the split
   (82 training WT, 8 validation WT, the rest testing) and the training manifest. Used by Model 1.
2. `prep_external.py`: the same crops for the external IMPC embryos, with M3T calls.
3. `prep_similarity.py --grids 64x128x80 96x192x128`: whole-body masks and a similarity
   registration (rigid plus isotropic scale) to the masked template, then crops at 114 µm and 72 µm
   for the lab and external embryos. Used by Model 2.
4. `fix_external_masks.py`: tightens external masks that include the scanner's field-of-view block
   (foreground fraction above 0.55) and marks registration failures (scale below 0.6) as `qc=fail`.
5. `bash train_e15.sh <crop dir> <run dir> <iterations> [batch]`: trains with the authors' 3D
   recipe; checkpoints every 1000 iterations.
6. `score_e15.py --run <run dir> --crops <crop dir> [--ckpt training_state_itNNNNNN.pt] --extra <external crop dir>`:
   writes `scores_<checkpoint>.csv` and `scores_<checkpoint>_<external dir>.csv`.
7. `analyze_e15.py <scores csv>`: lab AUCs (exencephaly, Gli2 MUT and HET), by strain, a supervised
   PCA plus logistic regression on the latents, and the Gli2 WT→MUT latent axis.
8. `analyze_external.py <scores csv>`: external AUCs against M3T, the difference between centres,
   correlation with M3T probability, and results by gene.
9. `cf_maps.py --run ... --crops ... --subjects ... [--keep-levels 3 4]`: "nearest normal"
   counterfactual maps, made by projecting the latents onto the wild-type principal subspace.

## Results (2026-11-12)

Model 2 (similarity crops, 64×128×80), unsupervised AUC:

- **External, M3T-called exencephalic vs negatives:** 0.988 at 2k iterations, 0.976 at 15k.
  Excludes Cnnm2 and the failed registration.
- **External, correlation with M3T probability:** Spearman ρ 0.69.
- **Lab, exencephaly vs held-out WT:** 0.94.
- **Lab, Gli2 MUT vs WT:** 0.81–0.87 unsupervised, 1.00 supervised.
- **Lab, Gli2 HET:** not distinguishable from WT.

Model 1 (rigid crops only) fails on the external set because of differences between centres.

Cnnm2 crops (BL4919-*) are not usable as processed: some keep the field-of-view block, one is
misoriented, and others keep streaks from the tube wall.
