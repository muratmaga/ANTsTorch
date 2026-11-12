# GPU template building with ANTsTorch: notes (Nov 2026)

What we changed in ANTsTorch to run our template rounds on GPUs, how a GPU round is run on
Sasquatch, what it costs, and what to watch for. The upstream ANTsTorch (d4f15ad) could not do
this on its own: it ran out of memory at 18 µm and its SyN is not the ANTs/ITK SyN. All changes
are on the `mem-opt` branch of `~/Desktop/ANTsTorch_E15/ANTsTorch` (export:
`git diff d4f15ad > antstorch_gpu.patch`). Upstream issue: ANTsX/ANTsTorch#91.

## 1. Where things stand

| Case | Hardware | Peak GPU memory | Time per subject | CPU reference |
|---|---|---|---|---|
| E18 18 µm (586 M template voxels), one GPU | 1 x A100 80 GB | 71-72 GiB | 55-79 min (round 9b) | median 7 h 51 min (round 9, 64 cores) |
| E18 12 µm (1.98 G voxels), finest level on 4 GPUs | 1 node, 4 x A100 | 55-58 GiB per GPU (old SyN); parity estimated 67-69 | 2 h 04 min (old SyN, 200 finest iterations) | median 23.2 h (round 11, whole 128-core node); 131123: 18.8 h |
| E15 9 µm (1.94 G voxels) | not tried | ~ same as E18 12 µm | | |

Per finest-level iteration: 12 µm 10.8 s on 4 A100 against ~812 s on 128 CPU cores; 18 µm ~16 s on one
A100 (32 CC slabs, measured; ~11 s expected with 8 slabs, from the Blackwell slab test) against 357 s on 64 cores.

Status of the template rounds (E18, 18 µm, all from the round-8 template `build18h`):
* GPU round 9 (`build18gpu_r9`): **invalid, discard.** Default ANTsTorch SyN, no convergence test.
* GPU round 9b (`build18gpu_r9b`): ANTs-parity SyN + convergence test, CPU round-9 caps. Running/QC pending.
* 12 µm parity test with 4 GPUs (`build12gpu_test2`): pending.

## 2. The two things that make or break a GPU round

1. **Use `ants_parity=True` (script flag `--ants-parity`).** ANTsTorch's default SyN is the "syntx"
   greedy variant, not ITK's SyN. With our settings (5 levels, high caps, step 0.25) it folds badly:
   subject 131123 had 15 % of the embryo with det(J) <= 0 and the round-9 template came out distorted
   (sharpness 0.0418 vs 0.0446 CPU, RMS change vs round 8 0.138 vs 0.024). `ants_parity` switches to
   ITK's update (read from `itkSyNImageRegistrationMethod.hxx`):
   * step = learning rate in voxels of each level's grid (ANTsTorch multiplied it by sqrt(shrink));
   * no midpoint-gradient averaging (ITK `m_AverageMidPointGradients = false`; ANTsTorch averaged);
   * every iteration both half-warps are inverted twice, `T_inv = Inv(T)`, `T = Inv(T_inv)`
     (fixed point, 20 iterations, mean 0.001 / max 0.1, stop when either is met). This round trip is
     what keeps the fields invertible. With it 131123 has 0 % folding (det(J) p1 0.40, p99 2.27).
2. **Use the convergence test (`--convergence-threshold 1e-9 --convergence-window 10`), never run to the
   caps.** Greedy SyN moves a fixed step every iteration, so it never slows down by itself; ANTs relies
   on `-c [iterations,1e-9,10]` to stop. The ANTsTorch implementation calls ITK's own B-spline fitter,
   so the stop iteration matches ANTs exactly on replayed round-9 logs (10/10 levels).

Even with parity, ANTsTorch levels 3-4 often run to their caps where ANTs stopped early (e.g. 131123:
GPU 218/119/400/300/200, CPU 634/157/50/33/161). Not wrong per se, but it costs time and may
over-register. To be judged on the round-9b template QC.

## 3. How a GPU round runs on Sasquatch

Same inputs, same `-z`, same averaging as `antsMultivariateTemplateConstruction2.sh`; only the SyN
stage moves to the GPU. Output files carry aMTC2's names, so its `shapeupdatetotemplate` runs unchanged.

1. **Linear stages, CPU array** (`e18_linear_cpu.slurm`, cpu-cdbrm or cpu-core, 32c/240G): aMTC2's
   rigid + affine `antsRegistration` call verbatim (MI 32 Regular 0.25, `1000x500x250x0`, `6x4x2x1`,
   `4x2x1x0`, `-u 1 -w [0.01,0.99] -z 1 -r [T,S,1]`), ANTs 2.5.4 container wrappers in `antsbin/`.
2. **SyN, GPU array** (`register_one_gpu*.py`), one subject per task:
   * ANTs-style preprocessing (winsorize 1-99 % to [0,1], histogram match 256 levels / 12 points),
     `normalize_intensities=False` so ANTsTorch does not re-clip;
   * SyN `cc2` radius 4, `flow_sigma = sqrt(2)` (ANTs gives the update variance 2), step 0.25,
     `-f 12x6x4x2x1 -s 6x3x2x1x0`, caps as the CPU round, `--ants-parity`, convergence test,
     `--metric-chunks 8`;
   * 18 µm: one A100, `--gres=gpu:a100:1 --cpus-per-task=16 --mem=96G` (host RSS ~33 GB);
   * 12 µm / 9 µm: one whole node, `--gres=gpu:a100:4 --cpus-per-task=64 --mem=600G`,
     `--final-level-devices cuda:0,cuda:1,cuda:2,cuda:3 --distribute-max-shrink 2`;
   * `antsApplyTransforms` writes `E18_input<jjjj>-modality0-<id>-WarpedToTemplate.nii.gz`.
3. **Template update, CPU** (`e18gpu_update_any.slurm`, **cpu-cdbrm**, 64c/480G): sources
   `summarizeimageset` + `shapeupdatetotemplate` verbatim from the round's aMTC2 copy; `-a 0` mean,
   Laplacian sharpening, gradient step 0.25, `AverageAffineTransformNoRigid`. ~50 min at 18 µm.
4. Chain with `--dependency=afterok` (linear -> GPU array -> update). Review each round before the next.

Each GPU round gets its own ANTsTorch copy (`antstorch_gpu/ANTsTorch_<round>`, via `PYTHONPATH`), so
editing code never changes a running round. Venv: `antstorch_gpu/venv` (python3.12, torch 2.11 cu128,
antspyx 0.6.3).

## 4. What we changed in ANTsTorch

Memory (18 µm went from >95 GiB / out of memory to 29 GiB in the default SyN, 71 GiB with parity):
* drop dead references: caller-held warps from the previous level, last iteration's steps and
  gradients, the loss graph (its AccumulateGrad nodes kept the previous warps alive);
* skip image gradients the objective computed and discarded (5 GiB on the moving grid);
* `AnalyticalGridSample.backward` computes the image gradient only on the rows the grid touches
  (was the whole image per call: 13 GiB at 18 µm, 44 GiB at 12 µm, and slow);
* slab-chunked local CC (`metric_chunks`): exact per-voxel values via a 2r halo;
* lean / row-chunked fixed-point inverse, leaner Anderson; lazy physical grid and row-chunked update
  in the parity configuration; final composition frees each field as soon as it is written;
* `expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF` (fragmentation 15 GiB -> 1.5 GiB).

Speed / correctness:
* separable Gaussian for the pyramid blur and affine smoothing; affine smoothing sigmas are mm and were
  a dense 3-D kernel (335^3 at 36 µm, hours). Pass `affine_smoothing_sigmas` x spacing or fix upstream;
* convergence test (`core/convergence.py`), `ants_parity`, `step_scaling='ants'`, `inverse_schedule='itk'`;
* 3x3 affine inverse on the CPU (cuSOLVER handle creation fails on a nearly full GPU: 6 round-9b
  subjects crashed at the very end);
* warps written uncompressed then compressed with `pigz -p <cores>` (ITK gzip is single-threaded:
  ~15 min per 22 GB field).

Multi-GPU (`core/distributed.py`, `final_level_devices`, `distribute_max_shrink`): the grid is split
into one slab of rows per GPU, images replicated. Each operation uses a halo (2r for CC, kernel radius
for smoothing, step + 2 for the update, max displacement + 2 for inversions and composition), so values
equal the single-GPU path up to summation order (checked: float64 CPU exact to 1e-13; GPU 18 µm split
vs single mean 1.5e-4 voxel). Supports the parity mode, warm-start inverses, any level.

Every change was checked against the unmodified code (warps within the original's own run-to-run GPU
noise; `tests/syn` 316 passed).

## 5. Things that went wrong (do not repeat)

* Running GPU SyN to the caps "to match the CPU caps": the CPU never ran to its caps.
* Using ANTsTorch's default SyN for templates (see 2.1).
* Submitting the update job to cpu-core: it waited 5 h behind other users. Use cpu-cdbrm.
* Assuming A6000s can do E18 18 µm: 48 GB is not enough (needs ~65-72 GiB on one GPU).
* GPU runs are not bit-reproducible (atomic adds in grid_sample backward): identical runs differ by up
  to a few voxels at a handful of locations; mean 2e-4 voxel.

## 6. Open items

* QC of the round-9b template against CPU round 9 (sharpness, RMS, per-subject fit, folding).
* 12 µm with parity on 4 A100s: memory and time (test `build12gpu_test2`).
* Levels 3-4 converge later than ANTs (see 2); investigate the energy (ANTs evaluates
  `0.5*(fixed+moving)` on the virtual domain with full-resolution smoothed images).
* The multi-GPU path drives the GPUs from Python threads; on mismatched local GPUs it gave little
  speed-up. Check scaling on 4 identical A100s.
* Upstream: open a PR with the memory fixes, the convergence test and `ants_parity`; report the
  affine-sigma units, the AnalyticalGridSample gradient and the label-registration set-up separately.
