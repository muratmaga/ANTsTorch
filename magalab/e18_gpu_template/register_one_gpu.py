"""One E18 template-round registration: aMTC2's linear stages (ANTs C++), GPU SyN (ANTsTorch), warp.

Reproduces the per-subject job that antsMultivariateTemplateConstruction2.sh generated for E18
round 9 (build18i), except that the SyN stage runs on the GPU:

  antsRegistration -d 3 --float 1 --verbose 1 -u 1 -w [ 0.01,0.99 ] -z 1
    -r [ T,S,1 ]
    -t Rigid[ 0.1 ]  -m MI[ T,S,1,32,Regular,0.25 ] -c [ 1000x500x250x0,1e-6,10 ] -f 6x4x2x1 -s 4x2x1x0
    -t Affine[ 0.1 ] -m MI[ T,S,1,32,Regular,0.25 ] -c [ 1000x500x250x0,1e-6,10 ] -f 6x4x2x1 -s 4x2x1x0
    -t SyN[ 0.25,2,0 ] -m CC[ T,S,1,4 ] -c [ 2000x600x400x300x200,1e-9,10 ] -f 12x6x4x2x1 -s 6x3x2x1x0
    -o <outdir>/E18_input<jjjj>-<id>-
  antsApplyTransforms -d 3 --float 1 --verbose 1 -i S -o <outdir>/E18_input<jjjj>-modality0-<id>-WarpedToTemplate.nii.gz
    -r T -t <prefix>1Warp.nii.gz -t <prefix>0GenericAffine.mat

Step 1 runs the first three lines (no SyN stage) with ANTs C++, so <prefix>0GenericAffine.mat is the
script's own collapsed initial + rigid + affine transform. Step 2 runs the SyN stage with ANTsTorch on
images preprocessed the way antsRegistration -w/-u does (winsorize 1-99 % -> [0,1], histogram-match the
subject to the template: 256 levels, 12 match points, threshold at mean), starting from that affine,
and writes <prefix>1Warp.nii.gz / 1InverseWarp.nii.gz. Step 3 is the script's own warp line. The files
therefore have exactly the names and conventions shapeupdatetotemplate() reads.

Differences from ANTs that remain (documented, not hidden):
  * the per-level convergence test is optional (--convergence-threshold 1e-9 = ANTs' rule, see
    antstorch/syn/core/convergence.py); without it every level runs to its cap;
  * ANTsTorch's SyN implementation (greedy symmetric, gaussian regularizer, sigma = sqrt(2) voxels for
    ANTs' update-field variance 2), not ANTs' C++ SyN; float32 throughout.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import ants
import numpy as np
import torch

GiB = 2**30


def run_ants(tool, args, ants_bin):
    """Run an ANTs tool: the cluster binaries if ants_bin is given, else ANTsPy's bundled C++ library."""
    if ants_bin:
        cmd = [os.path.join(ants_bin, tool)] + args
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return
    from ants.internal import get_lib_fn
    print("+ (ANTsPy)", tool, " ".join(args), flush=True)
    rc = get_lib_fn(tool)(args)
    if rc != 0:
        raise RuntimeError(f"{tool} exited {rc}")


def ants_preprocess(image, lower=0.01, upper=0.99):
    """antsRegistration -w [lower,upper]: window at the image's quantiles, rescale to [0, 1]."""
    a = image.numpy()
    sample = a.ravel()[:: max(1, a.size // 20_000_000)]          # quantiles from a 20 M-voxel sample
    lo, hi = np.quantile(sample, [lower, upper])
    out = np.clip((a - lo) / max(hi - lo, 1e-12), 0.0, 1.0).astype(np.float32)
    return image.new_image_like(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--index", type=int, required=True, help="position of the subject in the aMTC2 input list")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--outputname", default="E18_")
    ap.add_argument("--ants-bin", default="", help="directory with antsRegistration/antsApplyTransforms")
    ap.add_argument("--iterations", default="2000x600x400x300x200")
    ap.add_argument("--shrink", default="12x6x4x2x1")
    ap.add_argument("--sigmas", default="6x3x2x1x0")
    ap.add_argument("--grad-step", type=float, default=0.25)
    ap.add_argument("--update-variance", type=float, default=2.0)
    ap.add_argument("--cc-radius", type=int, default=4)
    ap.add_argument("--metric-chunks", type=int, default=16)
    ap.add_argument("--convergence-threshold", type=float, default=None,
                    help="antsRegistration -c threshold (e.g. 1e-9); default: run every level to its cap")
    ap.add_argument("--convergence-window", type=int, default=10)
    ap.add_argument("--ants-parity", action="store_true",
                    help="ITK/antsRegistration SyN update: level-voxel step, no midpoint averaging, per-iteration double inversion")
    ap.add_argument("--final-level-devices", default="",
                    help="comma-separated devices for the finest level (slab split), e.g. cuda:0,cuda:1,cuda:2,cuda:3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-linear", action="store_true", help="reuse an existing 0GenericAffine.mat")
    ap.add_argument("--linear-only", action="store_true",
                    help="run only the ANTs linear stages (CPU job) and write <prefix>0GenericAffine.mat")
    ap.add_argument("--distribute-max-shrink", type=int, default=1,
                    help="with --final-level-devices, also split levels with shrink factor <= this (e.g. 2)")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sid = Path(a.subject).name.replace(".nii.gz", "").replace(".nii", "")
    prefix = str(out / f"{a.outputname}input{a.index:04d}-{sid}-")
    deformed = str(out / f"{a.outputname}input{a.index:04d}-modality0-{sid}-WarpedToTemplate.nii.gz")
    T, S = a.template, a.subject
    times = {}
    t0 = time.time()

    # 1. linear stages, verbatim from the script (SyN stage removed)
    if not a.skip_linear:
        lin = ["-d", "3", "--float", "1", "--verbose", "1", "-u", "1", "-w", "[0.01,0.99]", "-z", "1",
               "-r", f"[{T},{S},1]",
               "-t", "Rigid[0.1]", "-m", f"MI[{T},{S},1,32,Regular,0.25]", "-c", "[1000x500x250x0,1e-6,10]",
               "-f", "6x4x2x1", "-s", "4x2x1x0",
               "-t", "Affine[0.1]", "-m", f"MI[{T},{S},1,32,Regular,0.25]", "-c", "[1000x500x250x0,1e-6,10]",
               "-f", "6x4x2x1", "-s", "4x2x1x0",
               "-o", prefix]
        run_ants("antsRegistration", lin, a.ants_bin)
    times["linear_s"] = time.time() - t0
    if a.linear_only:
        print("LINEAR", json.dumps({"subject": sid, "index": a.index, "linear_s": round(times["linear_s"], 1)}), flush=True)
        return 0
    if not os.path.exists(prefix + "0GenericAffine.mat"):
        raise FileNotFoundError(prefix + "0GenericAffine.mat (run the linear stage first)")

    # 2. GPU SyN
    t1 = time.time()
    fixed = ants.image_read(T).clone("float")
    moving = ants.image_read(S).clone("float")
    fixed_p = ants_preprocess(fixed)
    moving_p = ants.histogram_match_image(ants_preprocess(moving), fixed_p, number_of_histogram_bins=256,
                                          number_of_match_points=12, use_threshold_at_mean_intensity=True)
    del fixed, moving
    times["preprocess_s"] = time.time() - t1

    import antstorch.syn.syn as SYN
    from antstorch.ants_transform_io import read_affine_transform
    m, t = read_affine_transform(prefix + "0GenericAffine.mat", 3)
    initial_affine = (torch.as_tensor(m, dtype=torch.float32), torch.as_tensor(t, dtype=torch.float32))

    level_times = []
    orig_fit = SYN._fit_syn_level

    def timed_fit(*args, **kw):
        ts = time.time()
        r = orig_fit(*args, **kw)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        level_times.append(round(time.time() - ts, 1))
        print(f"  SyN level {len(level_times)} done in {level_times[-1]} s, "
              f"peak so far {torch.cuda.max_memory_allocated() / GiB:.1f} GiB", flush=True)
        return r
    SYN._fit_syn_level = timed_fit

    final_devs = [x for x in a.final_level_devices.split(",") if x]
    if final_devs:
        import antstorch.syn.core.distributed as DIST
        orig_dist = DIST.fit_final_level_distributed

        def timed_dist(*args, **kw):
            ts = time.time()
            r = orig_dist(*args, **kw)
            for dv in final_devs:
                torch.cuda.synchronize(dv)
            level_times.append(round(time.time() - ts, 1))
            print(f"  SyN final level ({len(final_devs)} devices) done in {level_times[-1]} s", flush=True)
            return r
        DIST.fit_final_level_distributed = timed_dist
    for dv in range(torch.cuda.device_count()):
        torch.zeros(1, device=f"cuda:{dv}")
        torch.cuda.reset_peak_memory_stats(f"cuda:{dv}")

    tmp = out / f".gpu_{a.index:04d}_{sid}"
    tmp.mkdir(exist_ok=True)
    t2 = time.time()
    result = SYN.syn_registration(
        fixed_p, moving_p, type_of_transform="SyNOnly", initial_affine=initial_affine,
        syn_metric="cc2", neighborhood_radius=a.cc_radius,
        levels=tuple(int(x) for x in a.shrink.split("x")),
        reg_iterations=tuple(int(x) for x in a.iterations.split("x")),
        smoothing_sigmas=tuple(float(x) for x in a.sigmas.split("x")),
        grad_step=a.grad_step, flow_sigma=math.sqrt(a.update_variance), total_sigma=0.0,
        inverse_method="fixed_point", metric_chunks=a.metric_chunks, normalize_intensities=False,
        convergence_threshold=a.convergence_threshold, convergence_window=a.convergence_window,
        final_level_devices=final_devs or None, ants_parity=a.ants_parity,
        distribute_max_shrink=a.distribute_max_shrink,
        outprefix=str(tmp) + "/", device=a.device, dtype=torch.float32)
    torch.cuda.synchronize()
    times["syn_s"] = time.time() - t2
    times["syn_levels_s"] = level_times
    times["syn_level_iterations"] = [len(h) for h in result["level_loss_history"]]
    del result
    times["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / GiB, 2)
    times["gpu_peak_gib_per_device"] = [round(torch.cuda.max_memory_allocated(f"cuda:{dv}") / GiB, 2)
                                        for dv in range(torch.cuda.device_count())]
    times["gpu_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GiB, 2)
    times["gpu"] = torch.cuda.get_device_name(0)
    for name in ("1Warp.nii.gz", "1InverseWarp.nii.gz"):
        shutil.move(str(tmp / name), prefix + name)
    shutil.rmtree(tmp, ignore_errors=True)  # its 0GenericAffine.mat is a copy of the ANTs one

    # 3. the script's own warp line
    t3 = time.time()
    run_ants("antsApplyTransforms", ["-d", "3", "--float", "1", "--verbose", "1", "-i", S, "-o", deformed,
                                     "-r", T, "-t", prefix + "1Warp.nii.gz", "-t", prefix + "0GenericAffine.mat"],
             a.ants_bin)
    times["warp_s"] = time.time() - t3
    times["total_s"] = time.time() - t0
    times = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in times.items()}
    print("TIMES", json.dumps(times), flush=True)
    with open(out / f"timing_{a.index:04d}_{sid}.json", "w") as fh:
        json.dump(dict(subject=sid, index=a.index, **times), fh)


if __name__ == "__main__":
    sys.exit(main())
