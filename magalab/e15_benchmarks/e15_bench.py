"""ANTsTorch GPU registration test on the E15.5 atlas03 template.

Two modes:
  label      antstorch label_image_registration: fixed = atlas03 template + 50 labels,
             moving = native-orientation specimen scan + its corrected 50 labels.
             Reports Dice of the warped specimen labels against the template labels.
  intensity  antstorch syn_registration (affine + SyN, CC): fixed = atlas03 template,
             moving = Atlas_02 masked, rigid-aligned 18 um input. Reports correlation.

Each run records wall time and peak CUDA memory, and appends a row to results.csv.
Run with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 (the Blackwell card).
"""

import argparse
import csv
import os
import time
from pathlib import Path

import ants
import numpy as np
import torch

P01 = Path("/home2/rachel/P01/MagaLab")
A03 = Path("/home/maga/Desktop/R24_resub/E15-atlas03/a02_to_a03")
TEMPLATE_18 = A03 / "E15_atlas03_r2_template_18um.nii.gz"
LABELS_18 = A03 / "labels50_on_atlas03_r2_18um.nii.gz"
TEMPLATE_9 = Path("/home/maga/Desktop/E15_atlas03_round2_RESULT_template_9um.nii.gz")
LABELS_9 = Path("/home/maga/Desktop/E15_atlas03_round2_labels50_9um.nii.gz")
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "outputs"
DTYPES = {"float32": torch.float32, "float64": torch.float64,
          "bfloat16": torch.bfloat16, "float16": torch.float16}


def resample(img, um, label=False):
    sp = um / 1000.0
    if np.allclose(img.spacing, sp, rtol=1e-3):
        return img
    return ants.resample_image(img, (sp,) * 3, use_voxels=False, interp_type=1 if label else 0)


def load_fixed(um):
    if um < 18:
        return ants.image_read(str(TEMPLATE_9)), ants.image_read(str(LABELS_9))
    return resample(ants.image_read(str(TEMPLATE_18)), um), resample(ants.image_read(str(LABELS_18)), um, True)


def dice(a, b, ids):
    a, b = a.numpy().astype(np.int32), b.numpy().astype(np.int32)
    out = {}
    for i in ids:
        x, y = a == i, b == i
        s = x.sum() + y.sum()
        out[int(i)] = 2.0 * np.logical_and(x, y).sum() / s if s else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["label", "intensity"], default="label")
    ap.add_argument("--scan", default="Scan_0005")
    ap.add_argument("--um", type=float, default=36, help="working resolution in microns (9, 18, 36, 72)")
    ap.add_argument("--engine", choices=["antstorch", "ants"], default="antstorch",
                    help="ants = ANTsPy/C++ CPU baseline with matched settings")
    ap.add_argument("--threads", type=int, default=64, help="ITK threads for the ants engine")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=list(DTYPES), default="float32")
    ap.add_argument("--iters", default=None, help="e.g. 100,70,50,20 (levels must match)")
    ap.add_argument("--levels", default=None, help="shrink factors, e.g. 8,4,2,1")
    ap.add_argument("--no-intensity", action="store_true", help="label mode: labels only")
    ap.add_argument("--affine-sigma-vox", action="store_true",
                    help="intensity mode: pass affine smoothing sigmas of 2,1,0 voxels instead of 2,1,0 mm")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    if a.engine == "ants":
        a.device, a.dtype = f"cpu{a.threads}", "float64/32"
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(a.threads)
    tag = f"{a.engine}_{a.mode}_{a.scan}_{a.um:g}um_{a.device}_{a.dtype.replace('/', '-')}"
    tag += "_noint" if a.no_intensity else ""
    tag += "_affvox" if a.affine_sigma_vox else ""
    prefix = OUT / tag
    prefix.mkdir(parents=True, exist_ok=True)
    extra = {}
    if a.levels:
        extra["levels"] = tuple(int(x) for x in a.levels.split(","))
    if a.iters:
        extra["reg_iterations"] = tuple(int(x) for x in a.iters.split(","))

    t0 = time.time()
    fixed, fixed_lab = load_fixed(a.um)
    if a.mode == "label":
        mov = resample(ants.image_read(str(P01 / f"Data/NRRDs-18um/{a.scan}__rec-18um.nrrd")), a.um)
        mov_lab = resample(ants.image_read(str(P01 / f"Labels/CorrectedRAR_50/{a.scan}__rec-18um-label.nrrd")), a.um, True)
    else:
        mov = resample(ants.image_read(str(P01 / f"AtlasBuilding/Atlas_02/Masked_NRRDs/{a.scan}__rec-18um-tx_masked.nrrd")), a.um)
        mov_lab = None
    fixed = fixed.clone("float")
    mov = mov.clone("float")
    t_io = time.time() - t0
    print(f"[{tag}] fixed {fixed.shape} moving {mov.shape}  ({np.prod(fixed.shape)/1e6:.0f} M fixed voxels), IO {t_io:.0f}s", flush=True)

    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # Time spent inside the SyN solver (affine fit + deformable), so CPU set-up can be separated out.
    solve = {"s": 0.0}
    import antstorch.syn.syn as _synmod
    _orig = _synmod.syn_registration

    def _timed(*args, **kw):
        ts = time.time()
        try:
            return _orig(*args, **kw)
        finally:
            if str(kw.get("device", "")).startswith("cuda"):
                torch.cuda.synchronize()
            solve["s"] += time.time() - ts
    _synmod.syn_registration = _timed

    t1 = time.time()
    status = "ok"
    try:
        if a.engine == "ants" and a.mode == "label":
            reg = ants.label_image_registration(
                fixed_lab, mov_lab,
                fixed_intensity_images=None if a.no_intensity else fixed,
                moving_intensity_images=None if a.no_intensity else mov,
                initial_transforms="affine",
                type_of_deformable_transform="antsRegistrationSyN[so]",
                output_prefix=str(prefix) + "/", verbose=a.verbose)
        elif a.engine == "ants":
            # Matched to antstorch's syntx defaults: CC radius 2, 100x100x20 at 4x2x1, grad step 0.25.
            reg = ants.registration(fixed, mov, type_of_transform="SyN", syn_metric="CC", syn_sampling=2,
                                    reg_iterations=(100, 100, 20), grad_step=0.25, flow_sigma=3, total_sigma=0,
                                    outprefix=str(prefix) + "/", verbose=a.verbose)
        elif a.mode == "label":
            from antstorch.syn.label_registration import label_image_registration
            reg = label_image_registration(
                fixed_lab, mov_lab,
                fixed_intensity_images=None if a.no_intensity else fixed,
                moving_intensity_images=None if a.no_intensity else mov,
                initial_transforms="affine",
                type_of_deformable_transform="antsRegistrationSyN[so]",
                output_prefix=str(prefix) + "/", verbose=a.verbose,
                device=a.device, dtype=DTYPES[a.dtype], **extra)
        else:
            if a.affine_sigma_vox:
                # antstorch's affine smoothing sigmas are physical units (mm) and use a dense 3-D kernel;
                # the (2, 1, 0) default means 2 mm here (~56 voxels at 36 um). Pass voxel-equivalent sigmas.
                extra["affine_smoothing_sigmas"] = tuple(s * float(fixed.spacing[0]) for s in (2.0, 1.0, 0.0))
            reg = _synmod.syn_registration(fixed, mov, type_of_transform="SyN", syn_metric="cc",
                                   outprefix=str(prefix) + "/", device=a.device,
                                   dtype=DTYPES[a.dtype], verbose=a.verbose, **extra)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
    except Exception as e:  # record OOMs and dtype rejections as results
        status = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        reg = None
    t_reg = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**30 if a.device.startswith("cuda") else float("nan")
    resv = torch.cuda.max_memory_reserved() / 2**30 if a.device.startswith("cuda") else float("nan")
    import resource
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # GiB (ru_maxrss is KiB on Linux)
    solve_s = solve["s"] if a.engine == "antstorch" else float("nan")
    print(f"[{tag}] {status}  registration {t_reg:.0f}s (solver {solve_s:.0f}s, set-up {t_reg - solve_s:.0f}s)  "
          f"peak GPU alloc {peak:.1f} GiB  reserved {resv:.1f} GiB  host RSS {host_peak:.1f} GiB", flush=True)

    metric = np.nan
    if reg is not None:
        if a.mode == "label":
            warped_lab = ants.apply_transforms(fixed_lab, mov_lab, reg["fwdtransforms"], interpolator="genericLabel")
            ids = [i for i in np.unique(fixed_lab.numpy()) if i]
            d = dice(fixed_lab, warped_lab, ids)
            metric = float(np.nanmean(list(d.values())))
            print(f"[{tag}] mean Dice over {len(ids)} labels: {metric:.3f}  (min {np.nanmin(list(d.values())):.3f})")
            ants.image_write(warped_lab, str(prefix / "warped_labels.nii.gz"))
        else:
            w = reg["warpedmovout"]
            m = fixed.numpy() > 0
            metric = float(np.corrcoef(fixed.numpy()[m], w.numpy()[m])[0, 1])
            print(f"[{tag}] correlation warped vs template (inside template): {metric:.3f}")
            ants.image_write(w, str(prefix / "warped.nii.gz"))

    row = dict(tag=tag, engine=a.engine, mode=a.mode, intensity=not a.no_intensity, scan=a.scan, um=a.um,
               device=a.device, dtype=a.dtype, affine_sigma_vox=a.affine_sigma_vox, levels=a.levels or "default", iters=a.iters or "default",
               fixed_mvox=round(np.prod(fixed.shape) / 1e6, 1), status=status, io_s=round(t_io, 1),
               reg_s=round(t_reg, 1), solver_s=round(solve_s, 1) if np.isfinite(solve_s) else "",
               peak_alloc_gib=round(peak, 2) if np.isfinite(peak) else "",
               peak_reserved_gib=round(resv, 2) if np.isfinite(resv) else "",
               host_rss_gib=round(host_peak, 1), metric=round(metric, 4) if np.isfinite(metric) else "")
    f = HERE / "results.csv"
    rows = []
    if f.exists():  # migrate older rows to the current column set
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
    rows.append(row)
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
