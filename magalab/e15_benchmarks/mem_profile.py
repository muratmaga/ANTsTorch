"""Profile ANTsTorch SyN GPU memory on the native 18 um E15 pair (no cropping).

- Caches the affine (computed once, ANTsTorch affine with voxel-scaled sigmas) so each
  experiment measures the deformable SyN stage only.
- Logs per-phase peak memory by wrapping selected antstorch functions.
- On OOM (or always with --snapshot), dumps a CUDA allocator snapshot and prints the
  live memory grouped by the innermost antstorch source line that allocated it.

Usage (Blackwell = physical GPU 2):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python scripts/mem_profile.py --tag baseline
"""

import argparse
import collections
import functools
import json
import os
import pickle
import time
from pathlib import Path

import ants
import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "memprof"
FIXED = "/home/maga/Desktop/R24_resub/E15-atlas03/a02_to_a03/E15_atlas03_r2_template_18um.nii.gz"
MOVING = "/home2/rachel/P01/MagaLab/AtlasBuilding/Atlas_02/Masked_NRRDs/Scan_0005__rec-18um-tx_masked.nrrd"
GiB = 2**30

PHASES = []  # (name, seconds, peak GiB during phase, allocated after)


_STACK = []  # running peaks of the enclosing (nested) phases
_GLOBAL_PEAK = [0]


def traced(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        torch.cuda.synchronize()
        current_max = torch.cuda.max_memory_allocated()
        _GLOBAL_PEAK[0] = max(_GLOBAL_PEAK[0], current_max)
        for i in range(len(_STACK)):  # the reset below would hide this from enclosing phases
            _STACK[i] = max(_STACK[i], current_max)
        _STACK.append(0)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            return fn(*args, **kw)
        finally:
            torch.cuda.synchronize()
            peak = max(_STACK.pop(), torch.cuda.max_memory_allocated())
            _GLOBAL_PEAK[0] = max(_GLOBAL_PEAK[0], peak)
            if _STACK:
                _STACK[-1] = max(_STACK[-1], peak)
            PHASES.append((name, time.time() - t, peak / GiB, torch.cuda.memory_allocated() / GiB))
            print(f"    phase {name:<28s} {time.time() - t:7.1f}s  peak {peak / GiB:6.1f} GiB  "
                  f"after {torch.cuda.memory_allocated() / GiB:6.1f} GiB", flush=True)
    return wrapper


def summarize_snapshot(snap, top=25):
    """Group live (allocated) blocks by the innermost antstorch frame."""
    by_site = collections.Counter()
    for seg in snap["segments"]:
        for blk in seg["blocks"]:
            if blk["state"] != "active_allocated":
                continue
            frames = blk.get("frames") or []
            hits = [fr for fr in frames if "antstorch" in fr.get("filename", "") or "scripts/" in fr.get("filename", "")]
            # innermost antstorch frame plus its caller, so shared helpers are attributed
            site = " <- ".join(f"{fr['filename'].split('antstorch/')[-1]}:{fr.get('line')} {fr.get('name')}"
                               for fr in hits[:2]) or "?"
            by_site[site] += blk["size"]
    total = sum(by_site.values())
    print(f"  live allocated at snapshot: {total / GiB:.1f} GiB")
    for site, size in by_site.most_common(top):
        print(f"    {size / GiB:7.2f} GiB  {site}")
    return by_site


def get_affine(fixed, moving):
    cache = OUT / "affine_cache.pt"
    if cache.exists():
        return torch.load(cache)
    from antstorch.syn.syn import _fit_affine_from_ants
    sp = float(fixed.spacing[0])
    m, t, _ = _fit_affine_from_ants(
        fixed, moving, "cuda", torch.float32, transform_type="Affine", similarity="mse",
        neighborhood_radius=2, num_bins=32, shrink_factors=(4, 2, 1),
        smoothing_sigmas=(2 * sp, 1 * sp, 0.0), iterations=(100, 100, 50),
        learning_rate=(0.05, 0.03, 0.02), multi_start=True, center_of_mass_init=True, verbose=False)
    print(f"  affine peak {torch.cuda.max_memory_allocated() / GiB:.1f} GiB", flush=True)
    pair = (m.detach().cpu(), t.detach().cpu())
    torch.save(pair, cache)
    return pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--snapshot", action="store_true", help="dump allocator snapshot even on success")
    ap.add_argument("--kw", default="{}", help="JSON dict of extra syn_registration kwargs")
    ap.add_argument("--iters", default=None, help="override reg_iterations, e.g. 100,100,20")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    fixed = ants.image_read(FIXED).clone("float")
    moving = ants.image_read(MOVING).clone("float")
    print(f"[{a.tag}] fixed {fixed.shape} ({np.prod(fixed.shape) / 1e6:.0f} M), moving {moving.shape} "
          f"({np.prod(moving.shape) / 1e6:.0f} M), IO {time.time() - t0:.0f}s", flush=True)
    print(f"  devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}", flush=True)

    matrix, translation = get_affine(fixed, moving)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    import antstorch.syn.syn as S
    import antstorch.syn.core.inverse as INV
    S._fit_syn_level = traced("fit_syn_level", S._fit_syn_level)
    S.update_inverse_field_nd = traced("update_inverse_field_nd", S.update_inverse_field_nd)
    S.compute_jacobian_determinant_nd = traced("jacobian_determinant", S.compute_jacobian_determinant_nd)
    S._compose_fixed_grid = traced("compose_fixed_grid", S._compose_fixed_grid)

    kw = json.loads(a.kw)
    if a.iters:
        kw["reg_iterations"] = tuple(int(x) for x in a.iters.split(","))
    torch.cuda.memory._record_memory_history(max_entries=400000)
    oom_snap = {}

    def on_oom(device, alloc, device_alloc, device_free):
        if not oom_snap:
            oom_snap["snap"] = torch.cuda.memory._snapshot()
    torch._C._cuda_attach_out_of_memory_observer(on_oom)
    status, reg = "ok", None
    t1 = time.time()
    try:
        reg = S.syn_registration(fixed, moving, type_of_transform="SyN", syn_metric="cc",
                                 initial_affine=(matrix, translation), outprefix=str(OUT / a.tag) + "_",
                                 device="cuda", dtype=torch.float32, **kw)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as e:
        status = "OOM: " + str(e).splitlines()[0][:160]
    t_reg = time.time() - t1
    overall_peak = max([p[2] for p in PHASES] + [_GLOBAL_PEAK[0] / GiB, torch.cuda.max_memory_allocated() / GiB])
    print(f"[{a.tag}] {status}  SyN {t_reg:.0f}s  peak alloc {overall_peak:.1f} GiB  "
          f"reserved {torch.cuda.max_memory_reserved() / GiB:.1f} GiB", flush=True)

    if status != "ok" or a.snapshot:
        snap = oom_snap.get("snap") or torch.cuda.memory._snapshot()
        with open(OUT / f"{a.tag}_snapshot.pickle", "wb") as fh:
            pickle.dump(snap, fh)
        summarize_snapshot(snap)
    torch.cuda.memory._record_memory_history(enabled=None)

    corr = ""
    if reg is not None:
        w = reg["warpedmovout"].numpy()
        f = fixed.numpy()
        m = f > 0
        corr = float(np.corrcoef(f[m], w[m])[0, 1])
        print(f"[{a.tag}] correlation warped vs template: {corr:.4f}", flush=True)
    per_device = {}
    for i in range(torch.cuda.device_count()):
        per_device[f"cuda:{i} {torch.cuda.get_device_name(i)[:24]}"] = round(
            (overall_peak * GiB if i == 0 else torch.cuda.max_memory_allocated(i)) / GiB, 2)
    if len(per_device) > 1:
        print(f"[{a.tag}] peak per device: {per_device}", flush=True)
    rec = dict(tag=a.tag, status=status, syn_s=round(t_reg, 1), peak_gib=round(overall_peak, 2),
               per_device=per_device,
               reserved_gib=round(torch.cuda.max_memory_reserved() / GiB, 2), corr=corr,
               kw=a.kw, iters=a.iters, alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
               phases=[(n, round(s, 1), round(p, 2)) for n, s, p, _ in PHASES])
    with open(HERE / "memprof_results.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
