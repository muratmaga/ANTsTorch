"""Per-iteration SyN cost on the E18 pair (template round 8 vs 131123), one pyramid level at a time.

Calls antstorch's _fit_syn_level directly (as a non-final level, so no end-of-level inverse) with the
round-9 settings (cc2 r4, sigma sqrt(2) voxels, step 0.25, 32 CC slabs), for two iteration counts; the
difference gives seconds per iteration without set-up. Cases:
  L4      shrink 2 on the full template grid (what the A100 jobs report as level 4)
  L5half  shrink 1 on the first half (z) of the template grid, so it fits a 48 GB card
"""
import argparse
import json
import math
import sys
import time

import ants
import numpy as np
import torch

sys.path.insert(0, "/home/maga/Desktop/ANTsTorch_E15/e18gpu")
from register_one_gpu import ants_preprocess  # noqa: E402

T = "/home/maga/Desktop/E18_template_18um_SyN025s2_5lv_round8.nii.gz"
S = "/home/maga/Desktop/R24_resub/hpc_e18/inputs18_controls/131123.nii.gz"
MAT = "/home/maga/Desktop/ANTsTorch_E15/e18gpu/trial/E18_input0000-131123-0GenericAffine.mat"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["L4", "L5half", "L5full"], required=True)
    ap.add_argument("--iters", default="3,13")
    ap.add_argument("--chunks", type=int, default=32)
    a = ap.parse_args()

    import antstorch.syn.syn as SYN
    from antstorch.syn.bridge import ants_image_metadata, ants_image_to_tensor
    from antstorch.ants_transform_io import read_affine_transform

    fixed = ants_preprocess(ants.image_read(T).clone("float"))
    moving = ants.histogram_match_image(ants_preprocess(ants.image_read(S).clone("float")), fixed, 256, 12, True)
    if a.case == "L5half":
        arr = fixed.numpy()
        half = arr.shape[2] // 2
        fixed = ants.from_numpy(np.ascontiguousarray(arr[:, :, :half]), origin=fixed.origin,
                                spacing=fixed.spacing, direction=fixed.direction)
    factor = 2 if a.case == "L4" else 1  # L5full: whole template grid
    sigma = 1.0 if a.case == "L4" else 0.0

    dev, dt = "cuda", torch.float32
    m, t = read_affine_transform(MAT, 3)
    M = torch.as_tensor(m, dtype=dt, device=dev)
    tt = torch.as_tensor(t, dtype=dt, device=dev)
    fm, mm = ants_image_metadata(fixed), ants_image_metadata(moving)
    I = ants_image_to_tensor(fixed, dev, dt, normalize=False)
    J = ants_image_to_tensor(moving, dev, dt, normalize=False)
    I_l, fm_l = SYN._downsample_image(I, fm, factor, False, sigma)
    J_l, mm_l = SYN._downsample_image(J, mm, factor, False, sigma)
    del I, J
    shape = (1,) + fm_l["torch_shape"] + (3,)
    res = {"case": a.case, "gpu": torch.cuda.get_device_name(0), "fixed_level_shape": fm_l["torch_shape"],
           "mvox": round(float(np.prod(fm_l["torch_shape"])) / 1e6, 1)}
    times = {}
    for n in (int(x) for x in a.iters.split(",")):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        state = [torch.zeros(shape, device=dev, dtype=dt), torch.zeros(shape, device=dev, dtype=dt), None, None]
        t0 = time.time()
        SYN._fit_syn_level(
            I_l, J_l, fm_l, mm_l, None, None, None, None, M, tt, warp_state=state,
            iterations=n, similarity="cc2", window_size=9, num_bins=32, flow_sigma=math.sqrt(2.0),
            total_sigma=0.0, regularizer="gaussian", update_field_mesh_size_at_base_level=0,
            total_field_mesh_size_at_base_level=0, bspline_enforce_stationary_boundary=True,
            grad_step=0.25, shrink_factor=factor, antisymmetric=True, inverse_method="fixed_point",
            in_loop_inverse_steps=6, verbose=False, level_index=0, num_levels=2,
            gaussian_sigma_mode="voxel", inverse_schedule="end_of_level", metric_chunks=a.chunks)
        torch.cuda.synchronize()
        times[n] = time.time() - t0
        res[f"peak_gib_{n}it"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        print(f"  {n} iterations: {times[n]:.1f} s", flush=True)
    n0, n1 = sorted(times)
    res["s_per_iter"] = round((times[n1] - times[n0]) / (n1 - n0), 3)
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
