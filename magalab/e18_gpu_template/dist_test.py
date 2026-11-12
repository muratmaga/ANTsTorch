"""E18 18 um pair: finest SyN level on one GPU vs split across several (short iteration counts).

python e18gpu/dist_test.py --devices cuda:0 [--devices cuda:0,cuda:1,cuda:2] --tag NAME
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
OUT = "/home/maga/Desktop/ANTsTorch_E15/e18gpu/dist/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--iters", default="40x20x15x10x12")
    ap.add_argument("--ants-parity", action="store_true")
    ap.add_argument("--distribute-max-shrink", type=int, default=1)
    a = ap.parse_args()
    import os
    os.makedirs(OUT, exist_ok=True)
    import antstorch.syn.syn as SYN
    from antstorch.ants_transform_io import read_affine_transform
    devices = a.devices.split(",")
    fixed = ants_preprocess(ants.image_read(T).clone("float"))
    moving = ants.histogram_match_image(ants_preprocess(ants.image_read(S).clone("float")), fixed, 256, 12, True)
    m, t = read_affine_transform(MAT, 3)
    level_t = []
    orig = SYN._fit_syn_level

    def timed(*args, **kw):
        ts = time.time(); r = orig(*args, **kw); torch.cuda.synchronize(); level_t.append(round(time.time() - ts, 1)); return r
    SYN._fit_syn_level = timed
    import antstorch.syn.core.distributed as DIST
    orig_d = DIST.fit_final_level_distributed

    def timed_d(*args, **kw):
        ts = time.time(); r = orig_d(*args, **kw)
        for d in devices:
            torch.cuda.synchronize(d)
        level_t.append(round(time.time() - ts, 1)); return r
    DIST.fit_final_level_distributed = timed_d
    for d in range(torch.cuda.device_count()):
        torch.zeros(1, device=f"cuda:{d}")          # initialise the device before touching its stats
        torch.cuda.reset_peak_memory_stats(f"cuda:{d}")
    import os as _os
    if _os.environ.get("MEMTRACE"):
        torch.cuda.memory._record_memory_history(max_entries=2000000)
    t0 = time.time()
    kw = dict(final_level_devices=devices, distribute_max_shrink=a.distribute_max_shrink) if len(devices) > 1 else {}
    r = SYN.syn_registration(
        fixed, moving, type_of_transform="SyNOnly",
        initial_affine=(torch.as_tensor(m, dtype=torch.float32), torch.as_tensor(t, dtype=torch.float32)),
        syn_metric="cc2", neighborhood_radius=4, levels=(12, 6, 4, 2, 1),
        reg_iterations=tuple(int(x) for x in a.iters.split("x")), smoothing_sigmas=(6, 3, 2, 1, 0),
        grad_step=0.25, flow_sigma=math.sqrt(2), total_sigma=0.0, inverse_method="fixed_point",
        metric_chunks=8, normalize_intensities=False, outprefix=OUT + a.tag + "_",
        device=devices[0], dtype=torch.float32, ants_parity=a.ants_parity, **kw)
    res = dict(tag=a.tag, devices=devices, total_s=round(time.time() - t0, 1), level_s=level_t,
               peak_gib={torch.cuda.get_device_name(d)[:22] + f" cuda:{d}": round(torch.cuda.max_memory_allocated(f"cuda:{d}") / 2**30, 2)
                         for d in range(torch.cuda.device_count())},
               final_level_iterations=len(r["level_loss_history"][-1]))
    print("RESULT", json.dumps(res), flush=True)
    if _os.environ.get("MEMTRACE"):
        import pickle
        pickle.dump(torch.cuda.memory._snapshot(), open(OUT + a.tag + "_snapshot.pickle", "wb"))


if __name__ == "__main__":
    main()
