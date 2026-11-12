"""Crop the P01 E15.5 60 um rigid-aligned volumes onto a fixed LAMNr grid.

Each embryo: Otsu foreground (largest component, dilated 2 voxels) as mask, background set to 0,
intensity divided by the masked 99.9th percentile and clipped to [0, 1], Gaussian anti-alias
(sigma 0.4 x target spacing, physical), linear resampling onto one isotropic grid that covers
the union of all embryos' foreground boxes. Output names sub-S<id>_e15.nii.gz (the LAMNr loader's
subject key is the 'sub-...' token). Writes a split manifest and a metadata table.

python prep_crops.py --grid 64 128 80 --out crops_64x128x80
"""
import argparse
import csv
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import ants
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def make_reference(c, grid):
    """The fixed LAMNr grid: isotropic spacing covering the cohort's union foreground box, centred on it."""
    sp_in = 0.06
    lo, hi = np.array(c["box_lo"]), np.array(c["box_hi"])
    ext = (hi - lo) * sp_in
    grid = np.array(grid)
    sp = float(np.ceil(1000 * max(ext / grid)) / 1000)
    first = ants.image_read(c["rows"][0]["vol"])
    center_idx = (lo + hi) / 2.0
    center_phys = np.array(first.origin) + np.array(first.direction) @ (center_idx * sp_in)
    origin = center_phys - np.array(first.direction) @ ((grid - 1) / 2.0 * sp)
    ref = ants.make_image(tuple(int(g) for g in grid), spacing=(sp,) * 3, origin=tuple(origin),
                          direction=first.direction, pixeltype="float")
    print(f"grid {tuple(int(g) for g in grid)} at {1000*sp:.0f} um covering {np.round(grid*sp,2)} mm (box {np.round(ext,2)} mm)")
    return ref, sp


def preprocess_volume(vol_path, ref, sp, out_path):
    """Otsu mask (largest component, dilated 2), /p99.9, clip [0,1], anti-alias, resample onto ref."""
    img = ants.image_read(vol_path).clone("float")
    m = ants.otsu_segmentation(img, k=1)
    m = ants.iMath(m, "GetLargestComponent")
    m = ants.iMath(m, "MD", 2)
    arr = img.numpy() * (m.numpy() > 0)
    p = float(np.percentile(arr[arr > 0], 99.9))
    img = img.new_image_like(np.clip(arr / p, 0, 1).astype("float32"))
    img = ants.smooth_image(img, 0.4 * sp, sigma_in_physical_coordinates=True)
    ants.image_write(ants.resample_image_to_target(img, ref, interp_type="linear"), out_path)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=3, required=True, help="H W D in ANTs (x y z) order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-test-wt", type=int, default=14, help="non-Gli2 wild types held out for testing")
    ap.add_argument("--n-val-wt", type=int, default=8)
    a = ap.parse_args()
    c = json.load(open(os.path.join(HERE, "cohort.json")))
    out = os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)
    ref, sp = make_reference(c, a.grid)

    def one(r):
        sid = r["scan"].replace("Scan_", "S")
        path = os.path.join(out, f"sub-{sid}_e15.nii.gz")
        p = preprocess_volume(r["vol"], ref, sp, path)
        return dict(subject_id=f"sub-{sid}", scan=r["scan"], strain=r["Strain"], genotype=r["Genotype"],
                    edema=r["Edema_severity"], exencephaly=r["Brain_exencephaly"], p999=round(p, 1), path=path)

    with ThreadPoolExecutor(12) as ex:
        meta = list(ex.map(one, c["rows"]))

    # split: training / validation = wild types except all Gli2 WT and n_test_wt others; testing = rest
    rng = random.Random(a.seed)
    wt = [m for m in meta if m["genotype"] == "WT/WT"]
    gli2_wt = [m for m in wt if m["strain"] == "Gli2_null"]
    others = [m for m in wt if m["strain"] != "Gli2_null"]
    rng.shuffle(others)
    test_ids = {m["subject_id"] for m in gli2_wt + others[:a.n_test_wt]}
    rest = others[a.n_test_wt:]
    val_ids = {m["subject_id"] for m in rest[:a.n_val_wt]}
    for m in meta:
        if m["genotype"] != "WT/WT" or m["subject_id"] in test_ids:
            m["split"] = "testing"
        elif m["subject_id"] in val_ids:
            m["split"] = "validation"
        else:
            m["split"] = "training"
    with open(os.path.join(out, "split.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject_id", "split"])
        w.writeheader()
        w.writerows({"subject_id": m["subject_id"], "split": m["split"]} for m in meta)
    with open(os.path.join(out, "meta.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    from collections import Counter
    print(Counter(m["split"] for m in meta), Counter((m["split"], m["genotype"]) for m in meta))


if __name__ == "__main__":
    main()
