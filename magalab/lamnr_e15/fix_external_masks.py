"""Re-crop external embryos whose given whole-body mask includes the rotated scan's field-of-view block.

Flag: crop foreground fraction > 0.55 (lab maximum 0.52). Refined mask = given mask AND
(image > Otsu threshold computed inside the given mask), largest component, holes filled, dilated 2.
Everything else as prep_similarity.py (similarity registration, warp, /p99.9, anti-alias, resample).
Also marks registration failures (similarity scale < 0.6) in meta.csv as qc=fail.
"""
import csv
import os

import ants
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/home2/rachel/P01/MagaLab/Data/RigidAligned"
GRIDS = ["64x128x80", "96x192x128"]


def main():
    from prep_similarity import MARGIN_MM, similarity_scale
    tmpl = ants.image_read(f"{R}/maskedtemplate0__lowRes.nii.gz").clone("float")
    tmask = ants.image_read(f"{R}/maskedtemplate0__lowRes-wholebody-label.nii.gz")
    tmpl_m = tmpl * (tmask > 0)
    idx = np.argwhere(tmask.numpy() > 0)
    lo, hi = idx.min(0), idx.max(0) + 1
    refs = {}
    for g in GRIDS:
        grid = np.array([int(v) for v in g.split("x")])
        ext = (hi - lo) * float(tmpl.spacing[0]) + 2 * MARGIN_MM
        sp = float(np.ceil(1000 * max(ext / grid)) / 1000)
        centre = np.array(tmpl.origin) + np.array(tmpl.direction) @ (((lo + hi - 1) / 2.0) * float(tmpl.spacing[0]))
        origin = centre - np.array(tmpl.direction) @ ((grid - 1) / 2.0 * sp)
        refs[g] = (ants.make_image(tuple(int(v) for v in grid), spacing=(sp,) * 3, origin=tuple(origin),
                                   direction=tmpl.direction, pixeltype="float"), sp)
    meta_path = os.path.join(HERE, "sim_64x128x80_external", "meta.csv")
    meta = list(csv.DictReader(open(meta_path)))
    fixed = 0
    for m in meta:
        frac = float((ants.image_read(m["path"]).numpy() > 0.02).mean())
        m["qc"] = "fail" if float(m["scale"]) < 0.6 else "ok"
        if frac <= 0.55:
            continue
        img = ants.image_read(m["vol"]).clone("float")
        given = ants.image_read(m["mask"])
        given = img.new_image_like((given.numpy() > 0).astype("float32"))
        inside = img.numpy()[given.numpy() > 0]
        # Otsu threshold of the intensities inside the given mask (two classes: FOV block vs tissue)
        hist, edges = np.histogram(inside, bins=256)
        w = np.cumsum(hist); w2 = w[-1] - w
        mu = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2)
        between = (mu[-1] * w / w[-1] - mu) ** 2 / np.maximum(w * w2, 1)
        thr = float(edges[np.argmax(between) + 1])
        mm = given.new_image_like(((given.numpy() > 0) & (img.numpy() > thr)).astype("float32"))
        mm = ants.iMath(ants.iMath(ants.iMath(mm, "GetLargestComponent"), "FillHoles"), "MD", 2)
        img_m = img * mm
        reg = ants.registration(tmpl_m, img_m, type_of_transform="Similarity", aff_metric="mattes",
                                outprefix=os.path.join(HERE, "sim_transforms", m["subject_id"] + "_refined_"))
        mat = reg["fwdtransforms"][0]
        warped = ants.apply_transforms(tmpl, img_m, [mat], interpolator="linear")
        wmask = ants.apply_transforms(tmpl, mm, [mat], interpolator="nearestNeighbor")
        arr = warped.numpy() * (wmask.numpy() > 0)
        p = float(np.percentile(arr[arr > 0], 99.9))
        norm = warped.new_image_like(np.clip(arr / p, 0, 1).astype("float32"))
        for g, (ref, sp) in refs.items():
            sm = ants.smooth_image(norm, 0.4 * sp, sigma_in_physical_coordinates=True)
            out = os.path.join(HERE, f"sim_{g}_external", f"{m['subject_id']}_e15.nii.gz")
            ants.image_write(ants.resample_image_to_target(sm, ref, interp_type="linear"), out)
        m.update(mask_src="refined", scale=round(similarity_scale(mat), 4), p999=round(p, 1))
        fixed += 1
        print(f"refined {m['subject_id']} {m['gene']}: fg {frac:.2f} -> "
              f"{(ants.image_read(m['path']).numpy() > 0.02).mean():.2f}, scale {m['scale']}")
    for g in GRIDS:
        d = os.path.join(HERE, f"sim_{g}_external")
        rows = list(csv.DictReader(open(os.path.join(d, "meta.csv"))))
        upd = {m["subject_id"]: m for m in meta}
        for r in rows:
            u = upd[r["subject_id"]]
            r.update(qc=u["qc"], mask_src=u["mask_src"], scale=u["scale"], p999=u["p999"])
        with open(os.path.join(d, "meta.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"refined {fixed}; registration failures: {[m['subject_id'] for m in meta if m['qc'] == 'fail']}")


if __name__ == "__main__":
    main()
