"""Similarity-normalised LAMNr crops for the lab cohort and the external IMPC embryos.

Each embryo: whole-body mask (lab: lowRes_Masks-wholebody; external: *-transformed-wholebody-label),
masked image registered to the masked 60 um template with a Similarity transform (rigid + isotropic
scale; size differences across centres/scans removed, body proportions kept, scale recorded), warped
image x warped mask, divided by the masked 99.9th percentile and clipped to [0,1], anti-aliased and
resampled onto a grid covering the template's whole-body box + margin. Writes one crop set per grid
with meta.csv and split.csv (same split as crops_64x128x80; external embryos are 'testing').

python prep_similarity.py --grids 64x128x80 96x192x128
"""
import argparse
import csv
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import ants
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/home2/rachel/P01/MagaLab/Data/RigidAligned"
EXT = "/home/maga/Desktop/R24_resub"
POS_GENES = {"sec24b": "Sec24b", "amt": "Amt", "snx3": "Snx3", "atp13a1": "Atp13a1", "cnnm2": "Cnnm2"}
MARGIN_MM = 0.3


def similarity_scale(mat_path):
    t = ants.read_transform(mat_path)
    A = np.asarray(t.parameters[:9]).reshape(3, 3)
    return float(abs(np.linalg.det(A)) ** (1 / 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=["64x128x80"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
    tmpl = ants.image_read(f"{R}/maskedtemplate0__lowRes.nii.gz").clone("float")
    tmask = ants.image_read(f"{R}/maskedtemplate0__lowRes-wholebody-label.nii.gz")
    tmpl_m = tmpl * (tmask > 0)
    idx = np.argwhere(tmask.numpy() > 0)
    lo, hi = idx.min(0), idx.max(0) + 1
    sp_in = float(tmpl.spacing[0])
    refs = {}
    for g in a.grids:
        grid = np.array([int(v) for v in g.split("x")])
        ext = (hi - lo) * sp_in + 2 * MARGIN_MM
        sp = float(np.ceil(1000 * max(ext / grid)) / 1000)
        centre = np.array(tmpl.origin) + np.array(tmpl.direction) @ (((lo + hi - 1) / 2.0) * sp_in)
        origin = centre - np.array(tmpl.direction) @ ((grid - 1) / 2.0 * sp)
        refs[g] = (ants.make_image(tuple(int(v) for v in grid), spacing=(sp,) * 3, origin=tuple(origin),
                                   direction=tmpl.direction, pixeltype="float"), sp)
        os.makedirs(os.path.join(HERE, f"sim_{g}"), exist_ok=True)
        print(f"{g}: {1000 * sp:.0f} um, covers {np.round(grid * sp, 2)} mm (template box {np.round((hi - lo) * sp_in, 2)} mm)")

    # subjects: lab cohort (same split as the rigid crops) + external
    cohort = json.load(open(os.path.join(HERE, "cohort.json")))
    old_split = {r["subject_id"]: r["split"] for r in csv.DictReader(open(os.path.join(HERE, "crops_64x128x80/split.csv")))}
    items = []
    for r in cohort["rows"]:
        sid = "sub-" + r["scan"].replace("Scan_", "S")
        mask = glob.glob(f"{R}/lowRes_Masks-wholebody/{r['scan']}__*wholebody-label.nii.gz")
        items.append(dict(subject_id=sid, source="lab", vol=r["vol"], mask=mask[0] if mask else "",
                          strain=r["Strain"], genotype=r["Genotype"], edema=r["Edema_severity"],
                          exencephaly=r["Brain_exencephaly"], gene="", group="lab", split=old_split[sid]))
    from prep_external import m3t_calls
    calls = m3t_calls()
    ext_list = [(g, "IMPC exencephaly gene", f) for d, g in POS_GENES.items()
                for f in sorted(glob.glob(f"{EXT}/{d}/rotated/*-transformed.nrrd"))]
    ext_list += [(f.split("/negatives/")[1].split("/")[0], "IMPC lethal-only gene", f)
                 for f in sorted(glob.glob(f"{EXT}/negatives/*/rotated/*-transformed.nrrd"))]
    for gene, group, f in ext_list:
        sample = os.path.basename(f).replace("-transformed.nrrd", "")
        pm, call = calls.get(sample, (float("nan"), None))
        items.append(dict(subject_id="sub-X" + re.sub(r"[^A-Za-z0-9]", "", sample), source="external", vol=f,
                          mask=f.replace("-transformed.nrrd", "-transformed-wholebody-label.nii.gz"),
                          strain="", genotype="", edema="", exencephaly="", gene=gene, group=group, split="testing",
                          sample=sample, m3t_meanP=pm, m3t_call="" if call is None else ("EXENCEPHALY" if call else "normal")))

    tx_dir = os.path.join(HERE, "sim_transforms")
    os.makedirs(tx_dir, exist_ok=True)

    def one(it):
        img = ants.image_read(it["vol"]).clone("float")
        if it["mask"] and os.path.exists(it["mask"]):
            m = ants.image_read(it["mask"])
            m = ants.resample_image_to_target(m, img, interp_type="nearestNeighbor") if m.shape != img.shape else m
            m = (m > 0).astype("float32") if hasattr(m, "astype") else m
            m = img.new_image_like((m.numpy() > 0).astype("float32"))
            mask_src = "given"
        else:
            m = ants.iMath(ants.iMath(ants.otsu_segmentation(img, k=1), "GetLargestComponent"), "MD", 2)
            mask_src = "otsu"
        img_m = img * m
        reg = ants.registration(tmpl_m, img_m, type_of_transform="Similarity", aff_metric="mattes",
                                outprefix=os.path.join(tx_dir, it["subject_id"] + "_"))
        mat = reg["fwdtransforms"][0]
        warped = ants.apply_transforms(tmpl, img_m, [mat], interpolator="linear")
        wmask = ants.apply_transforms(tmpl, m, [mat], interpolator="nearestNeighbor")
        arr = warped.numpy() * (wmask.numpy() > 0)
        p = float(np.percentile(arr[arr > 0], 99.9))
        norm = warped.new_image_like(np.clip(arr / p, 0, 1).astype("float32"))
        out = dict(it, mask_src=mask_src, scale=round(similarity_scale(mat), 4), p999=round(p, 1),
                   mattes=round(float(ants.image_similarity(tmpl_m, warped * wmask, metric_type="MattesMutualInformation")), 4))
        for g, (ref, sp) in refs.items():
            sm = ants.smooth_image(norm, 0.4 * sp, sigma_in_physical_coordinates=True)
            path = os.path.join(HERE, f"sim_{g}", f"{it['subject_id']}_e15.nii.gz")
            ants.image_write(ants.resample_image_to_target(sm, ref, interp_type="linear"), path)
            out[f"path_{g}"] = path
        return out

    with ThreadPoolExecutor(a.jobs) as ex:
        meta = list(ex.map(one, items))
    keys = sorted({k for m in meta for k in m})
    for g in refs:
        d = os.path.join(HERE, f"sim_{g}")
        lab = [dict(m, path=m[f"path_{g}"]) for m in meta if m["source"] == "lab"]
        ext = [dict(m, path=m[f"path_{g}"]) for m in meta if m["source"] == "external"]
        for name, rows in (("meta.csv", lab), ("meta_external.csv", ext)):
            with open(os.path.join(d, name), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys + ["path"], extrasaction="ignore")
                w.writeheader(); w.writerows(rows)
        with open(os.path.join(d, "split.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["subject_id", "split"])
            [w.writerow([m["subject_id"], m["split"]]) for m in meta]
        # external crops in a sibling directory so the trainer's glob only sees lab subjects
        exd = d + "_external"
        os.makedirs(exd, exist_ok=True)
        for m in ext:
            dst = os.path.join(exd, os.path.basename(m["path"]))
            if not os.path.exists(dst):
                os.replace(m["path"], dst)
            m["path"] = dst
        with open(os.path.join(exd, "meta.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys + ["path"], extrasaction="ignore")
            w.writeheader(); w.writerows(ext)
        with open(os.path.join(d, "split.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["subject_id", "split"])
            [w.writerow([m["subject_id"], m["split"]]) for m in meta if m["source"] == "lab"]
    sc = np.array([m["scale"] for m in meta])
    print(f"{len(meta)} embryos; similarity scale {sc.min():.3f}-{sc.max():.3f} (lab median "
          f"{np.median([m['scale'] for m in meta if m['source'] == 'lab']):.3f}, external median "
          f"{np.median([m['scale'] for m in meta if m['source'] == 'external']):.3f})")


if __name__ == "__main__":
    main()
