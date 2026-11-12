"""Crop the external IMPC E15.5 embryos (already rigid-registered to the same 60 um template) onto the
LAMNr grid, with exactly the preprocessing of prep_crops.py, and attach the M3T exencephaly calls.

python prep_external.py --grid 64 128 80 --out crops_64x128x80_external
"""
import argparse, csv, glob, json, os, re
from concurrent.futures import ThreadPoolExecutor
from prep_crops import HERE, make_reference, preprocess_volume

R = "/home/maga/Desktop/R24_resub"
POS_GENES = {"sec24b": "Sec24b", "amt": "Amt", "snx3": "Snx3", "atp13a1": "Atp13a1", "cnnm2": "Cnnm2"}


def m3t_calls():
    calls = {}
    for line in open(f"{R}/phenotype_similarity/classification_table.txt"):
        m = re.match(r"\s+(\S+)\s+([0-9.]+)\s+(\d)/5\s+(EXENCEPHALY|normal)", line)
        if m:
            calls[m.group(1)] = (float(m.group(2)), m.group(4) == "EXENCEPHALY")
    for d in json.load(open(f"{R}/phenotype_similarity/negatives_classification.json")):
        calls[d["sample"]] = (float(d["meanP"]), d["call"] != "normal")
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=3, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = json.load(open(os.path.join(HERE, "cohort.json")))
    ref, sp = make_reference(c, a.grid)
    out = os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)
    calls = m3t_calls()
    items = []
    for d, gene in POS_GENES.items():
        for f in sorted(glob.glob(f"{R}/{d}/rotated/*-transformed.nrrd")):
            items.append((gene, "IMPC exencephaly gene", f))
    for f in sorted(glob.glob(f"{R}/negatives/*/rotated/*-transformed.nrrd")):
        items.append((f.split("/negatives/")[1].split("/")[0], "IMPC lethal-only gene", f))

    def one(it):
        gene, group, f = it
        sample = os.path.basename(f).replace("-transformed.nrrd", "")
        sid = "X" + re.sub(r"[^A-Za-z0-9]", "", sample)
        path = os.path.join(out, f"sub-{sid}_e15.nii.gz")
        p = preprocess_volume(f, ref, sp, path)
        pm, call = calls.get(sample, (float("nan"), None))
        return dict(subject_id=f"sub-{sid}", sample=sample, gene=gene, group=group, m3t_meanP=pm,
                    m3t_call="" if call is None else ("EXENCEPHALY" if call else "normal"), p999=round(p, 1), path=path)

    with ThreadPoolExecutor(12) as ex:
        meta = list(ex.map(one, items))
    with open(os.path.join(out, "meta.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0].keys())); w.writeheader(); w.writerows(meta)
    with open(os.path.join(out, "split.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subject_id", "split"]); [w.writerow([m["subject_id"], "testing"]) for m in meta]
    from collections import Counter
    print(len(meta), Counter(m["group"] for m in meta), "missing M3T calls:", sum(1 for m in meta if not m["m3t_call"]))


if __name__ == "__main__":
    main()
