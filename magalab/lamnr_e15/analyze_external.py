"""External IMPC test: LAMNr normative scores vs the M3T exencephaly classifier.

python analyze_external.py runs/<run>/scores_<ckpt>.csv
(reads scores_<ckpt>_crops_..._external.csv next to it)
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def auc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg])


def main():
    lab_path = Path(sys.argv[1])
    lab = pd.read_csv(lab_path)
    # scores_<ckpt>_<crop dir>.csv; exclude other checkpoints (scores_training_state_it002000_... also
    # matches the glob for scores_training_state)
    cands = [p for p in glob.glob(str(lab_path.with_name(lab_path.stem + "_*external*.csv")))
             if not re.match(r"_it\d+_", Path(p).name[len(lab_path.stem):])]
    assert len(cands) == 1, cands
    ext_path = Path(cands[0])
    print(f"lab scores {lab_path.name}; external scores {ext_path.name}")
    ext = pd.read_csv(ext_path)
    zl = [c for c in ext.columns if c.startswith("z_L")]
    for d in (lab, ext):
        d["z_combined"] = d[zl].mean(1)  # mean level z-score (older CSVs stored a clipped RMS)
    cols = ["bpd"] + zl + ["z_combined"]
    lab_wt = lab[(lab.split == "testing") & (lab.genotype == "WT/WT")]
    if "qc" in ext.columns:
        print(f"excluded (registration QC fail): {list(ext[ext.qc == 'fail'].subject_id)}")
        ext = ext[ext.qc != "fail"].copy()

    def table(e, title):
        called = e[e.m3t_call == "EXENCEPHALY"]
        neg = e[e.group == "IMPC lethal-only gene"]
        gene_pos = e[e.group == "IMPC exencephaly gene"]
        rows = []
        for name, p, n in (("M3T-called exencephalic vs lethal-only negatives", called, neg),
                           ("exencephaly-gene embryos vs lethal-only negatives", gene_pos, neg),
                           ("M3T-normal embryos of exencephaly genes vs negatives", gene_pos[gene_pos.m3t_call == "normal"], neg),
                           ("DOMAIN SHIFT: external negatives vs lab held-out WT", neg, lab_wt)):
            rows.append(dict(comparison=name, n_pos=len(p), n_neg=len(n), **{c: round(auc(p[c].values, n[c].values), 3) for c in cols}))
        print(title)
        print(pd.DataFrame(rows).to_string(index=False))

    table(ext, "AUC on the external IMPC set (normative model trained on 82 lab WT only)")
    if "mask_src" in ext.columns:
        drop = (ext.gene == "Cnnm2") & (ext.mask_src == "refined")
        if drop.any():
            table(ext[~drop], f"\nSame, without the {int(drop.sum())} Cnnm2 embryos whose masks were refined")
    called = ext[ext.m3t_call == "EXENCEPHALY"]
    neg = ext[ext.group == "IMPC lethal-only gene"]
    gene_pos = ext[ext.group == "IMPC exencephaly gene"]
    print(f"\nSpearman correlation with M3T mean probability (all {len(ext)}):")
    for c in cols:
        r, p = spearmanr(ext[c], ext.m3t_meanP)
        print(f"  {c:11s} rho {r:+.3f}  p {p:.1e}")
    print("\nMedian z_combined: lab held-out WT %.2f | external negatives %.2f | external M3T-normal (exen genes) %.2f | "
          "external M3T-exencephalic %.2f" % (lab_wt.z_combined.median(), neg.z_combined.median(),
                                               gene_pos[gene_pos.m3t_call == "normal"].z_combined.median(), called.z_combined.median()))
    print("\nPer gene (median z_L1, z_combined; M3T calls):")
    for g, d in ext.groupby("gene"):
        print(f"  {g:9s} n={len(d)}  z_L1 {d.z_L1.median():6.2f}  z_comb {d.z_combined.median():5.2f}  M3T exencephalic {int((d.m3t_call == 'EXENCEPHALY').sum())}/{len(d)}")


if __name__ == "__main__":
    main()
