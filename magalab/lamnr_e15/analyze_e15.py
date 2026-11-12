"""Evaluate LAMNr normative scores against the exencephaly and Gli2 questions.

python analyze_e15.py runs/wt_64x128x80_20k/scores_latest.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def auc(df, pos, neg, col):
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    s = np.r_[df.loc[pos, col].values, df.loc[neg, col].values]
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


def main():
    path = Path(sys.argv[1])
    df = pd.read_csv(path)
    feats = np.load(path.with_suffix(".npz"))["pooled"]
    test_wt = (df.split == "testing") & (df.genotype == "WT/WT")
    exen = df.exencephaly.astype(str).str.upper() == "TRUE"
    gli2 = df.strain == "Gli2_null"
    comps = {
        "exencephaly vs held-out WT": (exen, test_wt),
        "non-exencephalic MUT/MUT vs held-out WT": ((df.genotype == "MUT/MUT") & ~exen, test_wt),
        "all MUT/MUT vs held-out WT": (df.genotype == "MUT/MUT", test_wt),
        "Gli2 MUT/MUT vs Gli2 WT": (gli2 & (df.genotype == "MUT/MUT"), gli2 & (df.genotype == "WT/WT")),
        "Gli2 MUT/WT vs Gli2 WT": (gli2 & (df.genotype == "MUT/WT"), gli2 & (df.genotype == "WT/WT")),
        "all MUT/WT vs held-out WT": (df.genotype == "MUT/WT", test_wt),
    }
    zl = [c for c in df.columns if c.startswith("z_L")]
    df["z_combined"] = df[zl].mean(1)  # mean level z-score (older CSVs stored a clipped RMS)
    cols = ["bpd"] + zl + ["z_combined"]
    rows = []
    for name, (pos, neg) in comps.items():
        rows.append(dict(comparison=name, n_pos=int(pos.sum()), n_neg=int(neg.sum()),
                         **{c: round(auc(df, pos, neg, c), 3) for c in cols}))
    print("Unsupervised AUC (score higher = more abnormal; normative model trained on 82 WT only)")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nPer-strain AUC, MUT/MUT vs held-out WT, z_combined:")
    for s, g in df[df.genotype == "MUT/MUT"].groupby("strain"):
        pos = (df.strain == s) & (df.genotype == "MUT/MUT")
        print(f"  {s:12s} n={pos.sum():2d}  AUC {auc(df, pos, test_wt, 'z_combined'):.3f}  "
              f"exencephalic {int((pos & exen).sum())}")

    print("\nScores by group (median z_combined, bpd):")
    grp = df.assign(group=np.where(df.split != "testing", "WT train/val", df.genotype + " test"))
    print(grp.groupby("group")[["z_combined", "bpd"]].median().round(3).to_string())

    # supervised check on pooled latents: logistic regression, 5-fold CV
    def cv_auc(mask_pos, mask_neg, seed=0):
        idx = np.flatnonzero(mask_pos | mask_neg)
        y = mask_pos[idx].astype(int).values
        k = min(5, int(min(y.sum(), (1 - y).sum())))
        if k < 2:
            return np.nan
        clf = make_pipeline(StandardScaler(), PCA(n_components=min(20, len(idx) - k - 1)),
                            LogisticRegression(C=0.1, max_iter=5000))
        p = cross_val_predict(clf, feats[idx], y, cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
        return roc_auc_score(y, p)
    all_wt = df.genotype == "WT/WT"
    print("\nSupervised (pooled latents, PCA20 + logistic, 5-fold CV):")
    print(f"  exencephaly vs all WT          AUC {cv_auc(exen, all_wt):.3f}")
    print(f"  Gli2 MUT/MUT vs Gli2 WT        AUC {cv_auc(gli2 & (df.genotype == 'MUT/MUT'), gli2 & (df.genotype == 'WT/WT')):.3f}")
    print(f"  Gli2 MUT/WT vs Gli2 WT         AUC {cv_auc(gli2 & (df.genotype == 'MUT/WT'), gli2 & (df.genotype == 'WT/WT')):.3f}")

    # Gli2 heterozygote position on the WT -> MUT axis (axis fit on WT and MUT only, as in GLI2_RESULT.md)
    wt_i = np.flatnonzero(gli2 & (df.genotype == "WT/WT"))
    mu_i = np.flatnonzero(gli2 & (df.genotype == "MUT/MUT"))
    het_i = np.flatnonzero(gli2 & (df.genotype == "MUT/WT"))
    X = StandardScaler().fit(feats[np.r_[wt_i, mu_i]]).transform(feats)
    axis = X[mu_i].mean(0) - X[wt_i].mean(0)
    proj = X @ axis / np.linalg.norm(axis)
    lo, hi = proj[wt_i].mean(), proj[mu_i].mean()
    frac = (proj[het_i].mean() - lo) / (hi - lo)
    print(f"\nGli2 latent WT->MUT axis: WT {lo:.1f}, HET {proj[het_i].mean():.1f}, MUT {hi:.1f}; "
          f"HET at {100 * frac:.1f}% of the WT->MUT span (registration shape analysis: 3.2%)")
    print("  (axis fit in-sample on the same WT and MUT; HET never used to define it)")


if __name__ == "__main__":
    main()
