"""Score every E15 crop with a trained LAMNr Glow 3D normative model.

For each embryo: exact bits/dim (-log p(x) / (D ln 2)), the multiscale latent z_l, and pooled latent
features. Then per latent level l: fit mean + low-rank (PCA, rank <= n-1) + diagonal residual covariance
on the TRAINING wild types only and compute each embryo's Mahalanobis distance (Woodbury), standardised
by the training-WT distribution (leave-one-out for training embryos so they are not scored against
themselves). Writes scores.csv (one row per embryo) and features.npz.

python score_e15.py --run runs/wt_64x128x80_20k --crops crops_64x128x80 [--ckpt latest.pt] [--device cuda:0]
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from antstorch.lamnr_flows.tools.lamnr_glow_tool_3d import GlowTool3D
from antstorch.lamnr_flows.core.lamnr_glow_tool_base import (_encode_latents, load_weights_into_model,
                                                             resolve_ckpt_path)


def lowrank_fit(Z, rank):
    """Z: (n, d) training latents. Returns mean, U (d, r), eig (r,), psi (d,) residual variances."""
    mu = Z.mean(0)
    X = Z - mu
    U_, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, len(s))
    eig = (s[:r] ** 2) / max(1, len(Z) - 1)
    U = Vt[:r].T
    resid = X - (X @ U) @ U.T
    psi = np.maximum(resid.var(0, ddof=1) if len(Z) > 1 else np.ones(Z.shape[1]), 1e-8)
    return mu, U, eig, psi


def mahalanobis(z, mu, U, eig, psi):
    """sqrt(d^T (U diag(eig) U^T + diag(psi))^-1 d) by Woodbury."""
    d = z - mu
    sq = np.sqrt(np.clip(eig, 0, None))
    Uw = U / psi[:, None]
    b = sq * (Uw.T @ d)
    M = (sq[:, None] * (Uw.T @ U)) * sq[None, :]
    q = float((d * d / psi).sum() - b @ np.linalg.solve(np.eye(len(eig)) + M, b))
    return math.sqrt(max(q, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rank", type=int, default=40)
    ap.add_argument("--pool", type=int, default=4, help="adaptive pool size per axis for pooled features")
    ap.add_argument("--out", default="")
    ap.add_argument("--extra", default="", help="another crop dir (meta.csv) scored against the same training fit")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    run, crops = here / a.run, here / a.crops
    meta = list(csv.DictReader(open(crops / "meta.csv")))
    split = {r["subject_id"]: r["split"] for r in csv.DictReader(open(crops / "split.csv"))}
    ckpt = resolve_ckpt_path(run / a.ckpt) if a.ckpt else resolve_ckpt_path(run)
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = blob.get("config", blob.get("cfg", {}))
    device = torch.device(a.device)
    tool = GlowTool3D()
    target = (int(cfg["H"]), int(cfg["W"]), int(cfg["D"]))
    model = tool.build_model(cfg, device, target)
    ok, src = load_weights_into_model(model, blob, 0, prefer_ema=True, view_name=None, cfg_views=cfg.get("views"))
    assert ok, "could not load weights"
    tool.prime_if_needed(model, target, device)
    model.eval()
    print(f"loaded {ckpt.name} ({src}), iteration {blob.get('iter', blob.get('it', '?'))}, target {target}")

    n_dims = int(np.prod(target))

    def encode(rows):
        bpd_, levels_, pooled_ = [], None, []
        for i, m in enumerate(rows):
            x = tool.read_image(Path(m["path"]), target).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
                lp = float(model.log_prob(x).item())
                z = _encode_latents(model, x)
            bpd_.append(-lp / (n_dims * math.log(2)))
            if levels_ is None:
                levels_ = [[] for _ in z]
            feats = []
            for l, zl in enumerate(z):
                levels_[l].append(zl.flatten().double().cpu().numpy().astype(np.float32))
                feats.append(F.adaptive_avg_pool3d(zl.float(), a.pool).flatten().cpu().numpy())
            pooled_.append(np.concatenate(feats))
            if (i + 1) % 50 == 0:
                print(f"  encoded {i + 1}/{len(rows)}")
        return bpd_, levels_, pooled_

    bpd, levels, pooled = encode(meta)
    bpd = np.array(bpd)
    train_idx = np.array([split[m["subject_id"]] == "training" for m in meta])

    # per-level Mahalanobis to the training-WT latent distribution (leave-one-out for training embryos)
    dist = np.zeros((len(meta), len(levels)))
    fits = []
    for l, Zl in enumerate(levels):
        Z = np.stack(Zl).astype(np.float64)
        Ztr = Z[train_idx]
        fit_all = lowrank_fit(Ztr, a.rank)
        fits.append(fit_all)
        tr_positions = np.flatnonzero(train_idx)
        for i in range(len(meta)):
            if train_idx[i]:
                k = int(np.flatnonzero(tr_positions == i)[0])
                fit = lowrank_fit(np.delete(Ztr, k, axis=0), a.rank)
            else:
                fit = fit_all
            dist[i, l] = mahalanobis(Z[i], *fit)
        print(f"  level {l + 1}: dim {Z.shape[1]}, done")
    # standardise each level by the training-WT (LOO) distances; combine as the mean z-score across levels
    # (an RMS of z clipped at 0 put most embryos at exactly 0 and lost ranking information)
    zsc = (dist - dist[train_idx].mean(0)) / (dist[train_idx].std(0) + 1e-12)
    combined = zsc.mean(1)
    out = Path(a.out) if a.out else run / f"scores_{ckpt.stem}.csv"
    with open(out, "w", newline="") as fh:
        fields = list(meta[0].keys()) + ["split", "bpd"] + [f"mahal_L{l + 1}" for l in range(len(levels))] + \
                 [f"z_L{l + 1}" for l in range(len(levels))] + ["z_combined"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, m in enumerate(meta):
            row = dict(m, split=split[m["subject_id"]], bpd=round(bpd[i], 5), z_combined=round(combined[i], 4))
            for l in range(len(levels)):
                row[f"mahal_L{l + 1}"] = round(dist[i, l], 4)
                row[f"z_L{l + 1}"] = round(zsc[i, l], 4)
            w.writerow(row)
    np.savez_compressed(out.with_suffix(".npz"), pooled=np.stack(pooled), bpd=bpd, dist=dist,
                        subject=np.array([m["subject_id"] for m in meta]))
    print("wrote", out)

    if a.extra:
        ex_dir = here / a.extra
        ex_meta = list(csv.DictReader(open(ex_dir / "meta.csv")))
        e_bpd, e_levels, e_pooled = encode(ex_meta)
        e_dist = np.array([[mahalanobis(np.asarray(e_levels[l][i], dtype=np.float64), *fits[l])
                            for l in range(len(levels))] for i in range(len(ex_meta))])
        e_z = (e_dist - dist[train_idx].mean(0)) / (dist[train_idx].std(0) + 1e-12)
        e_comb = np.sqrt((np.clip(e_z, 0, None) ** 2).mean(1))
        e_out = out.with_name(out.stem + "_" + ex_dir.name + ".csv")
        with open(e_out, "w", newline="") as fh:
            fields = list(ex_meta[0].keys()) + ["bpd"] + [f"mahal_L{l + 1}" for l in range(len(levels))] + \
                     [f"z_L{l + 1}" for l in range(len(levels))] + ["z_combined"]
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for i, m in enumerate(ex_meta):
                row = dict(m, bpd=round(e_bpd[i], 5), z_combined=round(e_comb[i], 4))
                for l in range(len(levels)):
                    row[f"mahal_L{l + 1}"] = round(e_dist[i, l], 4)
                    row[f"z_L{l + 1}"] = round(e_z[i, l], 4)
                w.writerow(row)
        np.savez_compressed(e_out.with_suffix(".npz"), pooled=np.stack(e_pooled), bpd=np.array(e_bpd), dist=e_dist,
                            subject=np.array([m["subject_id"] for m in ex_meta]))
        print("wrote", e_out)


if __name__ == "__main__":
    main()
