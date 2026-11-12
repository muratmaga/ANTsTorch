"""Counterfactual 'nearest normal' maps from the LAMNr normative model.

For an embryo x: z = f(x) (multiscale latents). For each latent level, z is replaced by its projection
onto the training wild types' principal subspace, z' = mu + U U^T (z - mu) (rank r), optionally keeping the
finest levels unchanged; x' = f^-1(z') is the nearest image the normal model can express, and
|x - x'| shows what it cannot. Writes NIfTI maps and a PNG of mid-slices.

python cf_maps.py --run runs/wt_64x128x80_20k --crops crops_64x128x80 --subjects sub-S0058 sub-S0003 ...
"""
import argparse
import csv
from pathlib import Path

import ants
import numpy as np
import torch

from antstorch.lamnr_flows.tools.lamnr_glow_tool_3d import GlowTool3D
from antstorch.lamnr_flows.core.lamnr_glow_tool_base import (_encode_latents, load_weights_into_model,
                                                             resolve_ckpt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--extra", default="", help="crop dir of additional subjects (e.g. the external set)")
    ap.add_argument("--rank", type=int, default=20)
    ap.add_argument("--keep-levels", type=int, nargs="*", default=[], help="1-based latent levels left unchanged")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="cf")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    run, crops = here / a.run, here / a.crops
    meta = list(csv.DictReader(open(crops / "meta.csv")))
    split = {r["subject_id"]: r["split"] for r in csv.DictReader(open(crops / "split.csv"))}
    if a.extra:
        meta += list(csv.DictReader(open(here / a.extra / "meta.csv")))
    by_id = {m["subject_id"]: m for m in meta}
    blob = torch.load(str(resolve_ckpt_path(run)), map_location="cpu", weights_only=False)
    cfg = blob.get("config", blob.get("cfg", {}))
    device = torch.device(a.device)
    tool = GlowTool3D()
    target = (int(cfg["H"]), int(cfg["W"]), int(cfg["D"]))
    model = tool.build_model(cfg, device, target)
    assert load_weights_into_model(model, blob, 0, prefer_ema=True, view_name=None, cfg_views=cfg.get("views"))[0]
    tool.prime_if_needed(model, target, device)
    model.eval()

    def enc(path):
        x = tool.read_image(Path(path), target).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
            return x, _encode_latents(model, x)

    # wild-type principal subspace per level, from the training embryos
    train = [m for m in meta if split.get(m["subject_id"]) == "training"]
    Z = None
    for m in train:
        _, z = enc(m["path"])
        if Z is None:
            Z = [[] for _ in z]
        for l, zl in enumerate(z):
            Z[l].append(zl.flatten().double().cpu().numpy())
    basis = []
    for Zl in Z:
        Zl = np.stack(Zl)
        mu = Zl.mean(0)
        _, _, Vt = np.linalg.svd(Zl - mu, full_matrices=False)
        basis.append((mu, Vt[:a.rank]))

    out_dir = run / f"cf_maps_{a.tag}"
    out_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(a.subjects), 6, figsize=(17, 3.1 * len(a.subjects)))
    ax = np.atleast_2d(ax)
    for r, sid in enumerate(a.subjects):
        m = by_id[sid]
        x, z = enc(m["path"])
        zc = []
        for l, zl in enumerate(z):
            if (l + 1) in a.keep_levels:
                zc.append(zl)
                continue
            mu, V = basis[l]
            v = zl.flatten().double().cpu().numpy()
            proj = mu + V.T @ (V @ (v - mu))
            zc.append(torch.from_numpy(proj).to(zl.dtype).to(device).view_as(zl))
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
            xh, _ = model.forward_and_log_det(zc)
        xa = x[0, 0].cpu().numpy()
        xc = xh[0, 0].float().cpu().numpy()
        diff = np.abs(xa - xc) * (xa > 0.02)
        ref = ants.image_read(m["path"])
        for name, arr in (("orig", xa), ("counterfactual", xc), ("absdiff", diff)):
            ants.image_write(ref.new_image_like(arr.astype("float32")), str(out_dir / f"{sid}_{name}.nii.gz"))
        label = m.get("genotype") or m.get("gene")
        extra = m.get("exencephaly") or m.get("m3t_call", "")
        sx, sz = xa.shape[0] // 2, xa.shape[2] // 2
        vmax = np.percentile(diff[diff > 0], 99.5) if (diff > 0).any() else 1
        for c, (img, cm, vm) in enumerate(((xa[sx], "gray", 1), (xc[sx], "gray", 1), (diff[sx], "magma", vmax),
                                           (xa[:, :, sz], "gray", 1), (xc[:, :, sz], "gray", 1), (diff[:, :, sz], "magma", vmax))):
            ax[r, c].imshow(img, cmap=cm, vmin=0, vmax=vm)
            ax[r, c].axis("off")
        ax[r, 0].set_title(f"{sid} {m.get('strain', m.get('gene', ''))} {label} exen={extra}  mean|d| {diff[xa > 0.02].mean():.3f}",
                           fontsize=8, loc="left")
    for c, t in enumerate(("original (sagittal)", "nearest normal", "|difference|", "original (axial)", "nearest normal", "|difference|")):
        ax[0, c].text(0.5, 1.12, t, transform=ax[0, c].transAxes, ha="center", fontsize=9)
    plt.tight_layout()
    png = out_dir / "cf_maps.png"
    plt.savefig(png, dpi=95)
    print("wrote", png)


if __name__ == "__main__":
    main()
