"""Finest SyN level split across several GPUs (slab decomposition along the first spatial axis).

The fixed (virtual) grid's rows (torch dim 0 of ``(D, H, W)``) are divided into one contiguous core
slab per device. Each device owns its slab of the two half-warps and holds full copies of the two
level images (warped points can land anywhere). Every operation of a greedy-SyN iteration is local up
to a bounded halo, so each core voxel gets exactly the value the single-device code computes:

  * local CC metric + gradient: warp halo ``2 r`` rows (one box window for the means, one for the
    variances); gradient contributions that fall in a neighbour's rows are sent to it and summed;
  * gaussian regularization of the gradient: halo = kernel radius, boundary rim from global indices;
  * CFL step: one global max (scalar reduction);
  * Eulerian update phi <- phi o (Id - delta) - delta: halo ``ceil(step) + 2`` rows;
  * end-of-level fixed-point inverse and the final half-warp composition: halo
    ``ceil(max |displacement| in voxels) + 2`` rows, measured once; the fixed-point clip threshold and
    the convergence test use the global max / mean error (scalar reductions per step).

Within a device, every phase works through its rows in sub-slabs and only small halo strips are
copied between devices, so the per-device peak is about six core-slab fields plus the two images.
Physical coordinates are rebuilt per sub-slab from global indices (same arithmetic as the full grid).

Only floating-point summation order differs from the single-device path. Supported configuration:
gaussian regularizer in voxel units, gradient_descent, antisymmetric, total_sigma = 0, end_of_level
inverse with the fixed-point solver, plain cc/cc2/lncc/lncc2 without masks. One process drives all
devices, one Python thread per device per phase.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..._torch_compat import grid_sample
from ..bridge import metadata_tensors_from_dict
from .chunked_metric import _local_cc_terms
from .grid import physical_to_normalized_torch_cached, prepare_mid_images_and_gradients_torch
from .smoothing import get_cached_gaussian_kernel_1d, separable_gaussian_filter


# --------------------------------------------------------------------------------------------- geometry
def _rows_meta(meta: Dict[str, tuple], a: int, b: int) -> Dict[str, tuple]:
    """Metadata of rows [a, b) of a grid (torch dim 0 = ITK's last axis)."""
    sub = dict(meta)
    ts = list(meta["torch_shape"])
    ts[0] = b - a
    sub["torch_shape"] = tuple(ts)
    sub["shape"] = tuple(reversed(ts))
    d = np.asarray(meta["direction"], dtype=np.float64).reshape(len(ts), len(ts))
    offset = np.zeros(len(ts))
    offset[-1] = a * meta["spacing"][-1]          # ITK index (0, 0, a)
    sub["origin"] = tuple(np.asarray(meta["origin"], dtype=np.float64) + d @ offset)
    return sub


def _phys_rows(meta: Dict[str, tuple], a: int, b: int, device, dtype) -> Tensor:
    """Rows [a, b) of get_physical_grid_torch(meta): the same arithmetic on the same global indices."""
    shape = meta["torch_shape"]
    dim = len(shape)
    spacing = torch.tensor(tuple(reversed(meta["spacing"])), device=device, dtype=dtype)
    origin = torch.tensor(tuple(reversed(meta["origin"])), device=device, dtype=dtype)
    dir_arr = np.asarray(meta["direction"]).reshape(dim, dim)[::-1, ::-1].copy()
    direction = torch.tensor(dir_arr, device=device, dtype=dtype)
    axes = [torch.arange(a, b, device=device, dtype=dtype)] + \
           [torch.arange(s, device=device, dtype=dtype) for s in shape[1:]]
    idxs = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    flat = (idxs * spacing).view(-1, dim)
    return (flat @ direction.t() + origin).view(b - a, *shape[1:], dim).unsqueeze(0)


class PhysicalRows:
    """Stand-in for the full physical grid ``(1, D, H, W, dim)`` that builds rows on demand.

    ``X[:, a:b]`` returns exactly the rows get_physical_grid_torch would hold (same arithmetic on the
    same global indices), so code that only ever slices whole rows can drop the stored grid.
    """

    def __init__(self, meta, device, dtype):
        self.meta, self.device, self.dtype = meta, device, dtype
        self.shape = (1,) + tuple(meta["torch_shape"]) + (len(meta["torch_shape"]),)

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None) and isinstance(key[1], slice)):
            raise TypeError("PhysicalRows supports X[:, a:b] only")
        a, b, step = key[1].indices(self.shape[1])
        if step != 1:
            raise TypeError("PhysicalRows supports contiguous row ranges only")
        return _phys_rows(self.meta, a, b, self.device, self.dtype)


def _boundary_rows(meta: Dict[str, tuple], a: int, b: int, device, dtype) -> Tensor:
    """Rows [a, b) of get_boundary_mask(meta['torch_shape']): zero on the global faces."""
    shape = meta["torch_shape"]
    mask = torch.ones((1, b - a, *shape[1:], 1), device=device, dtype=dtype)
    if a == 0:
        mask[:, 0] = 0
    if b == shape[0]:
        mask[:, -1] = 0
    for i in range(1, len(shape)):
        sl = [slice(None)] * mask.ndim
        sl[i + 1] = slice(0, 1)
        mask[tuple(sl)] = 0
        sl[i + 1] = slice(-1, None)
        mask[tuple(sl)] = 0
    return mask


def _sample_rows(field_ext: Tensor, ext_meta_t: Dict[str, Tensor], coords_phys: Tensor) -> Tensor:
    """grid_sample a channel-last field stored on a row range at physical coordinates (bilinear, border)."""
    norm = physical_to_normalized_torch_cached(coords_phys, ext_meta_t["shape_t"], ext_meta_t["spacing_t"],
                                               ext_meta_t["origin_t"], ext_meta_t["direction_t"])
    out = grid_sample(field_ext.movedim(-1, 1), norm, mode="bilinear", padding_mode="border", align_corners=True)
    return out.movedim(1, -1)


class _Layout:
    def __init__(self, D: int, devices: Sequence):
        self.D = D
        self.devices = [torch.device(d) for d in devices]
        n = len(self.devices)
        edges = [round(i * D / n) for i in range(n + 1)]
        self.cores = [(edges[i], edges[i + 1]) for i in range(n)]
        if any(z1 - z0 < 1 for z0, z1 in self.cores):
            raise ValueError("more devices than rows")

    def gather(self, parts: List[Tensor], a: int, b: int, device) -> Optional[Tuple[Tensor, int, int]]:
        """Rows [a, b) (clipped to the grid) assembled from the owners' core slabs onto ``device``."""
        a, b = max(0, a), min(self.D, b)
        if a >= b:
            return None
        pieces = []
        for (z0, z1), part in zip(self.cores, parts):
            lo, hi = max(a, z0), min(b, z1)
            if lo < hi:
                pieces.append(part[:, lo - z0:hi - z0].to(device))
        return (pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)), a, b


class _RowSource:
    """Rows of a field around one device's core: its own core (a view) plus halo strips from neighbours."""

    def __init__(self, layout: _Layout, k: int, parts: List[Tensor], halo: int):
        self.z0, self.z1 = layout.cores[k]
        self.core = parts[k]
        dev = layout.devices[k]
        self.top = layout.gather(parts, self.z0 - halo, self.z0, dev)
        self.bot = layout.gather(parts, self.z1, self.z1 + halo, dev)
        self.lo = self.top[1] if self.top else self.z0
        self.hi = self.bot[2] if self.bot else self.z1

    def rows(self, a: int, b: int) -> Tuple[Tensor, int, int]:
        a, b = max(a, self.lo), min(b, self.hi)
        pieces = []
        for src in (self.top, (self.core, self.z0, self.z1), self.bot):
            if not src:
                continue
            t, s0, s1 = src
            lo, hi = max(a, s0), min(b, s1)
            if lo < hi:
                pieces.append(t[:, lo - s0:hi - s0])
        return (pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)), a, b


def _max_disp_voxels(parts: List[Tensor], spacing_min: float) -> float:
    return max(float(torch.linalg.norm(p, dim=-1).max()) for p in parts) / spacing_min


# ------------------------------------------------------------------------------------------- the level
def fit_final_level_distributed(
    I_by_dev: Dict[str, Tensor], J_by_dev: Dict[str, Tensor],
    fixed_meta: Dict[str, tuple], moving_meta: Dict[str, tuple],
    warp_l2r_parts: List[Tensor], warp_r2l_parts: List[Tensor], layout: _Layout,
    M_phys: Tensor, t_phys: Tensor, *,
    iterations: int, similarity: str, window_size: int, flow_sigma: float, grad_step: float,
    shrink_factor: int, antisymmetric: bool, end_of_level_inverse_steps: int,
    use_analytical_gradients: bool = True, metric_subslab_rows: int = 64, subslab_rows: int = 128,
    convergence_threshold: Optional[float] = None, convergence_window: int = 10,
    max_error_threshold: float = 0.1, mean_error_threshold: float = 0.001,
    verbose: bool = False,
    inverse_schedule: str = "end_of_level", step_scaling: str = "sqrt_shrink",
    inv_l2r_parts: Optional[List[Tensor]] = None, inv_r2l_parts: Optional[List[Tensor]] = None,
    final: bool = True,
):
    """Run one SyN level on ``layout.devices``.

    final=True: return (syn_forward, syn_inverse, history) with the composed fields on CPU.
    final=False: return (warp_l2r, warp_r2l, inv_l2r, inv_r2l, history) as full CPU fields (inverses None
    unless inverse_schedule='itk'), for upsampling to the next level.
    inverse_schedule='itk' re-inverts both half-warps twice every iteration exactly as the single-device
    path does (ITK SyNImageRegistrationMethod); step_scaling='ants' uses the learning rate in level voxels.

    ``I_by_dev`` / ``J_by_dev`` are emptied (the image copies are released before the inverses).
    """
    if similarity not in ("cc", "lncc", "cc2", "lncc2"):
        raise ValueError("distributed level supports the local CC metrics only")
    # take ownership of the incoming slabs: the caller's lists are emptied so the initial warps are
    # freed after the first update instead of staying alive for the whole level
    incoming_l2r, incoming_r2l = warp_l2r_parts, warp_r2l_parts
    warp_l2r_parts, warp_r2l_parts = list(incoming_l2r), list(incoming_r2l)
    incoming_l2r.clear()
    incoming_r2l.clear()
    del incoming_l2r, incoming_r2l
    inv_l2r = inv_r2l = None
    if inv_l2r_parts is not None:
        inv_l2r = list(inv_l2r_parts)
        inv_l2r_parts.clear()
    if inv_r2l_parts is not None:
        inv_r2l = list(inv_r2l_parts)
        inv_r2l_parts.clear()
    squared = similarity in ("cc2", "lncc2")
    devs = layout.devices
    n = len(devs)
    D = layout.D
    dtype = warp_l2r_parts[0].dtype
    spacing_itk = tuple(fixed_meta["spacing"])
    spacing_min = float(min(spacing_itk))
    if inverse_schedule not in ("end_of_level", "itk"):
        raise ValueError("distributed level supports inverse_schedule 'end_of_level' or 'itk'")
    cfl = grad_step if step_scaling == "ants" else grad_step * math.sqrt(float(shrink_factor))
    hm = 2 * (window_size // 2)
    hr = get_cached_gaussian_kernel_1d(flow_sigma, "cpu", torch.float32).shape[-1] // 2
    he = int(math.ceil(cfl)) + 2

    consts = {str(d): (metadata_tensors_from_dict(fixed_meta, d, dtype), metadata_tensors_from_dict(moving_meta, d, dtype),
                       M_phys.to(d), t_phys.to(d), torch.tensor(tuple(reversed(spacing_itk)), device=d, dtype=dtype))
              for d in devs}
    pool = ThreadPoolExecutor(max_workers=n)

    def par(fn):
        def run(k):
            d = devs[k]
            if d.type == "cuda":
                with torch.cuda.device(d):
                    return fn(k)
            return fn(k)
        return list(pool.map(run, range(n)))

    def subslabs(z0, z1, step):
        return [(s, min(z1, s + step)) for s in range(z0, z1, step)]

    # ---------------------------------------------------------------- metric + gradient on one device
    def metric_grad(k):
        d = devs[k]
        fx_t, mv_t, M, t, _ = consts[str(d)]
        z0, z1 = layout.cores[k]
        srcL = _RowSource(layout, k, warp_l2r_parts, hm)
        srcR = _RowSource(layout, k, warp_r2l_parts, hm)
        g_core = [torch.zeros_like(warp_l2r_parts[k]), torch.zeros_like(warp_r2l_parts[k])]
        shape_rest = warp_l2r_parts[k].shape[2:]
        top0 = max(0, z0 - hm)
        bot1 = min(D, z1 + hm)
        g_top = [torch.zeros((1, z0 - top0, *shape_rest), device=d, dtype=dtype) for _ in range(2)]
        g_bot = [torch.zeros((1, bot1 - z1, *shape_rest), device=d, dtype=dtype) for _ in range(2)]
        S = torch.zeros((), device=d, dtype=torch.float64)
        N = torch.zeros((), device=d, dtype=torch.float64)
        for s0, s1 in subslabs(z0, z1, metric_subslab_rows):
            Lr, f0, f1 = srcL.rows(s0 - hm, s1 + hm)
            Rr, _, _ = srcR.rows(s0 - hm, s1 + hm)
            left = Lr.detach().requires_grad_(True)
            right = Rr.detach().requires_grad_(True)
            X = _phys_rows(fixed_meta, f0, f1, d, dtype)
            a, b, _, _, valid = prepare_mid_images_and_gradients_torch(
                left, right, None, None, I_by_dev[str(d)], J_by_dev[str(d)], X,
                fx_t["shape_t"], fx_t["spacing_t"], fx_t["origin_t"], fx_t["direction_t"],
                mv_t["shape_t"], mv_t["spacing_t"], mv_t["origin_t"], mv_t["direction_t"],
                fixed_meta["spacing"], moving_meta["spacing"], M, t, None,
                use_analytical_gradients=use_analytical_gradients, compute_gradients=False)
            del X
            cc, active = _local_cc_terms(a, b, valid, window_size, squared)
            del a, b, valid
            c0, c1 = s0 - f0, s1 - f0
            s_sub = torch.sum(cc[:, :, c0:c1] * active[:, :, c0:c1])
            N += torch.sum(active[:, :, c0:c1]).detach().double()
            del cc, active
            s_sub.backward()
            S += s_sub.detach().double()
            del s_sub
            for i, g in enumerate((left.grad, right.grad)):
                for lo, hi, buf, base in ((f0, min(f1, z0), g_top[i], top0), (max(f0, z0), min(f1, z1), g_core[i], z0),
                                          (max(f0, z1), f1, g_bot[i], z1)):
                    if lo < hi:
                        buf[:, lo - base:hi - base] += g[:, lo - f0:hi - f0]
            del left, right, Lr, Rr
        return g_core, g_top, g_bot, S, N, top0, bot1

    # ------------------------------------------------------ fixed-point inversion across the slabs
    def invert(W_parts, init_parts=None, steps=None, criterion="both"):
        """ITK fixed-point inverse of W (slab parts); init_parts = initial estimate (default -W).

        criterion 'both' stops when both tolerances are met (the end-of-level solve of the single-device
        path), 'either' when one is (ITK InvertDisplacementFieldImageFilter, used every iteration)."""
        steps = end_of_level_inverse_steps if steps is None else steps
        H = int(math.ceil(_max_disp_voxels(W_parts, spacing_min))) + 2
        src = par(lambda k: _RowSource(layout, k, W_parts, H))
        v = [(-W_parts[k]) if init_parts is None else init_parts[k] for k in range(n)]
        counts = [int(np.prod(p.shape[1:-1])) for p in W_parts]
        max_err, mean_err = float("inf"), float("inf")
        for it in range(steps):
            ok_max, ok_mean = max_err <= max_error_threshold, mean_err <= mean_error_threshold
            if (ok_max or ok_mean) if criterion == "either" else (ok_max and ok_mean):
                break

            def error(k):                          # error = v + W(x + v) and its scaled norm, per sub-slab
                d = devs[k]
                z0, z1 = layout.cores[k]
                sp = consts[str(d)][4]
                err = torch.empty_like(v[k])
                norm = torch.empty(v[k].shape[:-1] + (1,), device=d, dtype=dtype)
                mx = torch.zeros((), device=d, dtype=dtype)
                total = torch.zeros((), device=d, dtype=torch.float64)
                for s0, s1 in subslabs(z0, z1, subslab_rows):
                    vs = v[k][:, s0 - z0:s1 - z0]
                    ext, e0, e1 = src[k].rows(s0 - H, s1 + H)
                    ext_t = metadata_tensors_from_dict(_rows_meta(fixed_meta, e0, e1), d, dtype)
                    e = _sample_rows(ext, ext_t, _phys_rows(fixed_meta, s0, s1, d, dtype) + vs).contiguous()
                    e.add_(vs)
                    scaled = e / sp
                    scaled.pow_(2)
                    nrm = torch.sqrt(torch.sum(scaled, dim=-1, keepdim=True))
                    err[:, s0 - z0:s1 - z0] = e
                    norm[:, s0 - z0:s1 - z0] = nrm
                    mx = torch.maximum(mx, nrm.max())
                    total += nrm.sum().double()
                    del e, scaled, nrm, ext
                return err, norm, float(mx), float(total)
            e = par(error)
            max_err = max(x[2] for x in e)
            mean_err = sum(x[3] for x in e) / sum(counts)
            eps = 0.75 if it == 0 else 0.5
            clip_threshold = eps * max_err

            def step(k):
                err, norm, _, _ = e[k]
                upd = err.neg_()
                clip = torch.where(norm > clip_threshold, clip_threshold / norm.clamp(min=1e-10), torch.ones_like(norm))
                upd.mul_(clip)
                del clip
                v_new = upd.mul_(eps).add_(v[k])
                z0, z1 = layout.cores[k]
                v_new.mul_(_boundary_rows(fixed_meta, z0, z1, devs[k], dtype))
                return v_new
            v = par(step)
            del e
        del src
        return v

    # -------------------------------------------------------------------------------- iterations
    from .convergence import WindowConvergenceMonitor
    monitor = WindowConvergenceMonitor(convergence_window) if convergence_threshold is not None else None
    history = []
    for iteration in range(iterations):
        out = par(metric_grad)
        S = sum(float(o[3]) for o in out)
        Nv = sum(float(o[4]) for o in out)
        loss_value = -S / (Nv + 1e-8)
        if not math.isfinite(loss_value):
            raise FloatingPointError("non-finite SyN loss at the distributed level")
        scale = -1.0 / (Nv + 1e-8)

        def accumulate(k):          # add every other device's halo strips that overlap this core, then scale
            d = devs[k]
            z0, z1 = layout.cores[k]
            g = out[k][0]
            for i in range(2):
                for j, o in enumerate(out):
                    if j == k:
                        continue
                    zj0, zj1 = layout.cores[j]
                    for strip, base in ((o[1][i], o[5]), (o[2][i], zj1)):   # rows [top0_j, z0_j) and [z1_j, bot1_j)
                        lo, hi = max(z0, base), min(z1, base + strip.shape[1])
                        if lo < hi:
                            g[i][:, lo - z0:hi - z0] += strip[:, lo - base:hi - base].to(d)
                g[i].mul_(scale)
            return g
        grads = par(accumulate)
        del out
        g_l = [g[0] for g in grads]
        g_r = [g[1] for g in grads]
        del grads

        # gaussian regularization: halo strips first (all devices), then filter sub-slab by sub-slab
        srcs = par(lambda k: (_RowSource(layout, k, g_l, hr), _RowSource(layout, k, g_r, hr)))

        def regularize(k):
            d = devs[k]
            z0, z1 = layout.cores[k]
            sp = consts[str(d)][4]
            res, mx = [], []
            for src in srcs[k]:
                out_core = torch.empty_like(src.core)
                m = torch.zeros((), device=d, dtype=dtype)
                for s0, s1 in subslabs(z0, z1, subslab_rows):
                    ext, e0, e1 = src.rows(s0 - hr, s1 + hr)
                    ext = ext * _boundary_rows(fixed_meta, e0, e1, d, dtype)
                    ext = separable_gaussian_filter(ext, flow_sigma, spacing=spacing_itk, sigma_mode="voxel")
                    piece = ext[:, s0 - e0:s1 - e0]
                    out_core[:, s0 - z0:s1 - z0] = piece
                    m = torch.maximum(m, torch.linalg.norm(piece / sp, dim=-1).max())
                    del ext, piece
                res.append(out_core)
                mx.append(m)
            return res, mx
        reg = par(regularize)
        del srcs, g_l, g_r
        max_l = max(max(float(r[1][0]) for r in reg), 1e-12)
        max_r = max(max(float(r[1][1]) for r in reg), 1e-12)

        # steps + antisymmetric projection + Eulerian update (old-warp halo strips gathered first)
        old = par(lambda k: (_RowSource(layout, k, warp_l2r_parts, he), _RowSource(layout, k, warp_r2l_parts, he)))

        def update(k):
            d = devs[k]
            z0, z1 = layout.cores[k]
            (rl, rr), _ = reg[k]
            srcL, srcR = old[k]
            new_l = torch.empty_like(warp_l2r_parts[k])
            new_r = torch.empty_like(warp_r2l_parts[k])
            for s0, s1 in subslabs(z0, z1, subslab_rows):
                dl = rl[:, s0 - z0:s1 - z0] * (cfl / max_l)
                dr = rr[:, s0 - z0:s1 - z0] * (cfl / max_r)
                if antisymmetric:
                    common = dl + dr
                    dl = dl - 0.5 * common
                    dr = dr - 0.5 * common
                    del common
                X = _phys_rows(fixed_meta, s0, s1, d, dtype)
                for src, delta, dst in ((srcL, dl, new_l), (srcR, dr, new_r)):
                    ext, e0, e1 = src.rows(s0 - he, s1 + he)
                    ext_t = metadata_tensors_from_dict(_rows_meta(fixed_meta, e0, e1), d, dtype)
                    dst[:, s0 - z0:s1 - z0] = _sample_rows(ext, ext_t, X - delta) - delta
                    del ext
                del X, dl, dr
            return new_l, new_r
        new = par(update)
        del reg, old
        warp_l2r_parts = [w[0] for w in new]
        warp_r2l_parts = [w[1] for w in new]
        del new
        if inverse_schedule == "itk":
            # T_inv = Invert(T, warm start T_inv); T = Invert(T_inv, warm start T), for both half-warps
            if inv_l2r is None:
                inv_l2r = [torch.zeros_like(w) for w in warp_l2r_parts]
            if inv_r2l is None:
                inv_r2l = [torch.zeros_like(w) for w in warp_r2l_parts]
            inv_l2r = invert(warp_l2r_parts, inv_l2r, steps=20, criterion="either")
            warp_l2r_parts = invert(inv_l2r, warp_l2r_parts, steps=20, criterion="either")
            inv_r2l = invert(warp_r2l_parts, inv_r2l, steps=20, criterion="either")
            warp_r2l_parts = invert(inv_r2l, warp_r2l_parts, steps=20, criterion="either")

        history.append(loss_value)
        if verbose:
            print(f"  [SyN distributed level] iteration {iteration + 1:04d}/{iterations}: loss={loss_value:.8g}")
        if monitor is not None and monitor.add(loss_value) < convergence_threshold:
            if verbose:
                print(f"  [SyN distributed level] converged after {iteration + 1} iterations")
            break

    # the image copies are not needed any more
    for key in list(I_by_dev):
        del I_by_dev[key]
    for key in list(J_by_dev):
        del J_by_dev[key]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    if not final:
        def to_cpu(parts):
            return None if parts is None else torch.cat([q.cpu() for q in parts], dim=1)
        out_fields = (to_cpu(warp_l2r_parts), to_cpu(warp_r2l_parts), to_cpu(inv_l2r), to_cpu(inv_r2l))
        pool.shutdown()
        return out_fields + (history,)

    if inverse_schedule == "itk":
        l2r_inv, r2l_inv = inv_l2r, inv_r2l          # maintained every iteration
    else:
        l2r_inv = invert(warp_l2r_parts)
        r2l_inv = invert(warp_r2l_parts)

    # ------------------------------------------------------- final half-warp swap: first + second(x + first)
    def compose(first_parts, second_parts):
        H = int(math.ceil(_max_disp_voxels(first_parts, spacing_min))) + 2
        src = par(lambda k: _RowSource(layout, k, second_parts, H))

        def one(k):
            d = devs[k]
            z0, z1 = layout.cores[k]
            out = torch.empty(first_parts[k].shape, dtype=dtype)
            for s0, s1 in subslabs(z0, z1, subslab_rows):
                f = first_parts[k][:, s0 - z0:s1 - z0]
                ext, e0, e1 = src[k].rows(s0 - H, s1 + H)
                ext_t = metadata_tensors_from_dict(_rows_meta(fixed_meta, e0, e1), d, dtype)
                out[:, s0 - z0:s1 - z0] = (f + _sample_rows(ext, ext_t, _phys_rows(fixed_meta, s0, s1, d, dtype) + f)).cpu()
                del ext
            return out
        result = torch.cat(par(one), dim=1)
        del src
        return result

    syn_forward = compose(l2r_inv, warp_r2l_parts)
    del l2r_inv
    syn_inverse = compose(r2l_inv, warp_l2r_parts)
    del r2l_inv
    pool.shutdown()
    return syn_forward, syn_inverse, history
