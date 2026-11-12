"""Slab-chunked local-CC value and half-warp gradients for the greedy SyN loop.

The local (normalized) cross-correlation used by ``local_ncc_loss_nd`` is a masked mean of a
per-voxel quantity ``cc(x)`` that depends on the midpoint images only within ``2 r`` voxels of
``x`` (``r = window_size // 2``: one box window for the local means, a second one for the
variances/covariance). So the metric and its backward pass can be evaluated slab by slab along
the first spatial axis, each slab extended by a ``2 r`` halo, without changing any per-voxel
``cc`` value. Only one slab's autograd intermediates are alive at a time, and slabs can be
placed on different devices.

loss = -S / (N + 1e-8),   S = sum_x cc(x) * active(x),   N = sum_x active(x)

``active`` (variance floors and the in-bounds mask) is piecewise constant, so autograd gives it
no gradient in the unchunked code either; therefore dloss = -dS / (N + 1e-8), and the slab
gradients dS_c are accumulated and scaled once at the end. Results match the unchunked
objective up to floating-point summation order.
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .grid import prepare_mid_images_and_gradients_torch


def _local_cc_terms(I: Tensor, J: Tensor, mask: Tensor, window_size: int, squared: bool):
    """Per-voxel ``cc`` and ``active`` maps; the same operations as ``local_ncc_loss_nd``."""
    pad = window_size // 2
    pool_fn = F.avg_pool3d if I.ndim == 5 else F.avg_pool2d

    def box_filter(x):
        return pool_fn(x, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)

    I_mean = box_filter(I)
    J_mean = box_filter(J)
    I_var = torch.clamp(box_filter((I - I_mean) ** 2), min=0.0)
    J_var = torch.clamp(box_filter((J - J_mean) ** 2), min=0.0)
    IJ_cov = box_filter((I - I_mean) * (J - J_mean))
    var_floor = 1e-6
    safe_I_var = torch.clamp(I_var, min=var_floor)
    safe_J_var = torch.clamp(J_var, min=var_floor)
    if squared:
        cc = torch.clamp((IJ_cov ** 2) / (safe_I_var * safe_J_var + 1e-8), min=0.0, max=1.0)
    else:
        cc = torch.clamp(IJ_cov / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6), min=-1.0, max=1.0)
    active = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(dtype=I.dtype)
    return cc, active


class ChunkedLocalCC:
    """Callable computing ``(loss_value, grad_l2r, grad_r2l)`` for the plain SyN objective.

    Parameters
    ----------
    I_curr, J_curr : fixed/moving level images ``(1, 1, *spatial)`` on the main device.
    X_phys : fixed-grid physical coordinates ``(1, *spatial, dim)``.
    fixed_meta_t, moving_meta_t : metadata tensor dicts (main device).
    fixed_spacing, moving_spacing, M_phys, t_phys : as passed to
        ``prepare_mid_images_and_gradients_torch``.
    window_size : CC window (``2 r + 1``).
    squared : True for cc2/lncc2.
    n_chunks : number of slabs along the first spatial axis.
    devices : optional list of devices; slab ``k`` runs on ``devices[k % len(devices)]``.
        Each extra device holds its own copy of the two level images.
    """

    def __init__(self, I_curr, J_curr, X_phys, fixed_meta_t, moving_meta_t, fixed_spacing, moving_spacing,
                 M_phys, t_phys, window_size: int, squared: bool, n_chunks: int,
                 devices: Optional[Sequence] = None, use_analytical_gradients: bool = True):
        self.main = I_curr.device
        self.window_size = int(window_size)
        self.halo = 2 * (self.window_size // 2)
        self.squared = squared
        self.use_analytical_gradients = use_analytical_gradients
        self.fixed_spacing, self.moving_spacing = fixed_spacing, moving_spacing
        D = I_curr.shape[2]
        n_chunks = max(1, min(int(n_chunks), D))
        bounds = [round(i * D / n_chunks) for i in range(n_chunks + 1)]
        self.slabs = [(bounds[i], bounds[i + 1]) for i in range(n_chunks) if bounds[i + 1] > bounds[i]]
        self.D = D
        devices = [self.main] if not devices else [torch.device(d) for d in devices]
        self.devices = devices
        # per-device constant state: images, metadata, affine (the main device reuses the originals)
        self.state = {}
        for dev in devices:
            key = str(dev)
            if key in self.state:
                continue
            same = torch.device(dev) == self.main
            mv = {k: (v if same else v.to(dev)) for k, v in moving_meta_t.items()}
            fx = {k: (v if same else v.to(dev)) for k, v in fixed_meta_t.items()}
            self.state[key] = dict(
                I=I_curr if same else I_curr.to(dev), J=J_curr if same else J_curr.to(dev),
                fx=fx, mv=mv, M=M_phys if same else M_phys.to(dev), t=t_phys if same else t_phys.to(dev),
            )
        self.X_phys = X_phys

    def __call__(self, warp_l2r: Tensor, warp_r2l: Tensor):
        grad_l = torch.zeros_like(warp_l2r)
        grad_r = torch.zeros_like(warp_r2l)
        S_parts, N_parts = [], []
        for k, (z0, z1) in enumerate(self.slabs):
            dev = self.devices[k % len(self.devices)]
            st = self.state[str(dev)]
            e0, e1 = max(0, z0 - self.halo), min(self.D, z1 + self.halo)
            c0, c1 = z0 - e0, z1 - e0  # core rows inside the extended slab
            left = warp_l2r[:, e0:e1].to(dev).requires_grad_(True)
            right = warp_r2l[:, e0:e1].to(dev).requires_grad_(True)
            X = self.X_phys[:, e0:e1].to(dev)
            fx, mv = st["fx"], st["mv"]
            a, b, _, _, valid = prepare_mid_images_and_gradients_torch(
                left, right, None, None, st["I"], st["J"], X,
                fx["shape_t"], fx["spacing_t"], fx["origin_t"], fx["direction_t"],
                mv["shape_t"], mv["spacing_t"], mv["origin_t"], mv["direction_t"],
                self.fixed_spacing, self.moving_spacing, st["M"], st["t"], None,
                use_analytical_gradients=self.use_analytical_gradients, compute_gradients=False,
            )
            del X
            cc, active = _local_cc_terms(a, b, valid, self.window_size, self.squared)
            del a, b, valid
            cc_core = cc[:, :, c0:c1]
            active_core = active[:, :, c0:c1]
            S = torch.sum(cc_core * active_core)
            N_parts.append(torch.sum(active_core).detach().to(self.main))
            del cc, active, cc_core, active_core
            S.backward()
            S_parts.append(S.detach().to(self.main))
            del S
            grad_l[:, e0:e1] += left.grad.to(self.main)
            grad_r[:, e0:e1] += right.grad.to(self.main)
            del left, right
        S_total = torch.stack(S_parts).sum()
        N_total = torch.stack(N_parts).sum()
        scale = -1.0 / (N_total + 1e-8)
        grad_l.mul_(scale)
        grad_r.mul_(scale)
        loss_value = float((-S_total / (N_total + 1e-8)).item())
        return loss_value, grad_l, grad_r
