"""Run syn_registration from a given ANTsTorch source tree at 36 um and save the SyN warps.

python -I scripts/equiv_run.py <antstorch_src_dir> <out_prefix>
Run twice per tree to measure run-to-run (atomicAdd) noise, then compare with equiv_compare.py.
"""
import sys

src, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)

import ants  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import antstorch  # noqa: E402
from antstorch.syn import syn_registration  # noqa: E402

assert antstorch.__file__.startswith(src), antstorch.__file__
FIXED = "/home/maga/Desktop/R24_resub/E15-atlas03/a02_to_a03/E15_atlas03_r2_template_18um.nii.gz"
MOVING = "/home2/rachel/P01/MagaLab/AtlasBuilding/Atlas_02/Masked_NRRDs/Scan_0005__rec-18um-tx_masked.nrrd"
sp = 0.036
fixed = ants.resample_image(ants.image_read(FIXED), (sp,) * 3, False, 0).clone("float")
moving = ants.resample_image(ants.image_read(MOVING), (sp,) * 3, False, 0).clone("float")
torch.cuda.reset_peak_memory_stats()
reg = syn_registration(fixed, moving, type_of_transform="SyN", syn_metric="cc",
                       affine_smoothing_sigmas=(2 * sp, 1 * sp, 0.0),
                       outprefix=out + "_", device="cuda", dtype=torch.float32)
peak = torch.cuda.max_memory_allocated() / 2**30
w = reg["warpedmovout"].numpy()
f = fixed.numpy()
m = f > 0
print(f"{out}: peak {peak:.2f} GiB  corr {np.corrcoef(f[m], w[m])[0, 1]:.5f}  src {antstorch.__file__}")
