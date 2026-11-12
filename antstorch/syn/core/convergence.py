"""antsRegistration's per-level convergence test, for the greedy SyN loop.

ITK's ``WindowConvergenceMonitoringFunction`` (what ``-c [iterations,threshold,window]`` uses in
SyN): for each level, keep the last ``window`` energies and the running sum of their absolute
values over the whole level; divide the window by that sum, place the points at
``t = n / (window - 1)``, fit a first-order B-spline with ``BSplineScatteredDataPointSetToImageFilter``
(2 control points, 1 level, domain [0, 1] sampled at 11 points), and report minus the slope at
``t = 1``. The level stops when that value drops below the threshold. Until ``window`` energies
exist the value is +inf.

The fit is ITK's own filter (via ANTsPy), so the convergence values match antsRegistration's
DIAGNOSTIC output for the same energy sequence (checked on a round-9 E18 log: agreement to
~1e-4 relative, the precision of the log).
"""

from collections import deque

import numpy as np


class WindowConvergenceMonitor:
    def __init__(self, window: int = 10):
        if window < 2:
            raise ValueError("convergence window must be at least 2")
        self.window = int(window)
        self.values = deque(maxlen=self.window)
        self.total = 0.0
        self._t = (np.arange(self.window, dtype=np.float64) / (self.window - 1)).reshape(-1, 1)

    def add(self, energy: float) -> float:
        """Add one iteration's energy; return the current convergence value."""
        self.values.append(float(energy))
        self.total += abs(float(energy))
        if len(self.values) < self.window or self.total == 0.0:
            return float("inf")
        import ants

        z = (np.asarray(self.values, dtype=np.float64) / self.total).reshape(-1, 1)
        curve = ants.fit_bspline_object_to_scattered_data(
            z, self._t, parametric_domain_origin=[0.0], parametric_domain_spacing=[0.1],
            parametric_domain_size=[11], number_of_fitting_levels=1, mesh_size=1, spline_order=1)
        v = np.asarray(curve.numpy() if hasattr(curve, "numpy") else curve, dtype=np.float64).ravel()
        return -float(v[-1] - v[0])       # linear curve on [0, 1]: slope = end - start
