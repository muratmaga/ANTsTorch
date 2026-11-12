"""ITK WindowConvergenceMonitoringFunction (as used by antsRegistration's SyN), re-implemented.

Per level: energies E_1..E_k; total = sum |E_i| over the whole level; the last `window` values are
normalized by total, placed at t = n/(window-1), and fitted with ITK's
BSplineScatteredDataPointSetToImageFilter (spline order 1, 2 control points, 1 level: the
multilevel B-spline approximation of Lee et al.); the convergence value is minus the slope at t=1.
"""
from collections import deque


def _fit_slope(values):
    n = len(values)
    num = [0.0, 0.0]
    den = [0.0, 0.0]
    for i, z in enumerate(values):
        t = i / (n - 1)
        w = (1.0 - t, t)                      # linear B-spline basis on one span
        s = w[0] ** 2 + w[1] ** 2
        for k in range(2):
            delta = w[k] * z / s              # per-point control-point estimate
            num[k] += w[k] ** 2 * delta
            den[k] += w[k] ** 2
    phi = [num[k] / den[k] if den[k] else 0.0 for k in range(2)]
    return phi[1] - phi[0]                    # derivative of the linear curve (domain [0, 1])


class WindowConvergence:
    def __init__(self, window=10):
        self.window = window
        self.values = deque(maxlen=window)
        self.total = 0.0

    def add(self, energy):
        self.values.append(float(energy))
        self.total += abs(float(energy))
        if len(self.values) < self.window:
            return float("inf")
        return -_fit_slope([v / self.total for v in self.values])
