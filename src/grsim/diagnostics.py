"""
Diagnostics for geodesics: follow a quantity along an integrated path.

The conserved quantities themselves are methods of the metric classes:

    metric.norm(x, u)              g_μv u^μ u^v   (-1 for a massive particle, 0 for light)
    metric.energy(x, u)            E   (Killing vector ∂_t)
    metric.angular_momentum(x, u)  L_z (Killing vector ∂_φ)

Along an exact geodesic all three are constant, so their change during a
numerical integration measures the integration error. This module evaluates
such a quantity at every step of a path and measures how much it drifts.

The path ys is what integrator.integrate returns: an array of shape (N, 8)
whose rows are states (x^0..x^3, u^0..u^3).

Example:
    taus, ys = integrate(geodesic_rhs, y0, 0.0, 1000.0, h=1.0, metric=bh)
    E = along_path(bh.energy, ys)                 # E at every step
    print(max_relative_drift(E))                  # e.g. 1e-10
    print(np.max(np.abs(along_path(bh.norm, ys) + 1)))   # normalization drift
"""
import numpy as np


def along_path(quantity, ys):
    """
    Evaluates a quantity of position and 4-velocity at every step of a path.

    What happens:
        Each state y in ys is split into x = y[:4] and u = y[4:], and
        quantity(x, u) is called for it.

    Parameters:
        quantity (callable): A function quantity(x, u), e.g. metric.energy,
                             metric.angular_momentum or metric.norm.
        ys (np.ndarray): Path of shape (N, 8), or one state of length 8.

    Returns:
        np.ndarray: Length N, quantity(x, u) at each step; plot it against taus
                    from integrate to see how it evolves.
    """
    ys = np.atleast_2d(ys)
    return np.array([quantity(y[:4], y[4:]) for y in ys])


def conserved_quantities(metric, ys):
    """
    The norm g(u,u), energy E and angular momentum L_z at every step of a trajectory.

    What happens:
        along_path is applied with metric.norm, metric.energy and
        metric.angular_momentum. Along an exact geodesic all three arrays are
        constant, so how much they vary measures the integration error.

    Parameters:
        metric (grsim.Metric): The spacetime the trajectory was integrated in;
                               must define energy() and angular_momentum().
        ys (np.ndarray): Trajectory of shape (N, 8) as returned by
                         integrator.integrate, rows (x^0..x^3, u^0..u^3).

    Returns:
        tuple of three np.ndarray, each of length N:
            norm: g_μv u^μ u^v at each step (-1 for a massive particle, 0 for light)
            E:    energy at each step
            L:    angular momentum L_z at each step

    Example:
        norm, E, L = conserved_quantities(bh, ys)
        print(np.max(np.abs(norm + 1)), max_relative_drift(E), max_relative_drift(L))
    """
    return (along_path(metric.norm, ys),
            along_path(metric.energy, ys),
            along_path(metric.angular_momentum, ys))


def max_relative_drift(values):
    """
    The largest relative change of a quantity from its starting value.

    Parameters:
        values (array_like): A quantity at every step, e.g. along_path(metric.energy, ys).
                             Its first value must not be 0 (so not for the
                             normalization of light; use the absolute drift there).

    Returns:
        float: max_i |values[i] / values[0] - 1|, e.g. 1e-10 means the quantity
               changed by at most one part in 10^10.
    """
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values / values[0] - 1)))
