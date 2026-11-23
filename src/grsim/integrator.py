"""
Numerical geodesic integration.

The geodesic equation

    d²x^μ/dτ² = -Γ^μ_aβ (dx^a/dτ) (dx^β/dτ)

is a second-order ODE. Writing the 4-velocity u^μ = dx^μ/dτ as separate
unknowns turns it into a first-order system for the 8-component state

    y = (x^0, x^1, x^2, x^3, u^0, u^1, u^2, u^3),

which is then stepped forward in proper time τ with the classical
4th-order Runge-Kutta method (RK4).

Functions:
    geodesic_rhs  - dy/dτ for the geodesic equation in a given metric
    rk4_step      - one RK4 step of any ODE dy/dτ = f(τ, y, metric)
    integrate     - many RK4 steps with a fixed step size, returning the whole path
"""
import numpy as np

from grsim import metric

def geodesic_rhs(tau, y, metric):
    """
    Right-hand side of the geodesic equation, written as a first-order system.

    What happens:
        The state y is split into position x^μ = y[:4] and 4-velocity
        u^μ = y[4:]. The Christoffel symbols Γ^μ_aβ are evaluated at x, and
        the derivatives are
            dx^μ/dτ = u^μ
            du^μ/dτ = -Γ^μ_aβ u^a u^β
        The geodesic equation does not depend on τ explicitly, so tau is only
        accepted to match the f(tau, y, metric) signature used by rk4_step.

    Parameters:
        tau (float): Proper time (unused, see above).
        y (np.ndarray): State of length 8: (x^0..x^3, u^0..u^3), in the
                        metric's own coordinates, e.g. (t, r, θ, φ, u^t, u^r, u^θ, u^φ).
        metric (grsim.Metric): Spacetime to move in, e.g. Schwarzschild(M=1.0).
                               Its christoffel(x) must return an array of shape (4, 4, 4).

    Returns:
        np.ndarray: dy/dτ, length 8: (u^0..u^3, du^0/dτ..du^3/dτ).
    """
    x = y[:4]
    u = y[4:]
    Gamma = metric.christoffel(x)
    du = -np.einsum("lmn,m,n->l", Gamma, u, u)
    return np.concatenate([u, du])

def rk4_step(f, tau, y, h, metric):
    """
    Advances the solution of dy/dτ = f(τ, y, metric) by one classical RK4 step.

    What happens:
        Four slopes are evaluated: at the start (k1), twice at the midpoint
        τ + h/2 (k2, k3) and at the end τ + h (k4). Their weighted average
        (k1 + 2 k2 + 2 k3 + k4) / 6 is used to step y forward. The error per
        step is O(h⁵), so the error after a fixed time interval is O(h⁴):
        halving h makes it about 16 times smaller.

    Parameters:
        f (callable): Right-hand side, called as f(tau, y, metric) and returning
                      an array with the same shape as y (e.g. geodesic_rhs).
        tau (float): Current value of the independent variable (proper time).
        y (np.ndarray): Current state.
        h (float): Step size in τ. Negative h integrates backwards.
        metric: Passed through to f unchanged (may be None if f ignores it).

    Returns:
        np.ndarray: The state at τ + h.
    """
    k1 = f(tau, y, metric)
    k2 = f(tau+h/2, y+h/2*k1, metric)
    k3 = f(tau+h/2, y+h/2*k2, metric)
    k4 = f(tau+h, y+h*k3, metric)
    return y + h/6*(k1+2*k2+2*k3+k4)

def integrate(f, y0, tau0, tau_end, h, metric):
    """
    Integrates dy/dτ = f(τ, y, metric) from tau0 to tau_end with a fixed step size.

    What happens:
        The number of steps is round((tau_end - tau0) / h), so tau_end is
        reached exactly only if the interval is a whole number of steps h.
        Starting from y0, rk4_step is applied repeatedly and every
        intermediate state is stored.

    Parameters:
        f (callable): Right-hand side, f(tau, y, metric) -> dy/dτ (e.g. geodesic_rhs).
        y0 (array_like): Initial state at tau0 (length 8 for a geodesic).
        tau0 (float): Initial proper time.
        tau_end (float): Final proper time.
        h (float): Step size in τ.
        metric: Passed through to f (e.g. a grsim.Metric instance).

    Returns:
        tuple:
            taus (np.ndarray): Shape (N,), the proper times τ_0 .. τ_{N-1},
                               with N = number of steps + 1.
            ys (np.ndarray): Shape (N, len(y0)); ys[i] is the state at taus[i].
                             For a geodesic, ys[:, 1] is the r-coordinate over time, etc.
    """
    n_steps = int(round((tau_end - tau0) / h))
    taus = tau0 + h * np.arange(n_steps + 1)
    ys = np.zeros((n_steps + 1, len(y0)))
    ys[0] = y0

    for i in range(n_steps):
        ys[i + 1] = rk4_step(f, taus[i], ys[i], h, metric)

    return taus, ys

if __name__ == "__main__":
    y = np.array([0.0, 6.0, np.pi/2, 0.0,   1.2, 0.0, 0.0, 0.05])
    print(geodesic_rhs(0.0, y, metric.Schwarzschild(M=1.0)).shape)
    print(geodesic_rhs(0.0, y, metric.Minkowski()))
