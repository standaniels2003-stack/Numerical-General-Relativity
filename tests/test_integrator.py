"""RK4 integrator: accuracy, 4th-order convergence and time stepping on equations with known solutions."""
from grsim import integrator, metric
import numpy as np


def integrate(f, y0, h, tau_end=1.0):
    """
    Helper (not a test, its name doesn't start with test_): take RK4 steps up to tau_end.

    Input:   right-hand side f(tau, y, metric), initial state y0, step size h, end time tau_end.
    Returns: the state y at tau_end (only the final state, not the whole path),
             after round(tau_end / h) steps of integrator.rk4_step starting at τ = 0.
    """
    y = np.array(y0, dtype=float)
    for i in range(round(tau_end / h)):
        y = integrator.rk4_step(f, i * h, y, h, None)
    return y


def test_rk4_exponential_decay():
    """
    RK4 on dy/dτ = -y: matches e^−τ at τ = 1 to within 1e-6

    Input:   dy/dτ = -y with y(0) = 1, integrated to τ = 1 with h = 0.1 (10 steps).
    Checks:  the result matches the exact solution e^-1 to within 1e-6
             (the actual error is about 3.3e-7).
    Why:     the most basic accuracy check, on an equation with a known solution.
    """
    y = integrate(lambda tau, y, m: -y, [1.0], h=0.1)
    assert abs(y[0] - np.exp(-1.0)) < 1e-6


def test_rk4_is_fourth_order():
    """
    RK4 on dy/dτ = -y: halving h shrinks the error about 16 times (4th order)

    Input:   the same decay equation, integrated to τ = 1 with h = 0.1 and with h = 0.05.
    Checks:  the error ratio between the two runs is between 14 and 18.
    Why:     RK4 is a 4th-order method, so the error scales as h⁴ and halving h
             divides it by 2⁴ = 16. A mistake in the RK4 weights or stages would
             usually drop the order and give a ratio near 2, 4 or 8 instead.
    """
    exact = np.exp(-1.0)
    error_big = abs(integrate(lambda tau, y, m: -y, [1.0], h=0.1)[0] - exact)
    error_small = abs(integrate(lambda tau, y, m: -y, [1.0], h=0.05)[0] - exact)
    assert 14 < error_big / error_small < 18


def test_rk4_uses_time_correctly():
    """
    RK4 on dy/dτ = τ: exact answer τ²/2, so this catches wrong time arguments

    Input:   dy/dτ = τ with y(0) = 0, integrated to τ = 1 with h = 0.1.
    Checks:  the result equals the exact τ²/2 = 0.5 to within 1e-12.
    Why:     RK4 solves this exactly (up to rounding), but only if the stages are
             evaluated at τ, τ + h/2, τ + h/2 and τ + h. The decay equation ignores
             τ, so only a τ-dependent equation like this one catches wrong times.
    """
    y = integrate(lambda tau, y, m: np.array([tau]), [0.0], h=0.1)
    assert abs(y[0] - 0.5) < 1e-12

def test_geodesic_rhs_flat_space():
    """
    Minkowski: geodesic_rhs gives 8 components, dx/dτ = u and du/dτ = 0

    Input:   state y = (t, x, y, z, u^t, u^x, u^y, u^z) = (0, 1, 2, 3, 1.2, 0.1, -0.3, 0.05)
             in flat spacetime.
    Checks:  the output has shape (8,), its first half equals the velocity u and its
             second half (the acceleration) is zero.
    Why:     with all Γ = 0 a free particle has no acceleration, so this checks the
             layout of the state vector without any physics getting in the way.
    """
    y = np.array([0.0, 1.0, 2.0, 3.0,   1.2, 0.1, -0.3, 0.05])
    dy = integrator.geodesic_rhs(0.0, y, metric.Minkowski())
    assert dy.shape == (8,)
    assert np.allclose(dy[:4], y[4:])   # dx/dτ = u
    assert np.allclose(dy[4:], 0.0) 

def test_schwarzschild_circular_orbit_stays_circular():
    """
    Schwarzschild (M = 1): a circular orbit at r = 10 stays at r = 10 for 3 orbits

    Input:   a particle in the equatorial plane (θ = π/2) at r0 = 10M with the exact
             circular-orbit 4-velocity
                 u^t = 1 / sqrt(1 - 3M/r0),   u^φ = sqrt(M/r0³) / sqrt(1 - 3M/r0),
             integrated for three orbital periods (in proper time) with 1000 RK4 steps.
    Checks:  r never drifts more than 1e-6 from r0, and φ ends at 3 · 2π (to a relative 1e-6).
    Why:     a full test of geodesic_rhs + integrate on a curved spacetime with a
             known exact solution. Errors in Γ, the RHS or the integrator would make
             the orbit spiral in or out.
    """
    M, r0 = 1.0, 10.0
    ut   = 1 / np.sqrt(1 - 3*M/r0)
    uphi = np.sqrt(M / r0**3) / np.sqrt(1 - 3*M/r0)
    y0 = np.array([0.0, r0, np.pi/2, 0.0,   ut, 0.0, 0.0, uphi])

    T = 2*np.pi / uphi
    n_steps = 1000
    h = 3*T / n_steps

    taus, ys = integrator.integrate(integrator.geodesic_rhs, y0, 0.0, 3*T, h=h, metric=metric.Schwarzschild(M=M))

    assert np.max(np.abs(ys[:, 1] - r0)) < 1e-6          # r stays at r0
    assert np.isclose(ys[-1, 3], 3 * 2*np.pi, rtol=1e-6)  # three full orbits