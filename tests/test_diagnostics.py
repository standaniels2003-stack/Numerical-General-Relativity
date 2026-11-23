"""Geodesic diagnostics: normalization g(u,u) = κ, energy E and angular momentum L_z stay conserved."""
import numpy as np
import pytest

from grsim import Kerr, Schwarzschild, diagnostics
from grsim.integrator import geodesic_rhs, integrate

# Every test in this file is a diagnostic: run them with `pytest -m diagnostics`.
pytestmark = pytest.mark.diagnostics


def eccentric_orbit_start(bh, r0=12.0, factor=0.95):
    """
    Helper (not a test): starting state of a bound, eccentric orbit in Schwarzschild.

    Input:   the Schwarzschild metric bh, the starting radius r0 and a factor
             that scales u^φ relative to the circular-orbit value.
    Returns: the state (t, r, θ, φ, u^t, u^r, u^θ, u^φ) at the outer turning
             point r0 in the equatorial plane. With factor = 0.95 the particle
             swings between r ≈ 8.8 and 12. u^t is chosen so that g(u, u) = -1.
    """
    uphi = factor * np.sqrt(1 / r0**3) / np.sqrt(1 - 3 / r0)
    g = bh.g((0.0, r0, np.pi / 2, 0.0))
    ut = np.sqrt((1 + g[3, 3] * uphi**2) / -g[0, 0])
    return np.array([0.0, r0, np.pi / 2, 0.0,   ut, 0.0, 0.0, uphi])


def test_circular_orbit_conserved_quantities():
    """
    Schwarzschild (M = 1): along 3 circular orbits at r = 10, g(u,u) = -1 and E, L_z stay constant to 1e-8

    Input:   the circular orbit at r0 = 10M in the equatorial plane, with
             u^t = 1/sqrt(1 - 3M/r0) and u^φ = sqrt(M/r0³)/sqrt(1 - 3M/r0),
             integrated for three orbital periods (in proper time) with 1000 RK4 steps.
    Checks:  conserved_quantities() returns arrays with one value per step, the
             norm stays at -1 to 1e-8, E and L_z change by less than 1e-8 relative,
             and they keep their exact values (1 - 2M/r0) u^t and r0² u^φ.
    Why:     the simplest orbit where everything is known in closed form. In
             practice the drift is exactly 0 here: the gravitational and centrifugal
             terms in du^r/dτ cancel exactly, so r never changes. A wrong
             Christoffel symbol or sign would break that balance immediately. The
             eccentric-orbit diagnostic below is the more demanding accuracy check.
    """
    M, r0 = 1.0, 10.0
    bh = Schwarzschild(M=M)
    ut = 1 / np.sqrt(1 - 3 * M / r0)
    uphi = np.sqrt(M / r0**3) / np.sqrt(1 - 3 * M / r0)
    y0 = np.array([0.0, r0, np.pi / 2, 0.0,   ut, 0.0, 0.0, uphi])
    T = 2 * np.pi / uphi                               # orbital period in proper time

    taus, ys = integrate(geodesic_rhs, y0, 0.0, 3 * T, h=3 * T / 1000, metric=bh)
    norm, E, L = diagnostics.conserved_quantities(bh, ys)

    assert norm.shape == E.shape == L.shape == (len(ys),)
    assert np.max(np.abs(norm + 1)) < 1e-8
    assert diagnostics.max_relative_drift(E) < 1e-8
    assert diagnostics.max_relative_drift(L) < 1e-8
    assert np.allclose(E, (1 - 2 * M / r0) * ut)
    assert np.allclose(L, r0**2 * uphi)


def test_timelike_orbit_conserves_normalization_energy_and_angular_momentum():
    """
    Schwarzschild: eccentric orbit (r ≈ 8.8-12) keeps g(u,u) = -1, E and L_z to 1e-8

    Input:   a massive particle on a bound eccentric orbit (see eccentric_orbit_start),
             integrated for τ = 2000 (about 6 radial oscillations) with h = 1.
    Checks:  |g_μv u^μ u^v + 1| < 1e-8 at every step, and E and L_z change by less
             than 1e-8 relative to their starting values (measured: about 1e-10).
    Why:     along an exact geodesic all three are conserved, so their drift is a
             direct measure of the integration error on a non-trivial orbit.
    """
    bh = Schwarzschild(M=1.0)
    taus, ys = integrate(geodesic_rhs, eccentric_orbit_start(bh), 0.0, 2000.0, h=1.0, metric=bh)
    assert np.max(np.abs(diagnostics.along_path(bh.norm, ys) + 1)) < 1e-8
    assert diagnostics.max_relative_drift(diagnostics.along_path(bh.energy, ys)) < 1e-8
    assert diagnostics.max_relative_drift(diagnostics.along_path(bh.angular_momentum, ys)) < 1e-8


def test_energy_error_converges_at_fourth_order():
    """
    Schwarzschild: on the eccentric orbit, log(energy error) against log(h) has slope 4 ± 0.3

    Input:   the same eccentric orbit, integrated for 3.5 radial periods (so it ends
             at pericentre) with h ≈ 2, 1, 0.5, 0.25, each a whole number of steps.
             The radial period T_r comes from the exact integral (not the integrator):
             (dr/dτ)² = (1 - E²)(r_a - r)(r - r_p)(r - r_3)/r³, and with
             r = (r_a + r_p)/2 + (r_a - r_p)/2 cos χ,
             T_r = 2 ∫_0^π sqrt(r³/((1 - E²)(r - r_3))) dχ ≈ 274.3.
    Checks:  the straight-line fit np.polyfit(log h, log |E_end/E_0 - 1|, 1) has a
             slope within 0.3 of 4 (measured: about 4.1).
    Why:     RK4 is fourth order, so the error must shrink like h⁴; a lower slope
             means a bug in the geodesic RHS or the integrator. The run ends at
             pericentre because after whole radial periods the h⁴ part of the
             energy error cancels (slope ≈ 5), and at a random end time it can sit
             near a zero crossing. scripts/convergence.py makes the plot of this check.
    """
    M = 1.0
    bh = Schwarzschild(M=M)
    y0 = eccentric_orbit_start(bh)
    E0, L0 = bh.energy(y0[:4], y0[4:]), bh.angular_momentum(y0[:4], y0[4:])

    r3, rp, ra = np.sort(np.roots([1 - E0**2, -2 * M, L0**2, -2 * M * L0**2]).real)
    chi = np.linspace(0.0, np.pi, 20001)
    r = (ra + rp) / 2 + (ra - rp) / 2 * np.cos(chi)
    T_r = 2 * np.trapezoid(np.sqrt(r**3 / ((1 - E0**2) * (r - r3))), chi)
    tau_end = 3.5 * T_r

    n0 = int(np.ceil(tau_end / 2.0))
    hs = np.array([tau_end / (n0 * 2**k) for k in range(4)])
    errors = []
    for h in hs:
        taus, ys = integrate(geodesic_rhs, y0, 0.0, tau_end, h=h, metric=bh)
        errors.append(abs(bh.energy(ys[-1, :4], ys[-1, 4:]) / E0 - 1))
    slope = np.polyfit(np.log(hs), np.log(errors), 1)[0]
    assert abs(slope - 4) < 0.3, f"slope {slope:.2f}"


def test_null_geodesic_stays_null():
    """
    Schwarzschild: a light ray passing at closest distance r ≈ 6.7 keeps g(u,u) = 0, E and L_z to 1e-6

    Input:   a photon starting at r = 30 moving inward with energy E = 1 and
             impact parameter b = L/E = 8 (larger than the capture value 3√3 M ≈ 5.2,
             so it is deflected rather than captured), integrated with h = 0.25
             over affine parameter λ = 60.
    Checks:  |g_μv u^μ u^v| < 1e-6 at every step (κ = 0 for light), and E and L_z
             change by less than 1e-6 relative (measured: about 6e-8 and 3e-8).
    Why:     light rays are what a ray tracer follows; they must stay exactly on
             the light cone, and they pass much closer to the black hole than the
             massive orbit, where the curvature is stronger.
    """
    bh = Schwarzschild(M=1.0)
    r0, E, b = 30.0, 1.0, 8.0
    f = 1 - 2 / r0
    L = b * E
    ur = -np.sqrt(E**2 - f * L**2 / r0**2)            # inward, from g(u,u) = 0
    y0 = np.array([0.0, r0, np.pi / 2, 0.0,   E / f, ur, 0.0, L / r0**2])
    taus, ys = integrate(geodesic_rhs, y0, 0.0, 60.0, h=0.25, metric=bh)
    assert ys[:, 1].min() > 3.0                        # it really was deflected, not captured
    assert np.max(np.abs(diagnostics.along_path(bh.norm, ys))) < 1e-6
    assert diagnostics.max_relative_drift(diagnostics.along_path(bh.energy, ys)) < 1e-6
    assert diagnostics.max_relative_drift(diagnostics.along_path(bh.angular_momentum, ys)) < 1e-6


@pytest.mark.slow
def test_kerr_inclined_orbit_conserves_energy_and_angular_momentum():
    """
    Kerr (M = 1, a = 0.9): an inclined, eccentric orbit keeps g(u,u) = -1, E and L_z to 1e-5

    Input:   a massive particle starting at r = 10, θ = π/2 - 0.3 (above the equator)
             with u^θ = 0.01 and u^φ = 0.03, u^t solved from g(u, u) = -1, integrated
             for τ = 1000 with h = 0.5. It moves between r ≈ 4.3 and 10 and swings
             through the equatorial plane.
    Checks:  |g_μv u^μ u^v + 1| < 1e-5 at every step, and E and L_z change by less
             than 1e-5 relative (measured: about 1e-7, 2e-8 and 4e-7).
    Why:     Kerr has the off-diagonal g_tφ term (frame dragging), so E and L_z mix
             t- and φ-components. A sign or index error in g_tφ or in its
             Christoffel symbols would break their conservation.
    """
    kerr = Kerr(M=1.0, a=0.9)
    r0, th0, uth, uphi = 10.0, np.pi / 2 - 0.3, 0.01, 0.03
    g = kerr.g((0.0, r0, th0, 0.0))
    # g_tt ut² + 2 g_tφ ut uφ + (g_φφ uφ² + g_θθ uθ² + 1) = 0, future-directed root (ut > 0)
    A, B, C = g[0, 0], 2 * g[0, 3] * uphi, g[3, 3] * uphi**2 + g[2, 2] * uth**2 + 1
    ut = (-B - np.sqrt(B * B - 4 * A * C)) / (2 * A)
    y0 = np.array([0.0, r0, th0, 0.0,   ut, 0.0, uth, uphi])

    taus, ys = integrate(geodesic_rhs, y0, 0.0, 1000.0, h=0.5, metric=kerr)
    assert np.max(np.abs(diagnostics.along_path(kerr.norm, ys) + 1)) < 1e-5
    assert diagnostics.max_relative_drift(diagnostics.along_path(kerr.energy, ys)) < 1e-5
    assert diagnostics.max_relative_drift(diagnostics.along_path(kerr.angular_momentum, ys)) < 1e-5
