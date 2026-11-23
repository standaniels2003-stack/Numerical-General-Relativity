"""Metric classes: numerical evaluation of g, g⁻¹ and Γ for known spacetimes."""
import numpy as np
import pytest

from grsim import Kerr, Minkowski, MinkowskiSpherical, Schwarzschild

X = (0.0, 6.0, 1.2, 0.0)


def test_minkowski_christoffels_vanish():
    """
    Minkowski: all 64 numerically evaluated Γ vanish at a sample point

    Input:   Minkowski(), evaluated at X = (t, x, y, z) = (0, 6, 1.2, 0).
    Checks:  christoffel(X) returns a (4, 4, 4) array that is zero everywhere.
    Why:     checks the symbolic -> NumPy conversion (sp.lambdify) for the simplest
             metric, including the case where every expression is a constant.
    """
    assert not Minkowski().christoffel(X).any()


def test_minkowski_spherical_christoffel():
    """
    Minkowski (spherical): numerical Γʳ_θθ equals -r at r = 6

    Input:   MinkowskiSpherical(), evaluated at X = (t, r, θ, φ) = (0, 6, 1.2, 0).
    Checks:  christoffel(X)[1, 2, 2], i.e. Γ^r_θθ, equals -r = -6.
    Why:     a nonzero Christoffel symbol with a known value, so this checks both the
             numbers and the index order [upper, lower, lower] of the returned array.
    """
    assert np.isclose(MinkowskiSpherical().christoffel(X)[1, 2, 2], -X[1])  # Γ^r_θθ = -r


def test_schwarzschild_inverse_metric():
    """
    Schwarzschild (M = 1): g · g⁻¹ is the identity at r = 6, θ = 1.2

    Input:   Schwarzschild(M=1), evaluated at X = (0, 6, 1.2, 0), outside the horizon r = 2.
    Checks:  g(X) @ ginv(X) is the 4×4 identity matrix.
    Why:     the inverse metric is used in every Christoffel symbol, so it must be
             consistent with g at actual numerical points.
    """
    s = Schwarzschild(M=1.0)
    assert np.allclose(s.g(X) @ s.ginv(X), np.eye(4))


def test_schwarzschild_circular_orbit_norm_energy_angular_momentum():
    """
    Schwarzschild (M = 1): circular orbit at r = 10 has g(u,u) = -1, E ≈ 0.9562 and L_z ≈ 3.7796

    Input:   one state on the circular orbit at r0 = 10M in the equatorial plane, with
             u^t = 1/sqrt(1 - 3M/r0) and u^φ = sqrt(M/r0³)/sqrt(1 - 3M/r0).
    Checks:  norm(x, u) = -1, energy(x, u) = (1 - 2M/r0) u^t ≈ 0.9562 and
             angular_momentum(x, u) = r0² u^φ ≈ 3.7796.
    Why:     these methods (via killing_charge) are what the diagnostics use to judge
             the integrator, so they are checked here against the exact values for an
             orbit where everything is known in closed form.
    """
    M, r0 = 1.0, 10.0
    schw = Schwarzschild(M=M)
    ut   = 1 / np.sqrt(1 - 3*M/r0)
    uphi = np.sqrt(M / r0**3) / np.sqrt(1 - 3*M/r0)
    x = np.array([0.0, r0, np.pi/2, 0.0])
    u = np.array([ut, 0.0, 0.0, uphi])

    assert np.isclose(schw.norm(x, u), -1.0)
    assert np.isclose(schw.energy(x, u), (1 - 2*M/r0) * ut)
    assert np.isclose(schw.energy(x, u), 0.9562, atol=1e-4)
    assert np.isclose(schw.angular_momentum(x, u), r0**2 * uphi)
    assert np.isclose(schw.angular_momentum(x, u), 3.7796, atol=1e-4)


@pytest.mark.slow
def test_kerr_inverse_metric():
    """
    Kerr (M = 1, a = 0.9): g · g⁻¹ is the identity at r = 6, θ = 1.2

    Input:   Kerr(M=1, a=0.9), evaluated at X = (0, 6, 1.2, 0).
    Checks:  g(X) @ ginv(X) is the 4×4 identity matrix.
    Why:     unlike Schwarzschild, Kerr has the off-diagonal g_tφ term, so the
             inverse is not just 1/g_μμ on the diagonal. Marked slow because
             building Kerr computes its Christoffel symbols symbolically.
    """
    kerr = Kerr(M=1.0, a=0.9)
    assert np.allclose(kerr.g(X) @ kerr.ginv(X), np.eye(4))


@pytest.mark.slow
def test_kerr_a0_is_schwarzschild():
    """
    Kerr with spin a = 0 reproduces all 64 Schwarzschild Γ numerically

    Input:   Kerr(M=1, a=0) and Schwarzschild(M=1), both at X = (0, 6, 1.2, 0).
    Checks:  all 64 Christoffel symbols of the two metrics agree numerically.
    Why:     with zero spin the Kerr metric reduces exactly to Schwarzschild, so this
             checks the long Kerr metric formula against the simpler one.
    """
    K = Kerr(M=1.0, a=0.0).christoffel(X)
    S = Schwarzschild(M=1.0).christoffel(X)
    assert np.allclose(K, S)


@pytest.mark.slow
def test_kerr_is_vacuum():
    """
    Kerr (symbolic M, a): all 16 Ricci components vanish exactly, so it is a vacuum solution

    Input:   the symbolic Christoffel symbols of Kerr, with M and a kept as symbols.
    Checks:  the Riemann tensor is computed and all 16 Ricci components simplify to exactly 0.
    Why:     Kerr is a vacuum solution of Einstein's equations, but showing R_μv = 0
             needs heavy trig/rational simplification. This is the hardest test of
             smart_simplify (about 10 s) and a full check of the symbolic pipeline.
    """
    from grsim import curvature

    kerr = Kerr(M=1.0, a=0.9)
    Riemann = curvature.riemann_tensor(kerr.Gamma_sym, kerr.coords)
    Ricci = curvature.ricci_tensor(Riemann)
    assert all(component == 0 for row in Ricci for component in row)
