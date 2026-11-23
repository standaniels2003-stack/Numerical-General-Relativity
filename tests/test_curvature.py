"""Symbolic curvature: Christoffels, Riemann, Ricci and Kretschmann from a metric."""
import sympy as sp

from grsim import curvature

t, r, theta, phi = sp.symbols("t r theta phi", positive=True)
x, y, z = sp.symbols("x y z", real=True)
rs, a = sp.symbols("rs a", positive=True)

coords_spherical = [t, r, theta, phi]
coords_flat = [t, x, y, z]

metric_flat = sp.diag(-1, 1, 1, 1)
metric_spherical = sp.diag(-1, 1, r**2, r**2 * sp.sin(theta)**2)
metric_schwarzschild = sp.diag(-(1 - rs / r), 1 / (1 - rs / r), r**2, r**2 * sp.sin(theta)**2)


def all_zero(nested):
    """
    Helper (not a test): True if every entry of a nested list of SymPy expressions is exactly 0.

    Input:   a single expression, or a (nested) list of them such as Γ or Riemann.
    Returns: bool. Uses == 0 on purpose rather than sp.simplify, so it only passes
             if smart_simplify itself already produced an exact zero.
    """
    if isinstance(nested, list):
        return all(all_zero(item) for item in nested)
    return nested == 0  # the simplifier itself must produce an exact 0


def test_flat_cartesian_is_flat():
    """
    Minkowski in (t, x, y, z): every Γ, Riemann, Ricci, R and Kretschmann K is exactly 0

    Input:   the Minkowski metric diag(-1, 1, 1, 1) in Cartesian coordinates.
    Checks:  full_curvature returns exactly 0 for every Christoffel symbol, every
             Riemann and Ricci component, the Ricci scalar R and Kretschmann K.
    Why:     the simplest possible case; a constant metric has no derivatives at all,
             so anything nonzero would point to a bug in the formulas themselves.
    """
    gamma, Riemann, Ricci, R, K = curvature.full_curvature(metric_flat, coords_flat)
    assert all_zero(gamma)
    assert all_zero(Riemann)
    assert all_zero(Ricci)
    assert R == 0
    assert K == 0


def test_flat_spherical_has_christoffels_but_no_curvature():
    """
    Minkowski in (t, r, θ, φ): Γʳ_θθ = -r, Γᶿ_rθ = 1/r, Γʳ_φφ = -r sin²θ, yet all curvature is 0

    Input:   the same flat spacetime in spherical coordinates, diag(-1, 1, r², r² sin²θ).
    Checks:  three Christoffel symbols have their textbook values, while the Riemann
             tensor, Ricci tensor, R and K are all exactly 0.
    Why:     Γ ≠ 0 here only because the coordinates are curved, not spacetime. This
             makes sure the code doesn't confuse the two, and exercises the trig
             simplification (sin²θ + cos²θ = 1) needed to get the zeros.
    """
    gamma, Riemann, Ricci, R, K = curvature.full_curvature(metric_spherical, coords_spherical)
    assert sp.simplify(gamma[1][2][2] + r) == 0                       # Γ^r_θθ = -r
    assert sp.simplify(gamma[2][1][2] - 1 / r) == 0                   # Γ^θ_rθ = 1/r
    assert sp.simplify(gamma[1][3][3] + r * sp.sin(theta)**2) == 0    # Γ^r_φφ = -r sin²θ
    assert all_zero(Riemann)
    assert all_zero(Ricci)
    assert sp.simplify(R) == 0
    assert sp.simplify(K) == 0


def test_two_sphere_ricci_scalar():
    """
    2-sphere of radius a: R_θθ = 1, R_φφ = sin²θ, Ricci scalar R = 2/a², Kretschmann K = 4/a⁴

    Input:   the 2D metric of a sphere of radius a: diag(a², a² sin²θ) in (θ, φ).
    Checks:  R_θθ = 1, R_φφ = sin²θ, Ricci scalar R = 2/a², Kretschmann K = 4/a⁴.
    Why:     a curved space with a known answer. It also guards the bug where
             ricci_scalar contracted with g_ij instead of the inverse g^ij, which
             gave a²(sin⁴θ + 1) here; flat and vacuum tests can't catch that, since
             their Ricci tensor is zero anyway.
    """
    # A sphere of radius a has constant curvature R = 2/a^2.
    g = sp.diag(a**2, a**2 * sp.sin(theta)**2)
    _, _, Ricci, R, K = curvature.full_curvature(g, [theta, phi])
    assert sp.simplify(Ricci[0][0] - 1) == 0
    assert sp.simplify(Ricci[1][1] - sp.sin(theta)**2) == 0
    assert sp.simplify(R - 2 / a**2) == 0
    assert sp.simplify(K - 4 / a**4) == 0


def test_schwarzschild_vacuum_and_kretschmann():
    """
    Schwarzschild: vacuum (R_μv = 0, R = 0) with Kretschmann K = 12 r_s² / r⁶

    Input:   the Schwarzschild metric with Schwarzschild radius r_s = 2M.
    Checks:  every Ricci component and R vanish, and K = 12 r_s² / r⁶.
    Why:     Schwarzschild is a vacuum solution of Einstein's equations, so R_μv = 0.
             Its Riemann tensor is not zero, and K is the standard check that the
             curvature itself is right (it diverges at r = 0, the real singularity,
             but stays finite at the horizon r = r_s).
    """
    _, _, Ricci, R, K = curvature.full_curvature(metric_schwarzschild, coords_spherical)
    assert all_zero(Ricci)
    assert sp.simplify(R) == 0
    assert sp.simplify(K - 12 * rs**2 / r**6) == 0
