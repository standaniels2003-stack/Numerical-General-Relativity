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
    if isinstance(nested, list):
        return all(all_zero(item) for item in nested)
    return nested == 0  # the simplifier itself must produce an exact 0


def test_flat_cartesian_is_flat():
    """Flat space (Cartesian): all Christoffels and curvature vanish"""
    gamma, Riemann, Ricci, R, K = curvature.full_curvature(metric_flat, coords_flat)
    assert all_zero(gamma)
    assert all_zero(Riemann)
    assert all_zero(Ricci)
    assert R == 0
    assert K == 0


def test_flat_spherical_has_christoffels_but_no_curvature():
    """Flat space (spherical): Christoffels nonzero, curvature zero"""
    gamma, Riemann, Ricci, R, K = curvature.full_curvature(metric_spherical, coords_spherical)
    assert sp.simplify(gamma[1][2][2] + r) == 0                       # Γ^r_θθ = -r
    assert sp.simplify(gamma[2][1][2] - 1 / r) == 0                   # Γ^θ_rθ = 1/r
    assert sp.simplify(gamma[1][3][3] + r * sp.sin(theta)**2) == 0    # Γ^r_φφ = -r sin²θ
    assert all_zero(Riemann)
    assert all_zero(Ricci)
    assert sp.simplify(R) == 0
    assert sp.simplify(K) == 0


def test_two_sphere_ricci_scalar():
    """2-sphere of radius a: R = 2/a^2 and K = 4/a^4"""
    # A sphere of radius a has constant curvature R = 2/a^2.
    g = sp.diag(a**2, a**2 * sp.sin(theta)**2)
    _, _, Ricci, R, K = curvature.full_curvature(g, [theta, phi])
    assert sp.simplify(Ricci[0][0] - 1) == 0
    assert sp.simplify(Ricci[1][1] - sp.sin(theta)**2) == 0
    assert sp.simplify(R - 2 / a**2) == 0
    assert sp.simplify(K - 4 / a**4) == 0


def test_schwarzschild_vacuum_and_kretschmann():
    """Schwarzschild: vacuum (R_ij = 0) and K = 12 rs^2 / r^6"""
    _, _, Ricci, R, K = curvature.full_curvature(metric_schwarzschild, coords_spherical)
    assert all_zero(Ricci)
    assert sp.simplify(R) == 0
    assert sp.simplify(K - 12 * rs**2 / r**6) == 0
