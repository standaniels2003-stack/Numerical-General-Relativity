import numpy as np
import pytest

from grsim import Kerr, Minkowski, MinkowskiSpherical, Schwarzschild

X = (0.0, 6.0, 1.2, 0.0)


def test_minkowski_christoffels_vanish():
    """Minkowski metric class: numerical Christoffels are zero"""
    assert not Minkowski().christoffel(X).any()


def test_minkowski_spherical_christoffel():
    """Spherical Minkowski class: Gamma^r_thth = -r"""
    assert np.isclose(MinkowskiSpherical().christoffel(X)[1, 2, 2], -X[1])  # Γ^r_θθ = -r


def test_schwarzschild_inverse_metric():
    """Schwarzschild class: g . g^-1 = identity"""
    s = Schwarzschild(M=1.0)
    assert np.allclose(s.g(X) @ s.ginv(X), np.eye(4))


@pytest.mark.slow
def test_kerr_inverse_metric():
    """Kerr class (a = 0.9): g . g^-1 = identity"""
    kerr = Kerr(M=1.0, a=0.9)
    assert np.allclose(kerr.g(X) @ kerr.ginv(X), np.eye(4))


@pytest.mark.slow
def test_kerr_a0_is_schwarzschild():
    """Kerr class with a = 0 reduces to Schwarzschild"""
    K = Kerr(M=1.0, a=0.0).christoffel(X)
    S = Schwarzschild(M=1.0).christoffel(X)
    assert np.allclose(K, S)


@pytest.mark.slow
def test_kerr_is_vacuum():
    """Kerr (symbolic M, a): Ricci tensor vanishes exactly"""
    from grsim import curvature

    kerr = Kerr(M=1.0, a=0.9)
    Riemann = curvature.riemann_tensor(kerr.Gamma_sym, kerr.coords)
    Ricci = curvature.ricci_tensor(Riemann)
    assert all(component == 0 for row in Ricci for component in row)
