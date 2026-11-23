"""
grsim: symbolic curvature and numerical geodesics for general relativity.

Modules:
    grsim.curvature   - exact (SymPy) Christoffel symbols, Riemann and Ricci tensors,
                        Ricci and Kretschmann scalars of any metric
    grsim.metric      - ready-made spacetimes (Minkowski, Schwarzschild, Kerr) that can
                        be evaluated numerically at a point
    grsim.integrator  - geodesic equation and an RK4 integrator to follow particles
    grsim.diagnostics - conserved quantities along a path (g(u,u), energy E, angular
                        momentum L_z) to measure the integration error

The metric classes are also available directly, e.g.

    from grsim import Schwarzschild
    bh = Schwarzschild(M=1.0)
    bh.christoffel((0.0, 6.0, 1.2, 0.0))     # Γ^μ_aβ at t=0, r=6, θ=1.2, φ=0
"""
from . import curvature, metric
from .metric import Metric, Minkowski, MinkowskiSpherical, Schwarzschild, Kerr
