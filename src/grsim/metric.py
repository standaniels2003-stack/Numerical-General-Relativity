"""
Spacetime metrics that can be evaluated numerically.

Each metric is defined once symbolically (with SymPy). From that, the inverse
metric and the Christoffel symbols are derived exactly, and then turned into
fast NumPy functions, so a geodesic integrator can evaluate g, g⁻¹ and Γ at
any point x = (x^0, x^1, x^2, x^3) with plain numbers.

Classes:
    Metric              - base class: give it a symbolic metric, get numerical g, g⁻¹, Γ
    Minkowski           - flat spacetime, Cartesian (t, x, y, z)
    MinkowskiSpherical  - flat spacetime, spherical (t, r, θ, φ)
    Schwarzschild       - non-rotating black hole of mass M
    Kerr                - rotating black hole of mass M and spin a

All metrics use the (-, +, +, +) signature and units with G = c = 1.
"""
import sympy as sp
import numpy as np

from grsim import curvature

class Metric:
    """
    Base class for a spacetime metric: symbolic definition, numerical evaluation.

    What happens when you create one:
        1. The inverse metric g^μν is computed symbolically.
        2. The Christoffel symbols Γ^μ_aβ are computed symbolically
           (see curvature.christoffel_symbols). This is the slow part: for
           Kerr it takes a couple of seconds.
        3. g, g⁻¹ and Γ are compiled into NumPy functions with sp.lambdify,
           with the parameter values (e.g. M = 1) filled in at call time.

    Parameters:
        g_sym (sympy.Matrix): The metric g_μv as a symbolic nxn matrix.
        coords (tuple): SymPy symbols for the coordinates, in the order of the
                        rows/columns of g_sym, e.g. (t, r, theta, phi).
        params (tuple): SymPy symbols for the metric's parameters, e.g. (M,).
                        Use () if there are none.
        param_values (tuple): Numerical values for params, in the same order, e.g. (1.0,).

    Attributes:
        g_sym, ginv_sym: The symbolic metric and inverse metric.
        Gamma_sym: Symbolic Christoffel symbols as a nested list, Gamma_sym[μ][a][β] = Γ^μ_aβ.
        coords, params, param_values: As passed in.

    Example:
        >>> s = Schwarzschild(M=1.0)
        >>> float(s.christoffel((0.0, 6.0, 1.2, 0.0))[1, 2, 2])   # Γ^r_θθ = -(r - 2M) at r = 6
        -4.0
    """

    def __init__(self, g_sym, coords, params, param_values):
        self.g_sym = g_sym
        self.coords = coords
        self.params = params
        self.param_values = param_values

        self.ginv_sym = g_sym.inv()
        self.Gamma_sym = curvature.christoffel_symbols(g_sym, self.coords, g_inv=self.ginv_sym)

        self._g_func = sp.lambdify((coords, params), self.g_sym, "numpy")
        self._ginv_func = sp.lambdify((coords, params), self.ginv_sym, "numpy")
        self._Gamma_func = sp.lambdify((coords, params), self.Gamma_sym, "numpy")

    def g(self, x):
        """
        Evaluates the metric g_μv at a point.

        Parameters:
            x (array_like): Coordinates of the point, length n, in the order of
                            self.coords, e.g. (t, r, θ, φ).

        Returns:
            np.ndarray: Shape (n, n), the metric components g_μv at x.
        """
        return np.array(self._g_func(x, self.param_values), dtype=float)

    def ginv(self, x):
        """
        Evaluates the inverse metric g^μν at a point.

        Parameters:
            x (array_like): Coordinates of the point, length n.

        Returns:
            np.ndarray: Shape (n, n), the components g^μν at x; g(x) @ ginv(x) is the identity.
        """
        return np.array(self._ginv_func(x, self.param_values), dtype=float)

    def christoffel(self, x):
        """
        Evaluates the Christoffel symbols Γ^μ_aβ at a point.

        Parameters:
            x (array_like): Coordinates of the point, length n.

        Returns:
            np.ndarray: Shape (n, n, n), with christoffel(x)[μ, a, β] = Γ^μ_aβ(x).
                        This is what geodesic_rhs uses to compute the acceleration.
        """
        return np.array(self._Gamma_func(x, self.param_values), dtype=float)

    def killing_charge(self, x, u, xi):
        """
        Conserved quantity g_μv ξ^μ u^v belonging to a Killing vector ξ.

        What happens:
            A Killing vector ξ points along a symmetry of the spacetime (e.g. ξ = ∂_t
            if the metric does not depend on t). Along any geodesic, the
            contraction g_μv ξ^μ u^v is then constant. energy() and
            angular_momentum() are this function with a specific ξ.

        Parameters:
            x (array_like): Position, length 4, in the metric's coordinates.
            u (array_like): 4-velocity u^μ at x, length 4.
            xi (array_like): Components ξ^μ of the Killing vector at x, length 4.

        Returns:
            float: g_μv ξ^μ u^v.
        """
        return xi @ self.g(x) @ u

    def norm(self, x, u):
        """
        The squared length g_μv u^μ u^v of the 4-velocity.

        Parameters:
            x (array_like): Position, length 4.
            u (array_like): 4-velocity u^μ at x, length 4.

        Returns:
            float: g_μv u^μ u^v; -1 for a massive particle with τ = proper time,
                   0 for light, and constant along any exact geodesic.
        """
        return u @ self.g(x) @ u

    def energy(self, x, u):
        """
        Conserved energy E = -g_μv ξ^μ u^v with ξ the time-translation Killing vector.

        Defined by each subclass that has a time symmetry (here all of them,
        with ξ = ∂_t, so E = -u_t). For a massive particle E is the energy per
        unit mass measured far away.

        Parameters:
            x (array_like): Position, length 4.
            u (array_like): 4-velocity u^μ at x, length 4.

        Returns:
            float: E.

        Raises:
            NotImplementedError: For a Metric with no energy() defined.
        """
        raise NotImplementedError(f"{type(self).__name__} has no energy() defined")

    def angular_momentum(self, x, u):
        """
        Conserved angular momentum L_z = g_μv ξ^μ u^v about the z-axis (rotation Killing vector ξ).

        Defined by each subclass that is symmetric under rotations about the
        z-axis. In (t, r, θ, φ) coordinates ξ = ∂_φ, so L_z = u_φ; in Cartesian
        Minkowski ξ = x ∂_y - y ∂_x.

        Parameters:
            x (array_like): Position, length 4.
            u (array_like): 4-velocity u^μ at x, length 4.

        Returns:
            float: L_z.

        Raises:
            NotImplementedError: For a Metric with no angular_momentum() defined.
        """
        raise NotImplementedError(f"{type(self).__name__} has no angular_momentum() defined")

class Minkowski(Metric):
    """
    Flat spacetime in Cartesian coordinates (t, x, y, z).

    Metric: ds² = -dt² + dx² + dy² + dz².
    No parameters. All Christoffel symbols are zero, so geodesics are straight lines.

    Example:
        >>> bool(Minkowski().christoffel((0, 1, 2, 3)).any())
        False
    """

    def __init__(self):
        t, x, y, z = sp.symbols("t x y z", real=True)
        g_sym = sp.diag(-1, 1, 1, 1)

        super().__init__(g_sym, (t, x, y, z), (), ())

    def energy(self, x, u):
        return -self.killing_charge(x, u, np.array([1.0, 0.0, 0.0, 0.0]))

    def angular_momentum(self, x, u):
        t, X, Y, Z = x
        return self.killing_charge(x, u, np.array([0.0, -Y, X, 0.0]))


class MinkowskiSpherical(Metric):
    """
    Flat spacetime in spherical coordinates (t, r, θ, φ).

    Metric: ds² = -dt² + dr² + r² dθ² + r² sin²θ dφ².
    No parameters. Spacetime is still flat (zero curvature), but the Christoffel
    symbols are not zero because the coordinates are curvilinear, e.g. Γ^r_θθ = -r.
    Useful for checking that a code does not confuse Γ ≠ 0 with curvature.
    """

    def __init__(self):
        t, r, theta, phi = sp.symbols("t r theta phi", real=True)
        g_sym = sp.diag(-1, 1, r**2, r**2 * sp.sin(theta)**2)

        super().__init__(g_sym, (t, r, theta, phi), (), ())

    def energy(self, x, u):
        return -self.killing_charge(x, u, np.array([1.0, 0.0, 0.0, 0.0]))

    def angular_momentum(self, x, u):
        return self.killing_charge(x, u, np.array([0.0, 0.0, 0.0, 1.0]))


class Schwarzschild(Metric):
    """
    Non-rotating, uncharged black hole in Schwarzschild coordinates (t, r, θ, φ).

    Metric: ds² = -(1 - 2M/r) dt² + dr² / (1 - 2M/r) + r² dθ² + r² sin²θ dφ².
    The horizon is at r = 2M (a coordinate singularity; g_rr blows up there),
    the true curvature singularity at r = 0.

    Parameters:
        M (float): Mass of the black hole, in units with G = c = 1 (so r is measured in units of M).
    """

    def __init__(self, M):
        t, r, theta, phi = sp.symbols("t r theta phi", real=True)
        M_sym = sp.Symbol("M", real=True)

        f = 1 - 2 * M_sym / r
        g_sym = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(theta)**2)

        super().__init__(g_sym, (t, r, theta, phi), (M_sym,), (M,))

    def energy(self, x, u):
        return -self.killing_charge(x, u, np.array([1.0, 0.0, 0.0, 0.0]))

    def angular_momentum(self, x, u):
        return self.killing_charge(x, u, np.array([0.0, 0.0, 0.0, 1.0]))


class Kerr(Metric):
    """
    Rotating, uncharged black hole in Boyer-Lindquist coordinates (t, r, θ, φ).

    Metric, with Σ = r² + a² cos²θ and Δ = r² - 2Mr + a²:
        ds² = -(1 - 2Mr/Σ) dt² - (4Mar sin²θ / Σ) dt dφ + (Σ/Δ) dr² + Σ dθ²
              + (r² + a² + 2Ma²r sin²θ / Σ) sin²θ dφ²
    The dt dφ cross term (frame dragging) makes the metric non-diagonal.
    With a = 0 it reduces exactly to Schwarzschild.

    Parameters:
        M (float): Mass of the black hole (G = c = 1).
        a (float): Spin parameter a = J/M, with |a| <= M for a black hole.
                   The outer horizon is at r = M + sqrt(M² - a²).
    """

    def __init__(self, M, a):
        t, r, theta, phi = sp.symbols("t r theta phi", real=True)
        M_sym, a_sym = sp.symbols("M a", real=True)

        Sigma = r**2 + a_sym**2 * sp.cos(theta)**2
        Delta = r**2 - 2 * M_sym * r + a_sym**2
        s2 = sp.sin(theta)**2

        g_sym = sp.zeros(4, 4)
        g_sym[0, 0] = -(1 - 2 * M_sym * r / Sigma)
        g_sym[0, 3] = g_sym[3, 0] = -2 * M_sym * a_sym * r * s2 / Sigma
        g_sym[1, 1] = Sigma / Delta
        g_sym[2, 2] = Sigma
        g_sym[3, 3] = (r**2 + a_sym**2 + 2 * M_sym * a_sym**2 * r * s2 / Sigma) * s2

        super().__init__(g_sym, (t, r, theta, phi), (M_sym, a_sym), (M, a))

    def energy(self, x, u):
        return -self.killing_charge(x, u, np.array([1.0, 0.0, 0.0, 0.0]))

    def angular_momentum(self, x, u):
        return self.killing_charge(x, u, np.array([0.0, 0.0, 0.0, 1.0]))
