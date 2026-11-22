import sympy as sp
import numpy as np

from . import curvature

class Metric:
    """
    =================================================
                    Base metric class
    =================================================
    Initialization takes:
        g_sym - a SymPy matrix
        coords - SymPy symbols used as the coordinates of the metric
        params - SymPy symbols used as parameters(example: M as mass)
        param_values - a list with the values for the parameters
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
        return np.array(self._g_func(x, self.param_values), dtype=float)
    
    def ginv(self, x):
        return np.array(self._ginv_func(x, self.param_values), dtype=float)

    def christoffel(self, x):
        return np.array(self._Gamma_func(x, self.param_values), dtype=float)

class Minkowski(Metric):
    """
    Flat spacetime in Cartesian coordinates (t, x, y, z). All Christoffels are zero.
    """

    def __init__(self):
        t, x, y, z = sp.symbols("t x y z", real=True)
        g_sym = sp.diag(-1, 1, 1, 1)
        
        super().__init__(g_sym, (t, x, y, z), (), ())


class MinkowskiSpherical(Metric):
    """
    Flat spacetime in spherical coordinates (t, r, theta, phi). Christoffels are NOT zero.
    """

    def __init__(self):
        t, r, theta, phi = sp.symbols("t r theta phi", real=True)
        g_sym = sp.diag(-1, 1, r**2, r**2 * sp.sin(theta)**2)

        super().__init__(g_sym, (t, r, theta, phi), (), ())


class Schwarzschild(Metric):
    """
    Schwarzschild in Schwarzschild coordinates (t, r, theta, phi), mass M.
    """

    def __init__(self, M):
        t, r, theta, phi = sp.symbols("t r theta phi", real=True)
        M_sym = sp.Symbol("M", real=True)

        f = 1 - 2 * M_sym / r
        g_sym = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(theta)**2)

        super().__init__(g_sym, (t, r, theta, phi), (M_sym,), (M,))


class Kerr(Metric):
    """
    Kerr in Boyer-Lindquist coordinates (t, r, theta, phi), mass M and spin a.
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
