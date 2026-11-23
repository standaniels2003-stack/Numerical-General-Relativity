"""
Symbolic curvature of a metric, computed exactly with SymPy.

Pipeline (each step uses the result of the previous one):

    metric g_μv  ->  Christoffel symbols Γ^μ_aβ      christoffel_symbols
                 ->  Riemann tensor R^a_oμv          riemann_tensor
                 ->  Ricci tensor R_μv               ricci_tensor
                 ->  Ricci scalar R                  ricci_scalar
                 ->  Kretschmann scalar K            kretschmann_scalar

full_curvature runs the whole chain in one call. Every component is passed
through smart_simplify, so quantities that vanish come out as an exact 0.

Conventions:
    - Metrics are sympy.Matrix objects; coordinates are lists of SymPy symbols
      in the same order as the matrix rows/columns.
    - Tensors are nested Python lists indexed in the order they are written,
      e.g. Gamma[μ][a][β] = Γ^μ_aβ and Riemann[p][a][μ][v] = R^p_oμv.
    - Works in any number of dimensions (e.g. the 2-sphere in the tests).
"""
import sympy as sp
# =====================================================================================
def _reduce_cos_squares(expr):
    """
    Removes every cos² from an expression using cos²x = 1 - sin²x.

    What happens:
        Each power cos(x)**n with n >= 2 becomes cos(x)**(n % 2) * (1 - sin(x)**2)**(n // 2),
        so afterwards cos(x) appears at most to the first power. The result is expanded.
        Applied to a polynomial in sin and cos, this gives one unique form for
        each expression, which is what lets smart_simplify detect exact zeros.

    Parameters:
        expr (sympy.Expr): Expression (normally a polynomial numerator or denominator).

    Returns:
        sympy.Expr: The equivalent expanded expression without cos² terms.
    """
    expr = expr.replace(
        lambda e: e.is_Pow and isinstance(e.base, sp.cos) and e.exp.is_Integer and e.exp >= 2,
        lambda e: e.base**(e.exp % 2) * (1 - sp.sin(e.base.args[0])**2)**(e.exp // 2),
    )
    return sp.expand(expr)


def smart_simplify(term, trigsimp=False):
    """
    Performs fast, deterministic algebraic and trigonometric simplification.

    Steps:
        1. Expand multiple-angle identities (sin(2x) -> 2 sin(x) cos(x)) and
           rewrite tan(x) as sin(x)/cos(x).
        2. Put the expression over a common denominator (sp.cancel).
        3. In numerator and denominator, replace cos(x)**2 by 1 - sin(x)**2.
           This is a normal form for polynomials in sin and cos, so any
           combination that vanishes because sin^2 + cos^2 = 1 really becomes 0.
        4. Cancel again and factor for a compact result.

    This deliberately avoids sp.simplify: its heuristics depend on Python's
    randomised hash ordering, so the same Kerr Christoffel symbol could take
    1 s in one run and over 100 s in the next.

    Parameters:
        term (sympy.Expr): The symbolic SymPy expression to simplify.
        trigsimp (bool, optional): If True, additionally applies sp.trigsimp()
                                   for a prettier (but slower) result. Defaults to False.

    Returns:
        sympy.Expr: The simplified expression.
    """
    term = sp.sympify(term)
    term = sp.expand_trig(term)
    term = term.replace(sp.tan, lambda x: sp.sin(x) / sp.cos(x))

    num, den = sp.fraction(sp.cancel(term))
    num = _reduce_cos_squares(sp.expand(num))
    den = _reduce_cos_squares(sp.expand(den))
    term = sp.factor(sp.cancel(num / den))

    if trigsimp:
        term = sp.factor(sp.cancel(sp.trigsimp(term)))
    return term

# =====================================================================================
def christoffel_symbols(g, coords, trigsimp=False, g_inv=None):
    """
    Computes the Christoffel symbols (of the second kind) of a metric.

    Formula:
        Γ^i_jk = 1/2 g^il (∂_j g_kl + ∂_k g_jl - ∂_l g_jk)

    What happens:
        The metric is differentiated symbolically with respect to each coordinate,
        contracted with the inverse metric, and every component is simplified
        with smart_simplify. All n³ components are computed (Γ^i_jk = Γ^i_kj,
        so about half of them are duplicates). They describe how the coordinate
        basis changes from point to point; they appear directly in the geodesic
        equation and are the input for riemann_tensor.

    Parameters:
        g (sympy.Matrix): The metric g_ij, an nxn symmetric matrix.
        coords (list): The n coordinate symbols, in the order of the rows of g.
        trigsimp (bool): If True, also applies sp.trigsimp (prettier, slower).
        g_inv (sympy.Matrix, optional): Precomputed inverse metric g^ij, to avoid
                                        computing it twice. Computed from g if omitted.

    Returns:
        list: Nested list of shape nxnxn with Gamma[i][j][k] = Γ^i_jk (SymPy expressions).

    Example:
        >>> r, th = sp.symbols("r theta", positive=True)
        >>> christoffel_symbols(sp.diag(1, r**2), [r, th])[0][1][1]    # flat plane in polar coordinates
        -r
    """
    n = g.shape[0]
    if g_inv is None:
        g_inv = g.inv()
    Gamma = [[[0 for i in range(n)] for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                term = 0
                for l in range(n):
                    dg_kl_j = g[k, l].diff(coords[j])
                    dg_jl_k = g[j, l].diff(coords[k])
                    dg_jk_l = g[j, k].diff(coords[l])
                    
                    term += g_inv[i, l] * (dg_kl_j + dg_jl_k - dg_jk_l)

                Gamma[i][j][k] = smart_simplify(sp.Rational(1, 2) * term, trigsimp)
    return Gamma

# =====================================================================================
def riemann_tensor(gamma, coords, trigsimp=False):
    """
    Computes the Riemann curvature tensor R^i_jkl from the Christoffel symbols.

    Formula:
        R^i_jkl = ∂_k Γ^i_jl - ∂_l Γ^i_jk + Γ^i_km Γ^m_jl - Γ^i_lm Γ^m_jk

    What happens:
        The Christoffel symbols are differentiated and multiplied according to the
        formula above, and each of the n⁴ components is simplified. The Riemann
        tensor measures true curvature: it is zero everywhere if and only if
        spacetime is flat, whatever coordinates are used. This is the slowest
        step of the chain (for Kerr it takes about 10 s).

    Parameters:
        gamma (list): Christoffel symbols as returned by christoffel_symbols,
                      gamma[i][j][k] = Γ^i_jk.
        coords (list): The coordinate symbols used for gamma.
        trigsimp (bool): If True, also applies sp.trigsimp (prettier, slower).

    Returns:
        list: Nested list of shape nxnxnxn with Riemann[i][j][k][l] = R^i_jkl
              (first index up, the other three down).
    """
    n = len(gamma)
    Riemann = [[[[0 for i in range(n)] for i in range(n)] for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    dgamma_k = gamma[i][j][l].diff(coords[k])
                    dgamma_l = gamma[i][j][k].diff(coords[l])

                    term = dgamma_k - dgamma_l

                    for m in range(n):
                        gammagamma_k = gamma[i][k][m] * gamma[m][j][l]
                        gammagamma_l = gamma[i][l][m] * gamma[m][j][k]
                        term += gammagamma_k - gammagamma_l

                    Riemann[i][j][k][l] = smart_simplify(term, trigsimp)
    return Riemann

# =====================================================================================
def ricci_tensor(Riemann, trigsimp=False):
    """
    Computes the Ricci tensor R_ij by contracting the Riemann tensor.

    Formula:
        R_ij = R^k_ikj   (sum over k: the upper index with the third index)

    What happens:
        For each pair (i, j) the n terms R^k_ikj are summed and simplified.
        No metric is needed for this contraction. In vacuum (no matter or
        energy), Einstein's equations reduce to R_ij = 0, so this is the
        quantity to check for vacuum solutions such as Schwarzschild and Kerr.

    Parameters:
        Riemann (list): Riemann tensor as returned by riemann_tensor,
                        Riemann[i][j][k][l] = R^i_jkl.
        trigsimp (bool): If True, also applies sp.trigsimp (prettier, slower).

    Returns:
        list: Nested list of shape nxn with Ricci[i][j] = R_ij (symmetric).
    """
    n = len(Riemann)
    Ricci = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            term = 0
            for k in range(n):
                term += Riemann[k][i][k][j]

            Ricci[i][j] = smart_simplify(term, trigsimp)
    return Ricci

# =====================================================================================
def ricci_scalar(g, Ricci, trigsimp=False):
    """
    Computes the Ricci scalar (scalar curvature) R from the Ricci tensor.

    Formula:
        R = g^ij R_ij

    What happens:
        The inverse metric is computed from g, multiplied with the Ricci tensor
        and summed over both indices, then simplified. The result is a single
        coordinate-independent number at each point, e.g. R = 2/a² for a sphere
        of radius a and R = 0 for any vacuum solution.

    Parameters:
        g (sympy.Matrix): The metric g_ij (not the inverse; the inverse is
                          computed inside this function).
        Ricci (list): Ricci tensor as returned by ricci_tensor, Ricci[i][j] = R_ij.
        trigsimp (bool): If True, also applies sp.trigsimp (prettier, slower).

    Returns:
        sympy.Expr: The Ricci scalar R.
    """
    n = len(Ricci)
    g_inv = g.inv()
    R = 0
    for i in range(n):
        for j in range(n):
            R += g_inv[i, j] * Ricci[i][j]

    return smart_simplify(R, trigsimp)

# =====================================================================================
def kretschmann_scalar(g, Riemann, trigsimp=False):
    """
    Computes the Kretschmann scalar K = R_{abcd} R^{abcd}.

    K is the full contraction of the Riemann tensor with itself. It is a
    coordinate-invariant scalar, used to detect true curvature singularities
    (K diverges) as opposed to coordinate singularities (K stays finite).
    For Schwarzschild: K = 12 rs^2 / r^6 = 48 M^2 / r^6 (rs = 2M).

    Method (the metric is essential: every contracted pair must be one
    index up and one index down):
        1. Lower the first index:  R_{abcd} = g_{ae} R^e_{bcd}
        2. Raise all four indices one at a time with the inverse metric:
           R^{abcd} = g^{ae} g^{bf} g^{cg} g^{dh} R_{efgh}
        3. Contract:  K = sum over a,b,c,d of R_{abcd} * R^{abcd}
    This also works for non-diagonal metrics (e.g. Kerr).

    Parameters:
        g (sympy.Matrix): The metric tensor g_{ab} (not the inverse; the
                          inverse is computed inside this function).
        Riemann (list): The 4D nested list of Riemann tensor components
                       R^a_{bcd} (first index up, the other three down),
                       as returned by riemann_tensor.
        trigsimp (bool): If True, applies heavy trigonometric simplification
                         to the final result.

    Returns:
        sympy expression: The Kretschmann scalar K.
    """
    n = len(Riemann)
    g_inv = g.inv()

    Riemann_down = [[[[sum(g[a, e] * Riemann[e][b][c][d] for e in range(n)) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]

    Riemann_up = Riemann_down
    for pos in range(4):
        raised = [[[[0] * n for i in range(n)] for i in range(n)] for i in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        idx = [a, b, c, d]
                        total = 0
                        for e in range(n):
                            src = idx.copy()
                            src[pos] = e
                            total += g_inv[idx[pos], e] * Riemann_up[src[0]][src[1]][src[2]][src[3]]
                        raised[a][b][c][d] = total
        Riemann_up = raised

    K = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    K += Riemann_down[i][j][k][l] * Riemann_up[i][j][k][l]
    K = smart_simplify(K, trigsimp)

    return K

# =====================================================================================
def full_curvature(g, coords, trigsimp=False):
    """
    Computes all curvature quantities of a metric in one call.

    What happens:
        Runs the whole chain: christoffel_symbols -> riemann_tensor ->
        ricci_tensor -> ricci_scalar and kretschmann_scalar, each step using
        the previous result. Run time is dominated by the Riemann tensor.

    Parameters:
        g (sympy.Matrix): The metric g_ij, an nxn symmetric matrix.
        coords (list): The n coordinate symbols, in the order of the rows of g.
        trigsimp (bool): If True, also applies sp.trigsimp at every step (prettier, slower).

    Returns:
        tuple: (gamma, Riemann, Ricci, R, K) where
            gamma[i][j][k]      = Γ^i_jk     (nxnxn nested list)
            Riemann[i][j][k][l] = R^i_jkl    (nxnxnxn nested list)
            Ricci[i][j]         = R_ij       (nxn nested list)
            R                   = Ricci scalar      (sympy.Expr)
            K                   = Kretschmann scalar (sympy.Expr)

    Example:
        >>> th, ph, a = sp.symbols("theta phi a", positive=True)
        >>> _, _, _, R, K = full_curvature(sp.diag(a**2, a**2 * sp.sin(th)**2), [th, ph])
        >>> R, K                                   # sphere of radius a
        (2/a**2, 4/a**4)
    """
    gamma = christoffel_symbols(g, coords, trigsimp)
    Riemann = riemann_tensor(gamma, coords, trigsimp)
    Ricci = ricci_tensor(Riemann, trigsimp)
    R = ricci_scalar(g, Ricci, trigsimp)
    K = kretschmann_scalar(g, Riemann, trigsimp)

    return gamma, Riemann, Ricci, R, K