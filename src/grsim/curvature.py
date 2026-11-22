import sympy as sp
# =====================================================================================
def _reduce_cos_squares(expr):
    """Rewrites cos(x)**n (n >= 2) as cos(x)**(n % 2) * (1 - sin(x)**2)**(n // 2), then expands."""
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
    Function to calculate the Christoffel Symbols using the metric tensor.

    Mathematical Formula: Gamma^i_{jk} = 1/2 * g^{il} * (g_{kl,j} + g_{jl,k} - g_{jk,l})

    Parameters:
        g (sympy.Matrix): The metric tensor.
        coords (list): List of coordinate symbols.
        trigsimp (bool): If True, applies heavy trigonometric simplification.
        g_inv (sympy.Matrix, optional): Precomputed inverse metric. Computed from g if omitted.

    Returns:
        list: A 3D nested list representing the components Gamma^i_{jk}
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
    Computes the Riemann curvature tensor of the second kind from Christoffel symbols.

    Mathematical Formula:
        R^i_{jkl} = d_k(Gamma^i_{jl}) - d_l(Gamma^i_{jk}) + Gamma^i_{km} Gamma^m_{jl} - Gamma^i_{lm} Gamma^m_{jk}

    Parameters:
        gamma (list): The 3D nested list of Christoffel symbols.
        coords (list): List of coordinate symbols.
        trigsimp (bool): If True, applies heavy trigonometric simplification.

    Returns:
        list: A 4D nested list representing the components R^i_{jkl}.
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
    Computes the Ricci tensor by contracting the Riemann tensor.

    Mathematical Formula:
        R_{ij} = R^k_{ikj}

    Parameters:
        Riemann (list): The 4D Riemann tensor.
        trigsimp (bool): If True, applies heavy trigonometric simplification.

    Returns:
        list: A 2D matrix (list of lists) representing the Ricci tensor components.
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
    Computes the Ricci scalar by contracting the Ricci tensor with the inverse metric.

    Mathematical Formula:
        R = g^{ij} R_{ij}

    Parameters:
        g (sympy.Matrix): The metric tensor g_{ij} (not the inverse; the
                          inverse is computed inside this function).
        Ricci (list): The Ricci tensor.
        trigsimp (bool): If True, applies heavy trigonometric simplification.

    Returns:
        sympy expression: The scalar curvature.
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
    Computes all curvature quantities of a metric in one go.

    Parameters:
        g (sympy.Matrix): The metric tensor g_{ij}.
        coords (list): List of coordinate symbols.
        trigsimp (bool): If True, applies heavy trigonometric simplification.

    Returns:
        tuple: (Gamma^i_{jk}, R^i_{jkl}, R_{ij}, R, K)
    """
    gamma = christoffel_symbols(g, coords, trigsimp)
    Riemann = riemann_tensor(gamma, coords, trigsimp)
    Ricci = ricci_tensor(Riemann, trigsimp)
    R = ricci_scalar(g, Ricci, trigsimp)
    K = kretschmann_scalar(g, Riemann, trigsimp)

    return gamma, Riemann, Ricci, R, K