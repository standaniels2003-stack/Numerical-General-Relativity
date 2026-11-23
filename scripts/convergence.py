"""
RK4 self-check: the energy error on an eccentric Schwarzschild orbit must shrink like h^4.

What it does:
    1. Starts a massive particle at apoapsis r = 12M of a bound eccentric orbit
       (it swings between r ≈ 8.8M and 12M) in Schwarzschild with M = 1.
    2. Integrates it for 3.5 radial periods, so every run ends at pericentre, with
       step sizes h, h/2, h/4, ... (each a whole number of steps).
    3. Takes the final relative energy error |E(τ_end)/E(0) - 1| of each run.
    4. Fits a straight line through log(error) against log(h):
       np.polyfit(np.log(hs), np.log(errors), 1). RK4 is fourth order, so the
       slope must be close to 4. If it is not, something in the geodesic
       right-hand side or the integrator is wrong.
    5. Saves the log-log plot to figures/rk4_convergence.png and exits with
       status 1 if the slope is off by more than 0.3.

Why end at pericentre:
    The energy error oscillates with the orbit: it builds up near pericentre,
    where the curvature is strongest, and mostly cancels again on the way back
    out. Right after a whole number of radial periods (back at apoapsis) the
    h^4 part has cancelled and the error drops to order h^5, so you would measure
    a slope of about 5. At a random end time the error can be near a zero
    crossing, which makes the points scatter. At pericentre the h^4 term is at
    its largest, so the points lie on a clean line.

Run from the project root:
    python scripts/convergence.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")                     # only save the figure, never open a window
import matplotlib.pyplot as plt
import numpy as np

from grsim import Schwarzschild
from grsim.integrator import geodesic_rhs, integrate

M = 1.0
R_APO = 12.0                # start (and outer turning point) of the orbit
FACTOR = 0.95               # u^φ relative to the circular orbit at R_APO
N_PERIODS = 3.5             # end at pericentre, after 3 full radial periods
H0, N_HALVINGS = 4.0, 6     # step sizes H0, H0/2, ..., H0/2^5 (rounded to whole steps)
TOLERANCE = 0.3             # |slope - 4| allowed

FIGURE = Path(__file__).resolve().parents[1] / "figures" / "rk4_convergence.png"


def eccentric_orbit_start(bh):
    """
    Starting state at apoapsis R_APO, with u^φ = FACTOR times the circular-orbit value.

    Returns: the state (t, r, θ, φ, u^t, u^r, u^θ, u^φ) in the equatorial plane,
             with u^r = 0 and u^t chosen so that g(u, u) = -1.
    """
    uphi = FACTOR * np.sqrt(M / R_APO**3) / np.sqrt(1 - 3 * M / R_APO)
    g = bh.g((0.0, R_APO, np.pi / 2, 0.0))
    ut = np.sqrt((1 + g[3, 3] * uphi**2) / -g[0, 0])
    return np.array([0.0, R_APO, np.pi / 2, 0.0,   ut, 0.0, 0.0, uphi])


def radial_period(E, L):
    """
    Proper time from apoapsis to apoapsis, computed exactly (not from the integrator).

    What happens:
        (dr/dτ)² = E² - (1 - 2M/r)(1 + L²/r²) = (1 - E²)(r_a - r)(r - r_p)(r - r_3) / r³,
        with r_a > r_p > r_3 the roots of (1 - E²) r³ - 2M r² + L² r - 2M L² = 0.
        Substituting r = (r_a + r_p)/2 + (r_a - r_p)/2 cos χ removes the 1/sqrt
        singularities at the turning points, leaving the smooth integral
        T_r = 2 ∫_0^π sqrt(r³ / ((1 - E²)(r - r_3))) dχ.

    Returns: (T_r, r_p): the radial period and the pericentre radius.
    """
    r3, rp, ra = np.sort(np.roots([1 - E**2, -2 * M, L**2, -2 * M * L**2]).real)
    chi = np.linspace(0.0, np.pi, 20001)
    r = (ra + rp) / 2 + (ra - rp) / 2 * np.cos(chi)
    return 2 * np.trapezoid(np.sqrt(r**3 / ((1 - E**2) * (r - r3))), chi), rp


def main():
    bh = Schwarzschild(M=M)
    y0 = eccentric_orbit_start(bh)
    E0 = bh.energy(y0[:4], y0[4:])
    L0 = bh.angular_momentum(y0[:4], y0[4:])
    T_r, r_peri = radial_period(E0, L0)
    tau_end = N_PERIODS * T_r

    n0 = int(np.ceil(tau_end / H0))                       # whole number of steps
    hs = np.array([tau_end / (n0 * 2**k) for k in range(N_HALVINGS)])
    errors = []
    print(f"Eccentric orbit, E = {E0:.6f}, L = {L0:.6f}, integrated to τ = {tau_end:.2f} (pericentre)\n")
    print(f"{'h':>9}  {'steps':>6}  {'|E/E0 - 1|':>11}  {'ratio':>6}")
    for h in hs:
        taus, ys = integrate(geodesic_rhs, y0, 0.0, tau_end, h, bh)
        errors.append(abs(bh.energy(ys[-1, :4], ys[-1, 4:]) / E0 - 1))
        ratio = f"{errors[-2] / errors[-1]:6.1f}" if len(errors) > 1 else ""
        print(f"{h:9.4f}  {len(ys) - 1:6d}  {errors[-1]:11.3e}  {ratio:>6}")
    errors = np.array(errors)

    slope, intercept = np.polyfit(np.log(hs), np.log(errors), 1)
    ok = abs(slope - 4) <= TOLERANCE
    print(f"\nfitted slope = {slope:.3f}  (RK4: 4, halving h should divide the error by 2^4 = 16)")
    print("OK: fourth-order convergence" if ok else f"FAIL: slope is not within {TOLERANCE} of 4, look for a bug")

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", color="#e4e3df", lw=0.8)
    hh = np.geomspace(hs.min() / 1.3, hs.max() * 1.3, 100)
    ax.loglog(hh, np.exp(intercept) * hh**slope, color="#eb6834", lw=1.5, ls="--",
              label=f"fit: slope {slope:.2f}")
    ax.loglog(hs, errors, "o", ms=8, color="#2a78d6", mec="white", mew=2, zorder=3,
              label=r"final $|E/E_0 - 1|$")
    ax.set(xlabel="step size h / M", ylabel="relative energy error at the end",
           title="RK4 convergence on an eccentric Schwarzschild orbit")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.97, 0.05, f"r between {r_peri:.1f}M and {R_APO:.0f}M, τ = {tau_end:.0f}M",
            transform=ax.transAxes, ha="right", fontsize=8, color="#8a897f")
    fig.tight_layout()
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"figure saved to {FIGURE}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
