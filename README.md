# Numerical General Relativity

A numerical relativity sandbox in Python. It starts from exact curvature and particle
orbits around black holes, and is growing towards ray tracing, gravitational collapse
and full 3+1 black-hole evolutions.

## What it does

- **Exact curvature:** Christoffel symbols, Riemann and Ricci tensors, and the Ricci
  and Kretschmann scalars of any metric written down in SymPy.
- **Spacetimes:** Minkowski, Schwarzschild and Kerr, turned into fast numerical
  functions; a new spacetime is just a new metric.
- **Geodesics:** an RK4 integrator for massive particles and light rays.
- **Diagnostics:** the normalization, energy and angular momentum along an orbit, to
  measure how accurate the integration is.
- **Orbit notebook:** experiment with circular, precessing, zoom-whirl and plunging
  orbits, light rays, and orbits around a spinning black hole.

## Installation

```bash
git clone https://github.com/standaniels2003-stack/Numerical-General-Relativity.git
cd Numerical-General-Relativity
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the tests with `pytest`, or open `notebooks/orbits.ipynb`.

## Repository layout

```text
Numerical-General-Relativity/
├── src/grsim/              Python package
│   ├── curvature.py        exact Christoffel symbols, Riemann, Ricci and scalars
│   ├── metric.py           Minkowski, Schwarzschild and Kerr spacetimes
│   ├── integrator.py       geodesic equation and RK4 integrator
│   └── diagnostics.py      conserved quantities along an orbit
├── tests/                  pytest checks
├── scripts/                standalone checks, e.g. the RK4 convergence plot
├── notebooks/              exploratory work
├── nrpy_runs/              NRPy+ parameter files and run scripts
├── etk_runs/               Einstein Toolkit parameter files and run scripts
├── labbook/                development notes: what was tried, what broke, what was learned
├── figures/                generated figures
├── docs/                   setup notes and background
├── pyproject.toml          package metadata and pytest settings
└── requirements.txt        dependencies
```
