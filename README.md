# Numerical-General-Relativity

A numerical relativity sandbox in Python: geodesic integration and ray tracing in Kerr spacetime, spherically symmetric scalar-field collapse, and a pipeline for BSSN black-hole evolutions and gravitational-wave extraction.

## What it does

- **Geodesics and imaging:** Symbolic curvature from any metric, geodesic integration in Schwarzschild and Kerr, and a backward ray tracer that renders lensed skies and thin accretion disks with Doppler and gravitational redshift.
- **1+1D collapse:** A finite-difference Einstein-Klein-Gordon solver in spherical symmetry with Kreiss-Oliger dissipation, built to probe the critical threshold of black-hole formation.
- **3+1 evolution:** Tools around BSSN evolutions generated with NRPy+: variable transforms, constraint monitoring, standard stability tests, and puncture black-hole runs.
- **Waveforms:** Extraction of $\Psi_4$, integration to the strain, and comparison against public SXS waveforms.

## Installation

```bash
git clone <repo-url>
cd grsim
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Repo Layout

```text
src/grsim/      Python package
tests/          pytest checks (curvature identities, convergence order, transform round trips)
notebooks/      exploratory work
nrpy_runs/      NRPy+ parameter files and run scripts
etk_runs/       Einstein Toolkit parameter files and run scripts
labbook/        development notes: what was tried, what broke, what was learned
figures/        generated figures
docs/           setup notes and background
```
