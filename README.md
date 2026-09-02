# bisetar

`bisetar` is a Python package for Bayesian inference and forecasting with the Bidirectional Self-Exciting Threshold Autoregressive (BiSETAR) model for insurance loss reserving.

The package accompanies the paper:

**Bayesian BiSETAR Models for Loss Reserving with Threshold Uncertainty**

## Overview

The package provides tools for:

- simulating BiSETAR loss triangles;
- estimating BiSETAR models using conditional least squares (CLS);
- sampling from the Bayesian posterior using discretised approximation (DA) and adaptive random-walk Metropolis (RW) methods;
- sampling regime-specific parameters conditional on threshold draws;
- producing lower-triangle forecasts;
- reproducing the empirical applications in the paper.

## Installation

The package can be installed directly from this repository using:

```bash
pip install git+https://github.com/wilson-ye-chen/bisetar
```

## Basic usage

### Simulating a BiSETAR triangle

```python
from bisetar.sim import sim_bisetar

n = 50

omg = [2, 2, 0, -5]
a1 = [0.1, 0.3, 0.5, 0.2]
b1 = [0.7, 0.6, 0.2, 0.3]
ss = [200, 165, 142, 87]

r1 = 0
r2 = 0

x = sim_bisetar(omg, a1, b1, ss, r1, r2, n)
```

### CLS estimation

```python
from bisetar.cls import BiSetarCls

cls = BiSetarCls(x)

r_cls = cls.learn_r()[0]
phi_cls = cls.learn_phi(r_cls)
```

### Bayesian posterior sampling

```python
from bisetar.mcmc import BiSetarMarginal

mpr = BiSetarMarginal(x)

# Discretised approximation sampler
r_da = mpr.sample_r(5000, method="grid")
phi_da = mpr.sample_phi(r_da)

# Adaptive random-walk Metropolis sampler
r_rw = mpr.sample_r(5000, method="mcmc")
phi_rw = mpr.sample_phi(r_rw)
```

### Lower-triangle forecasting

```python
from bisetar.frc import BiSetarForecast

theta = phi_rw

frc = BiSetarForecast(mpr.x, theta, nmc=len(theta))
lower_triangle_forecast = frc.forecast_all()
```

## Data

The empirical datasets used in the paper are included in the repository under:

```text
bisetar/data/
```

These include:

- the public NAIC `othliab` triangle;
- the public NAIC `prodliab` triangle;
- the aggregated CTP run-off triangle used in the empirical application.

The NAIC datasets were obtained from the Casualty Actuarial Society loss reserving data repository:

```text
https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p
```

## Repository structure

```text
bisetar/
  cls.py        # CLS estimation
  frc.py        # forecasting routines
  mcmc.py       # Bayesian posterior sampling
  sim.py        # BiSETAR simulation
  data/         # empirical datasets
```

## Citation

If you use this package, please cite the accompanying paper:

```text
Chen et al. Bayesian BiSETAR Models for Loss Reserving with Threshold Uncertainty.
```
