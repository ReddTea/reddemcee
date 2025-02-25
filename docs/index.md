# Reddemcee

Adaptive Parallel Tempering MCMC Ensemble Sampler, made for the exoplanet finder algorithm [`EMPEROR`](https://github.com/ReddTea/astroemperor/). This sampler works as a stand-alone program, so the community might find it useful.


## Overview
`reddemcee` is an Adaptive Parallel Tempering MCMC implementation based on the excellent [emcee](https://arxiv.org/abs/1202.3665) code, and a modified version of the [Vousden et al. implementation](https://arxiv.org/abs/1501.05823).

It's coded in such a way that minimal differences in input are required compared to emcee (v. 3.1.3).
Make sure to check reddemcee's [documentation](https://reddemcee.readthedocs.io/en/latest) !


## Quick Install

### Dependencies

This code makes use of:
  - numpy
  - tqdm (https://pypi.python.org/pypi/tqdm)
  - emcee (https://github.com/dfm/emcee)

### Pip
In the console type
```sh
pip install reddemcee
```

### From Source
In the console type
```sh
git clone https://github.com/ReddTea/reddemcee
cd reddemcee
python -m pip install -e .
```

## Quick usage
```python
import numpy as np
import reddemcee

def log_like(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)

def log_prior(x):
    return 0.0

ndim = 2
ntemps, nwalkers, nsweeps, nsteps = 5, 50, 100, 2

ivar = 1. / np.random.rand(ndim)
p0 = np.random.randn(ntemps, nwalkers, ndim)

sampler = reddemcee.PTSampler(nwalkers,
                             ndim,
                             log_like,
                             log_prior,
                             ntemps=ntemps,
                             loglargs=[ivar],
                             )
                             
sampler.run_mcmc(p0, nsweeps, nsteps)  # starting pos, nsweeps, nsteps
```

