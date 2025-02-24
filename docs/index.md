# Reddemcee

An Adaptative Parallel Tempering wrapper for emcee3 initially made for personal use, released to the community with the hope it might be helpful to someone.


## Overview
Reddemcee is a wrapper for the excellent MCMC python implementation [emcee](https://arxiv.org/abs/1202.3665), that contains an adaptative parallel tempering version of the sampler, roughly according to [Vousden et al.](https://arxiv.org/abs/1501.05823) implementation.
It's coded in such a way that minimal differences in input are required, respect to emcee3.


## Quick Install

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
```
import numpy as np
import reddemcee

def log_like(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)

def log_prior(x):
    return 0.0

ndim = 2
ntemps, nwalkers, nsweeps, nsteps = 5, 50, 100, 2

ivar = 1. / np.random.rand(ndim)
p0 = list(np.random.randn(ntemps, nwalkers, ndim))

sampler = reddemcee.PTSampler(nwalkers,
                             ndim,
                             log_like,
                             log_prior,
                             ntemps=ntemps,
                             logl_args=[ivar],
                             )
                             
sampler.run_mcmc(p0, nsweeps, nsteps)  # starting pos, nsweeps, nsteps
```