# Reddemcee

An Adaptative Parallel Tempering wrapper for emcee 3 for personal use, which
someone in the community might find useful on it's own.

# Overview
Reddemcee is simply a wrapper for the excellent MCMC implementation [emcee](https://arxiv.org/abs/1202.3665),
that contains an adaptative parallel tempering version of the sampler, according to [Vousden et al. implementation](https://arxiv.org/abs/1501.05823).
It's coded in such a way that minimal differences in input are required, and it's
fully compatible with emcee (v. 3.1.3).

# Dependencies

This code makes use of:
  - Numpy
  - pandas
  - tqdm (https://pypi.python.org/pypi/tqdm)
  - emcee (https://github.com/dfm/emcee)

Most of them come with conda, if some are missing they can be easily installed with pip.

# Installation

In the console type in your work folder
```sh
pip install reddemcee
```

# Usage

Please refer to the test file in the tests folder.

```python
import numpy as np
import reddemcee

def log_like(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)

def log_prior(x):
    return 0.0

ndim, nwalkers = 5, 100
ntemps = 5
ivar = 1. / np.random.rand(ndim)
p0 = list(np.random.randn(10, nwalkers, ndim))
sampler = reddemcee.PTSampler(nwalkers,
                             ndim,
                             log_like,
                             log_prior,
                             ntemps=ntemps,
                             adaptative=True,
                             logl_args=[ivar],
                             )
                             
sampler.run_mcmc(p0, 100, 2)  # starting pos, nsweeps, nsteps
```

# Additional Options

ntemps
betas
pool
adaptative
config_adaptation_halflife rn: adaptations reduced by half at this time
config_adaptation_rate     rn: smaller, faster
moves
backend

## Stored
ratios
betas_history
betas_history_bool
ratios_history

## Funcs
thermodynamic_integration(self,
                          coef=3,
                          sampler_dict = {'flat':False,
                                          'discard':10})

get_Z(discard=1, coef=3, largo=100)
get_attr(x)
get_func(x)