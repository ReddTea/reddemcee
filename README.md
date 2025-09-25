# ```reddemcee```
<!-- Light -->
![reddemcee light](docs/img/light/logo-full-crop.svg#gh-light-mode-only)
<!-- Dark -->
![reddemcee dark](docs/img/dark/logo-full-crop.svg#gh-dark-mode-only)

Adaptive Parallel Tempering MCMC Ensemble Sampler, made for the exoplanet finder algorithm [`EMPEROR`](https://github.com/ReddTea/astroemperor/). This sampler works as a stand-alone program, so the community might find it useful.


# Overview
[`reddemcee`](https://reddemcee.readthedocs.io/en/latest) is an Adaptive Parallel Tempering MCMC implementation based on the excellent [emcee](https://arxiv.org/abs/1202.3665) code, and a modified version of the [Vousden et al. implementation](https://arxiv.org/abs/1501.05823).

It's coded in such a way that minimal differences in input are required compared to emcee (v. 3.1.3).
Make sure to check reddemcee's [documentation](https://reddemcee.readthedocs.io/en/latest) !


# Dependencies

This code makes use of:

  - [`numpy`](https://numpy.org)
  - [`tqdm`](https://pypi.python.org/pypi/tqdm)
  - [`emcee`](https://github.com/dfm/emcee)
  - scipy

`tqdm` is used to display progress bars on the terminal.
`emcee` is used for calculating the autocorrelation times.

# Installation
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


# Usage

For a complete tutorial please refer to the [documentation's tutorial](https://reddemcee.readthedocs.io/en/latest/tutorials/quickstart/quickstart/) or the test file in the tests folder.

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
p0 = np.random.randn(10, nwalkers, ndim)
sampler = reddemcee.PTSampler(nwalkers,
                             ndim,
                             log_like,
                             log_prior,
                             ntemps=ntemps,
                             loglargs=[ivar],
                             )
                             
sampler.run_mcmc(p0, nsweeps, nsteps)  # starting coords, nsweeps, nsteps
```

# Configuring the Sampler
When setting up `PTSampler`, you can use the arguments:

| Arg         | Type    | Description |
|-------------|---------|---------------------|
| ntemps      | int     | The number of temperatures. If None, determined from betas.|
| betas       | list    | The inverse temperatures of the parallel chains.           |
| pool        | Pool    | A pool object for parallel processing.                     |
| loglargs    | list    | Positional arguments for the log-likelihood function.      |
| loglkwargs  | list    | Keyword arguments for the log-likelihood function.         |
| logpargs    | list    | Positional arguments for the log-prior function.           |
| logpkwargs  | list    | Keyword arguments for the log-prior function.              |
| backend     | Backend | A backend object for storing the chain.                    |
| smd_history | bool    | Whether to store swap mean distance history.               |
| tsw_history | bool    | Whether to store temperature swap history.                 |
| adapt_tau   | float   | Halflife of adaptation hyper-parameter.                    |
| adapt_nu    | float   | Rate of adaptation hyper-paramter.                         |
| adapt_mode  | 0-4     | Mode of adaptation.                                        |

The adaptation modes try to equalise the following quantity:

| Mode | Description           |
|------|-----------------------|
| 0    | Temperature Swap Rate |
| 1    | Swap Mean Distance    |
| 2    | Specific Heat         |
| 3    | dE/sig                |
| 4    | Thermodynamic Length  |


# Additional Functions
Additional functions on the sampler include:

| Function                          | Description                                 |
|-----------------------------------|---------------------------------------------|
| get_evidence_ti       | Calculates evidence by Thermodynamic Integration  |
| get_evidence_ss       | Calculates evidence by Stepping Stones            |
| get_evidence_hybrid   | Calculates evidence by Stepping Stones            |
| get_autocorr_time     | Auto-correlation time.                            |
| get_betas             | Returns beta history                              |
| get_chain             | Returns chain                                     |
| get_log_like          | Returns log likelihoods                           |
| get_logprob           | Returns log posteriors                            |

All these functions accept as arguments:

| Arg     | Description                       |
|---------|-----------------------------------|
| flat    | Flatten the walkers.              |
| thin    | Take one every *thin* samples.    |
| discard | Drop the first *discard* steps in samples.|

For example, the previous chain would have shape (5, 200, 50, 2), for
5 temperatures, 200 steps (nsweeps*nsteps), 50 walkers, and 2 dimensions.

```python
sampler.get_chain(discard=100)
```

Would return the samples with shape (5, 100, 50, 2). Dropping the first 100 for every walker.

```python
sampler.get_chain(discard=100, flat=True)
```

Would return the samples with shape (5, 5000, 2), 'linearising' the walkers.