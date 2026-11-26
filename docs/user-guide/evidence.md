# Evidence with MBAR

`PTSampler.get_evidence_mbar()` combines samples from every temperature to
estimate the log evidence, `log Z`, with a Monte Carlo standard error (MCSE).
It uses saved log likelihoods and NumPy/SciPy, with no extra likelihood
evaluations or additional MBAR dependency. See the
[API reference](../api.md) for all options.

MBAR fits normalizing-constant ratios directly, avoiding thermodynamic
integration's temperature-grid quadrature approximation. Its accuracy still
depends on equilibrium sampling and overlap between temperatures. See
[Shirts and Chodera (2008)](https://doi.org/10.1063/1.2978177).

## Adapt, freeze, and retain production samples

For absolute evidence, provide a proper, normalized prior and include the
likelihood's normalization constants. The temperature ladder must include
`beta=1` for the posterior and `beta=0` for the prior. Retained log likelihoods
must be finite.

Adapt the ladder during warmup, freeze it, and allow it to settle before
collecting evidence samples. This example uses a normalized Gaussian prior
with covariance `4 I` and a normalized Gaussian likelihood with covariance `I`:

```python
import numpy as np
from reddemcee import PTSampler

np.random.seed(2026)
ndim, nwalkers = 2, 24
prior_variance = 4.0
betas = np.r_[np.geomspace(1.0, 1e-3, 11), 0.0]

def log_prior(theta):
    return -0.5 * (
        np.dot(theta, theta) / prior_variance
        + ndim * np.log(2 * np.pi * prior_variance)
    )

def log_like(theta):
    return -0.5 * (np.dot(theta, theta) + ndim * np.log(2 * np.pi))

p0 = np.random.normal(
    scale=np.sqrt(prior_variance),
    size=(len(betas), nwalkers, ndim),
)
sampler = PTSampler(
    nwalkers, ndim, log_like, log_prior,
    betas=betas, adapt_mode="SAR",
)

# Warmup: learn the temperature ladder.
state = sampler.run_mcmc(p0, nsweeps=500, nsteps=1)

# Freeze and re-equilibrate at the learned temperatures.
sampler.select_adjustment("NONE")
state = sampler.run_mcmc(state, nsweeps=200, nsteps=1)

# Clear warmup/settling storage; retain the state explicitly across reset().
sampler.reset()
sampler.run_mcmc(state, nsweeps=2000, nsteps=1)

logz, mcse, diagnostics = sampler.get_evidence_mbar(
    return_diagnostics=True,
)
logz_exact = -0.5 * ndim * np.log(2 * np.pi * (prior_variance + 1))
print("log Z:", logz, "MCSE:", mcse, "reference:", logz_exact)
print("overlap gap:", diagnostics["overlap_spectral_gap"])
```

The run lengths illustrate the workflow; choose adequate warmup and production
lengths using sampling diagnostics and independent runs. `reset()` clears the
stored draws and the implicit resume state, so pass the saved `state` when
starting production after a reset. Freezing is separate from resetting.

## Selecting retained draws

By default the sampler method returns `(logz, mcse)`. Set
`return_diagnostics=True` to receive `(logz, mcse, diagnostics)`.
`discard` and `thin` select the same stored rows from beta history and log
likelihoods:

```python
logz, mcse = sampler.get_evidence_mbar(discard=100, thin=2)
```

The example uses `nsteps=1`, so each stored row is one complete PT sweep.
`discard` does not stop adaptation: all selected rows must already come from
a frozen ladder. Thinning is optional; the MCSE calculation accounts for
serial dependence in the retained series. Keep at least three retained rows,
and use substantially longer series for useful uncertainty estimates.

For saved arrays, the same estimator is available directly:

```python
from reddemcee import get_mbar

B = sampler.get_betas(discard=100, thin=2)
L = sampler.get_log_like(discard=100, thin=2)
logz, mcse, diagnostics = get_mbar(B, L, return_diagnostics=True)
```

`L` has shape `(ntemps, nrows, nwalkers)`. `B` has shape `(ntemps, nrows)`;
a one-dimensional fixed ladder of length `ntemps` is also accepted. Supply
log likelihoods, rather than tempered log posterior values. Both ascending and
descending ladders are accepted; duplicate temperatures or a drifting retained
ladder are rejected. `fixed_ladder=False` does not permit adaptive samples.

## Diagnostics and uncertainty

| Diagnostic | Meaning |
| --- | --- |
| `absolute_evidence` | Whether the selected ladder contains both zero and one. |
| `converged` and `score_residual` | Convergence of the numerical normalizing-constant fit. |
| `log_normalizers` | Fitted log normalizers in ascending-beta order, relative to the hottest selected state. |
| `overlap_matrix` and `overlap_spectral_gap` | Observed connectivity among the sampled temperature states. |
| `mcse_reliable` | Whether the observed overlap passes the requested threshold. |

The default MCSE uses overlapping batch means on a synchronized influence
series, retaining covariance among temperatures and walkers. `mode="spe"`
selects a spectral estimate instead. `batch_size` sets the batch length for
`mode="obm"`, or the lag bandwidth for `mode="spe"`; compare sensible lengths
when checking uncertainty.

An overlap spectral gap below `overlap_threshold` (default `1e-3`) produces a
warning, `mcse=np.inf`, and `mcse_reliable=False`. Numerical nonidentifiability
raises `ValueError`, and a failed solve raises `RuntimeError`. Collect more
equilibrated samples or add bridging temperatures before relying on such a
result. A converged fit and finite MCSE cannot detect an unvisited mode or
establish that the sampler equilibrated. MCSE describes sampling uncertainty;
it is not a guarantee of 95% coverage on a short run.

## Partial normalizing-constant ratios

Without both beta endpoints, the method raises `ValueError` by default.
Use `allow_partial=True` when a ratio is intended. For example, exclude the
prior state from the fitted ladder:

```python
log_ratio, mcse_ratio, ratio_diagnostics = sampler.get_evidence_mbar(
    ba=1, allow_partial=True, return_diagnostics=True,
)
print(ratio_diagnostics["beta_min"], ratio_diagnostics["beta_max"])
assert not ratio_diagnostics["absolute_evidence"]
```

`ba` and `bb` are inclusive indices after ordering beta **ascending**, regardless
of the stored ladder orientation. Here `ba=1` starts at the first positive beta
and `bb=None` retains beta one. The result is
`log Z(beta_max) - log Z(beta_min)`, not an absolute log evidence.
