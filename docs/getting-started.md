# Getting Started with Reddemcee

An introductory tutorial!

---

## Installation

To install reddemcee, run the following command from the command line:

```bash
pip3 install reddemcee
```

For more details, see the [Installation Guide].

## Why reddemcee?

- Adaptative Parallel Tempering MCMC, or APTMCMC for the friends
- Easy to adapt emcee codes to this
- Cool

### Usage example

We will start with a simple 2D gaussian shell evaluation using emcee:
- Widely used in the literature (vg, dynesty and multinest papers, as well as Vousden and Lartillot&philippe)
- It is analytically tractable.
- Not directly relevant to the Keplerian problem


#### Likelihood
The likelihood is given by:

$$ p(\vec{\theta}) = \sum_{i=1}^n \frac{1}{\sqrt{2\pi w^2}} \exp{\left( -\frac{(|\vec{\theta} - \vec{c_i}| - r)^2}{2w^2} \right)} $$

where $n$ are the number of dimensions, $r$ corresponds to the radius, $w$ the width and $\vec{c_i}$ to the constant vectors describing the centre of the peaks.

#### Constants
In the following section we will define the relevant constants to the problem

```python
import numpy as np
import reddemcee

ndim_ = 2  # n dimensions
r_ = 2.  # radius
w_ = 0.1  # width
hard_limit = 6  # hard search boundary

limits_ = [-hard_limit,  hard_limit]
c1_ = np.zeros(ndim_)
c1_[0] = -3.5
c2_ = np.zeros(ndim_)
c2_[0] = 3.5
const_ = np.log(1. / np.sqrt(2. * np.pi * w_**2))  # normalization constant
```

#### Important functions
We will define the functions for prior and likelihood, as well as a unit boundary affine transformation for dynesty, which should be useful for benchmarking.

```python
def logcirc(theta, c):
    # log-likelihood of a single shell
    d = np.sqrt(np.sum((theta - c)**2, axis=-1))  # |theta - c|
    return const_ - (d - r_)**2 / (2. * w_**2)


def loglike(theta):
    # log-likelihood of two shells
    return np.logaddexp(logcirc(theta, c1_), logcirc(theta, c2_))


def logprior(theta):
    # prior for our parameters
    lp = 0.
    for i in range(ndim_):
        if  theta[i] <= limits_[0] or limits_[1] <= theta[i]:
            return -np.inf
    return lp
```

## reddemcee
```python
setup = [4, 100, 200, 2]
ntemps, nwalkers, nsweeps, nsteps = setup
p0 = np.random.uniform(limits_[0], limits_[1], [ntemps, nwalkers, ndim_])
```

```python
sampler = reddemcee.PTSampler(nwalkers, ndim_, loglike, logprior,
                              ntemps=ntemps)
    
sampler.run_mcmc(p0, nsweeps, nsteps, progress=True)
```

## Getting help
See the [User Guide] for more complete documentation of all of reddemcee' features.