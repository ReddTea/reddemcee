```python
from IPython.display import Image, display
import numpy as np
import reddemcee
np.random.seed(1234)
```

# Parallelization
You can parallelize the sampler by using a pool, multiprocessing is recommended, but multiprocess and schwimmbad were tested as well.


```python
import multiprocessing as mp
```

You can check how many threads you have available by using:


```python
mp.cpu_count()
```




    24



We will build a likelihood that holds each thread for a set amount of time:


```python
import time

def loglike(theta):
    t = time.time() + np.random.uniform(0.005, 0.008)
    while True:
        if time.time() >= t:
            break
    return -0.5 * np.sum(theta**2)

def logprior(theta):
    return 0.
```

## Serial
This likelihood function will sleep for a random second fraction when called. We start by evaluating the performance in a serial initialization of the sampler:


```python
ndim_ = 2

setup = [2, 20, 40, 2]
ntemps, nwalkers, nsweeps, nsteps = setup

p0 = list(np.random.randn(ntemps, nwalkers, ndim_))
```


```python
sampler_s = reddemcee.PTSampler(nwalkers, ndim_,
                              loglike, logprior,
                              ntemps=ntemps,
                              )

start = time.time()
samp_s = sampler_s.run_mcmc(p0, nsweeps, nsteps)
time_serial = time.time() - start
```

    100%|██████████████| 80/80 [00:21<00:00,  3.77it/s]



```python
print(f'Serial took {time_serial:.1f} seconds')
```

    Serial took 21.2 seconds


## Parallel


```python
with mp.Pool(10) as mypool:
    sampler_p = reddemcee.PTSampler(nwalkers, ndim_,
                                  loglike, logprior,
                                  ntemps=ntemps,
                                  pool=mypool)
    start = time.time()
    samp_p = sampler_p.run_mcmc(p0, nsweeps, nsteps)
    time_parallel = time.time() - start
```

    100%|██████████████| 80/80 [00:02<00:00, 32.45it/s]



```python
print(f'Serial took {time_parallel:.1f} seconds')
```

    Serial took 2.5 seconds


Almost a tenth of the time!!
