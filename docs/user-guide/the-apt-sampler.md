# The APT Sampler

```
class PTSampler(object):
    def __init__(self, nwalkers, ndim, log_likelihood_fn, log_prior_fn,
                 pool=None, moves=None, backend=None, vectorize=False,
                 blobs_dtype=None, parameter_names=None, logl_args=None,
                 logl_kwargs=None, logp_args=None, logp_kwargs=None,
                 ntemps=None, betas=None, adaptative=True, a=None,
                 postargs=None, threads=None, live_dangerously=None,
                 runtime_sortingfn=None,
                 config_adaptation_halflife=1000,
                 config_adaptation_rate=100,
                 config_adaptation_decay=0):

        
        Initializes the sampler with the specified parameters for a Markov Chain Monte Carlo (MCMC) process. This setup includes defining the number of walkers, dimensions, likelihood and prior functions, and various configuration options for the sampling process.
```

Args:
    nwalkers (int): The number of walkers in the ensemble.

    ndim (int): The number of dimensions of the parameter space.

    log_likelihood_fn (callable): The log-likelihood function to evaluate the likelihood of the parameters.

    log_prior_fn (callable): The log-prior function to evaluate the prior distribution of the parameters.

    pool (optional): A multiprocessing pool for parallel computation.
    
    moves (optional): A list of move functions for the walkers.
    
    backend (optional): A backend for storing samples.
    
    vectorize (bool, optional): Whether to vectorize the likelihood and prior functions.
    
    blobs_dtype (optional): Data type for the blobs returned by the sampler.
    
    parameter_names (optional): Names of the parameters being sampled.
    
    logl_args (optional): Additional arguments for the log-likelihood function.
    
    logl_kwargs (optional): Additional keyword arguments for the log-likelihood function.
    
    logp_args (optional): Additional arguments for the log-prior function.
    
    logp_kwargs (optional): Additional keyword arguments for the log-prior function.
    
    ntemps (int, optional): The number of temperatures for parallel tempering.
    
    betas (optional): The temperature ladder for parallel tempering.
    
    adaptative (bool, optional): Whether to enable adaptive configuration.

    config_adaptation_halflife (int, optional): The halflife for adaptation configuration.
    
    config_adaptation_rate (int, optional): The rate of adaptation configuration.
    
    config_adaptation_decay (int, optional): The decay option for adaptation configuration.

Deprecated Args:

    a (optional): Deprecated argument, use 'moves' instead.
    
    postargs (optional): Deprecated argument.
             
    threads (optional): Deprecated argument.
                
    live_dangerously (optional): Deprecated argument.
               
    runtime_sortingfn (optional): Deprecated argument.