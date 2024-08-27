# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
from copy import deepcopy

import emcee
import numpy as np
from emcee.autocorr import integrated_time
from emcee.utils import deprecation_warning
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

from .reddwrappers import PTWrapper

temp_table = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                      2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                      2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                      1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                      1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                      1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                      1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                      1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                      1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                      1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                      1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                      1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                      1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                      1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                      1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                      1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                      1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                      1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                      1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                      1.26579, 1.26424, 1.26271, 1.26121,
                      1.25973])

dmax = temp_table.shape[0]


def set_temp_ladder(ntemps, ndims, temp_table=temp_table):
    tstep = temp_table[ndims-1]
    if ndims > dmax:
        # An approximation to the temperature step at large dimension
        tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndims)

    return np.exp(np.linspace(0, -(ntemps-1)*np.log(tstep), ntemps))


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
        """
        Initializes the sampler with the specified parameters for a Markov Chain Monte Carlo (MCMC) process. This setup includes defining the number of walkers, dimensions, likelihood and prior functions, and various configuration options for the sampling process.

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
            a (optional): Deprecated argument, use 'moves' instead.
            postargs (optional): Deprecated argument.
            threads (optional): Deprecated argument.
            live_dangerously (optional): Deprecated argument.
            runtime_sortingfn (optional): Deprecated argument.
            config_adaptation_halflife (int, optional): The halflife for adaptation configuration.
            config_adaptation_rate (int, optional): The rate of adaptation configuration.
            config_adaptation_decay (int, optional): The decay option for adaptation configuration.

        Raises:
            DeprecationWarning: If deprecated arguments are used.

        """
        # Default arguments
        # self.ll_args_ = logl_args if logl_args is not None else [] # safer but verbosy
        self.ll_args_ = logl_args or []
        self.lp_args_ = logp_args or []

        self.ll_kwargs_ = logl_kwargs or {}
        self.lp_kwargs_ = logp_kwargs or {}
        
        ######################################################
        # Warn about deprecated arguments
        deprecated_args = [postargs, threads, runtime_sortingfn, live_dangerously]
        deprecated_args_str = ['postargs', 'threads', 'runtime_sortingfn', 'live_dangerously']

        if a is not None:
            deprecation_warning("The 'a' argument is deprecated, use 'moves' instead")

        for arg, arg_str in zip(deprecated_args, deprecated_args_str):
            if arg is not None:
                deprecation_warning(f"The '{arg_str}' argument is deprecated")
        

        # Beta ladder initialization
        self.ntemps = ntemps or 5
        self.betas = betas if betas is not None else set_temp_ladder(self.ntemps, ndim)
        #####################################################
        # Initialize instance variables
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.lp_ = log_prior_fn
        self.ll_ = log_likelihood_fn

        self.n_swap_accept = np.zeros(ntemps-1)
        self.time0 = 0
        self.nglobal = 0
        self.ratios = None
        self.ratios_history = np.array([])
        self.betas_history = np.array([])

        self.pool = pool
        self.betas_history_bool = True
        self.z_num_grid = 10001
        self.z_num_simulations = 1000
        self.adaptative = adaptative
        self.config_adaptation_halflife = config_adaptation_halflife  # adaptations reduced by half at this time
        self.config_adaptation_rate = config_adaptation_rate  # smaller, faster
        self.select_decay(option=config_adaptation_decay)
        # Default values for lists
        #default_list = lambda lst, size, default: [default for _ in range(size)] if lst is None else lst
        self.vectorize = [False for _ in range(self.ntemps)] if vectorize is False else vectorize
        #self.moves = default_list(moves, self.ntemps, None)
        #self.blobs_dtype = default_list(blobs_dtype, self.ntemps, None)
        #self.backend = default_list(backend, self.ntemps, None)

        self.moves = moves if moves is not None else [None] * self.ntemps
        self.blobs_dtype = blobs_dtype if blobs_dtype is not None else [None] * self.ntemps
        self.backend = backend if backend is not None else [None] * self.ntemps


        ## BACKEND
        #self.backend = Backend() if backend is None else backend
        # Deal with re-used backends
        # Check the backend shape
        #########
        # Probability function wrappers
        self.my_probs_fn = np.array([PTWrapper(self.ll_, self.lp_, b, loglargs=self.ll_args_,
                                  logpargs=self.lp_args_, loglkwargs=self.ll_kwargs_,
                                  logpkwargs=self.lp_kwargs_) for b in self.betas])

        self.sampler = np.array([emcee.EnsembleSampler(self.nwalkers, self.ndim,
                        self.my_probs_fn[t], pool=self.pool,
                        moves=self.moves[t], backend=self.backend[t],
                        vectorize=self.vectorize[t], blobs_dtype=self.blobs_dtype[t],
                        ) for t in range(self.ntemps)])


    def sample(self,
               initial_state,
               iterations=1,
               tune=False,
               skip_initial_state_check=False,
               thin_by=1,
               thin=None,
               store=True,
               progress=False,
               progress_kwargs=None):

        self.samp = initial_state

        for t in range(self.ntemps):
            sampler_t = self[t]
            #samp_t = self.samp[t]
            for self.samp[t] in sampler_t.sample(self.samp[t],
                                               iterations=iterations,
                                               tune=tune,
                                               skip_initial_state_check=skip_initial_state_check,
                                               thin_by=thin_by,
                                               thin=thin,
                                               store=store,
                                               progress=progress,
                                               progress_kwargs=progress_kwargs):
                pass


        if self.betas_history_bool:
            self.betas_history = np.append(self.betas_history, self.betas)
        self.temp_swaps_()

        if self.adaptative:
            self.ladder_adjustment()

        yield self.samp


    def run_mcmc(self, initial_state, nsweeps, nsteps, progress=True):
        if initial_state is None:
            print('Initial state is none')
            if self.__previous_state[0] is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called.")
            initial_state = self.__previous_state

        results = None
        

        pbar = tqdm(total=nsteps*nsweeps, disable=not progress)
        for _ in range(nsweeps):
            for results in self.sample(initial_state, iterations=nsteps):
                pbar.update(nsteps)

        pbar.close()

        return results


    def temp_swaps_(self):
        dbetas = self.betas[:-1] - self.betas[1:]
        for t in range(self.ntemps-1, 0, -1):
            dbeta = dbetas[t-1]

            ll1 = self.samp[t].blobs  # hot
            ll2 = self.samp[t-1].blobs  # cold

            raccept = np.log(np.random.uniform(size=self.nwalkers))
            paccept = dbeta * (ll1 - ll2)

            asel = paccept > raccept

            self.n_swap_accept[t-1] = np.sum(asel)

            self.samp[t].coords[asel], self.samp[t-1].coords[asel] = self.samp[t-1].coords[asel], self.samp[t].coords[asel]
            self.samp[t].log_prob[asel], self.samp[t-1].log_prob[asel] = self.samp[t-1].log_prob[asel] - dbeta*ll2[asel], self.samp[t].log_prob[asel] + dbeta*ll1[asel]
            self.samp[t].blobs[asel], self.samp[t-1].blobs[asel] = ll2[asel], ll1[asel]
            
        self.ratios = self.n_swap_accept / self.nwalkers
        self.ratios_history = np.append(self.ratios_history, self.ratios[::-1])
        

    def ladder_adjustment(self):
        # sourcery skip: remove-redundant-pass
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.
        """

        betas = self.betas.copy()
        time = self.time0 + self[0].iteration

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.config_adaptation_halflife / (time + self.config_adaptation_halflife)
        
        decay = self.decay_fn(decay)
        # decay 1
        #decay *= 1/(np.exp(-np.std(betas)))

        # decay 2
        #decay *= 1/(1-np.std(betas))

        kappa = decay / self.config_adaptation_rate

        # Construct temperature adjustments.
        dSs = kappa * (self.ratios[:-1] - self.ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        dbetas = betas - self.betas
        self.betas = betas
        for t in range(self.ntemps-1, 0, -1):
            self.my_probs_fn[t].beta = self.betas[t]
            self.samp[t].log_prob += self.samp[t].blobs * dbetas[t]
        pass


    def calc_decay0(self, decay):
        return decay

    def calc_decay1(self, decay):
        return decay / (np.exp(-np.std(self.ratios)))

    def calc_decay2(self, decay):
        #ratios = 
        #afc = abs(np.mean(self.ratios)-self.ratios[:-1])/np.std(self.ratios)
        #return decay * afc
        return decay / (1-np.std(self.ratios))
    
    def select_decay(self, option=0):
        self.decay_fn = getattr(self, f'calc_decay{option}')

    def thermodynamic_integration_classic(self, discard=1):
        logls0 = self.get_logls(flat=True, discard=discard)
        logls = np.mean(logls0, axis=1)

        if self.betas[-1] != 0:
            betas1 = np.concatenate((self.betas, [0]))
            betas2 = np.concatenate((self.betas[::2], [0]))
            logls1 = np.concatenate((logls, [logls[-1]]))
            logls2 = np.concatenate((logls[::2], [logls[-1]]))
        else:
            betas1 = self.betas
            betas2 = np.concatenate((self.betas[:-1:2], [0]))
            logls1 = logls
            logls2 = np.concatenate((logls1[:-1:2], [logls1[-1]]))

        logZ1 = -np.trapz(logls1, betas1)
        logZ2 = -np.trapz(logls2, betas2)

        return logZ1, np.abs(logZ1 - logZ2)


    def thermodynamic_integration(self, discard=1):
        '''
        Method to get Z and its error.
        '''

        # SORT BETAS AND LOGLS
        x = deepcopy(self.betas)[::-1]
        logls0 = self.get_logls(flat=True, discard=discard)
        logls1 = deepcopy(logls0)[::-1]

        # if hot chain beta !=0, add it
        if x[0] != 0:
            x = np.concatenate(([0], x))
            logls1 = np.concatenate(([logls1[0]], logls1))

        
        # get logL means
        y = np.mean(logls1, axis=1)

        # support variables
        n = np.array([len(l) for l in logls1])
        tau = np.array([integrated_time(logls1[i], c=5, tol=5, quiet=False)
                      for i in range(len(y))]).flatten()

        neff = n/tau

        d1 = np.diff(x)
        stdyn = np.std(logls1, axis=1, ddof=1) / np.sqrt(neff)

        # calculate Z
        num_grid = self.z_num_grid
        num_simulations = self.z_num_simulations
        xnew = np.linspace(min(x), max(x), num_grid)

        # Interpolated values storage
        interpolated_values = np.zeros((num_simulations, num_grid))

        for i in range(num_simulations):
            # Perturb data with random noise based on errors
            y_noisy = y + np.random.normal(0, stdyn)

            # Create PchipInterpolator object with noisy data
            pchip = PchipInterpolator(x, y_noisy)

            # Interpolate over fine grid and store results
            interpolated_values[i, :] = pchip(xnew)

        mean_interpolated = np.mean(interpolated_values, axis=0)
        std_interpolated = np.std(interpolated_values, axis=0)

        integral_trapz = np.trapz(mean_interpolated, xnew)

        z_classic, zerr_classic = self.thermodynamic_integration_classic(discard=discard)
        err_disc = abs(z_classic - integral_trapz)

        err_samp = 1/num_grid**2 * (1/4*std_interpolated[0]**2 +
                                    np.sum(std_interpolated[1:-1]**2) +
                                    1/4*std_interpolated[-1]**2)
        err_samp = np.sqrt(err_samp)
        err = np.sqrt(err_disc**2+err_samp**2)

        return integral_trapz, err, err_disc, err_samp


    def reset(self):
        self.time0 += self[0].iteration
        
        for s in self:
            s.reset()

        self.ratios = None
        self.betas_history = [[] for _ in range(self.ntemps)]
        self.ratios_history = np.array([])
        

    def get_attr(self, x):
        return np.array([getattr(sampler_instance, x) for sampler_instance in self])


    def get_func(self, x, kwargs=None):
        if kwargs is None:
            kwargs = {}
        return np.array([getattr(sampler_instance, x)(**kwargs) for sampler_instance in self])


    def get_chains(self, **kwargs):
        return self.get_func('get_chain', kwargs=kwargs)


    def get_logls(self, **kwargs):
        return self.get_func('get_blobs', kwargs=kwargs)

    
    def get_log_probs(self, **kwargs):
        return self.get_func('get_log_prob', kwargs=kwargs)


    def __str__(self):
        return 'My sampler, ntemps = %i' % self.ntemps


    def __getitem__(self, n):
        return self.sampler[n]


    def __setitem__(self, n, thing):
        self.sampler[n] = thing


    pass
