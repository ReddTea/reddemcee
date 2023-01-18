# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.3.2
# date 17 jan 2023

__version__ = '0.3.3'
__name__ = 'reddemcee'
__all__ = ['PTSampler']

import numpy as np
import emcee
from emcee.utils import deprecation_warning

from tqdm import tqdm

from typing import Dict, List, Optional, Union

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
    # returns betas
    dmax = temp_table.shape[0]
    if ndims > dmax:
        # An approximation to the temperature step at large dimension
        tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndims)
    else:
        tstep = temp_table[ndims-1]

    return np.exp(np.linspace(0, -(ntemps-1)*np.log(tstep), ntemps))


class PTSampler(object):
    def __init__(self, nwalkers, ndim, log_likelihood_fn, log_prior_fn, pool=None, moves=None, backend=None, vectorize=False, blobs_dtype=None, parameter_names: Optional[Union[Dict[str, int], List[str]]] = None, logl_args=None, logl_kwargs=None, logp_args=None, logp_kwargs=None, ntemps=None, betas=None, adaptative=True, a=None, postargs=None, threads=None, live_dangerously=None, runtime_sortingfn=None):
        # nsweeps = None  # ? should add this?
        if logl_args is None:
            logl_args = []
        if logl_kwargs is None:
            logl_kwargs = {}
        if logp_args is None:
            logp_args = []
        if logp_kwargs is None:
            logp_kwargs = {}
        ######################################################
        # Warn about deprecated arguments
        deprecated_args = [postargs, threads, runtime_sortingfn, live_dangerously]
        deprecated_args_str = ['postargs', 'threads', 'runtime_sortingfn', 'live_dangerously']

        if a is not None:
            deprecation_warning("The 'a' argument is deprecated, use 'moves' instead")
        for i in range(len(deprecated_args)):
            if deprecated_args[i] is not None:
                deprecation_warning(f"The '{deprecated_args_str[i]}' argument is deprecated")
        ######################################################
        # Parse the move schedule

        # beta ladder
        if ntemps:
            self.betas = betas or set_temp_ladder(ntemps, ndim)
        else:
            ntemps = 5
            self.betas = set_temp_ladder(ntemps, ndim)
        self.ntemps = ntemps
        #####################################################
        # Name things
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.lp_ = log_prior_fn
        self.ll_ = log_likelihood_fn

        self.lp_args_ = logp_args
        self.ll_args_ = logl_args

        self.n_swap_accept = np.zeros(ntemps-1)

        self.pool = pool
        self.vectorize = vectorize
        self.blobs_dtype = blobs_dtype
        self.moves = moves
        self.backend = backend
        self.ratios = None

        if self.vectorize is False:
            self.vectorize = [False for _ in range(self.ntemps)]

        if self.moves is None:
            self.moves = [None for _ in range(self.ntemps)]
        if self.blobs_dtype is None:
            self.blobs_dtype = [None for _ in range(self.ntemps)]
        if self.backend is None:
            self.backend = [None for _ in range(self.ntemps)]

        self.my_probs_fn = [] #list of functions

        self.adaptative = adaptative
        self.config_adaptation_lag = 10000
        self.config_adaptation_time = 100


        ## BACKEND
        #self.backend = Backend() if backend is None else backend

        # Deal with re-used backends
        # Check the backend shape
        #########
        # NAMED PARAMETERS?
        self.my_probs_fn.extend(
            PTWrapper(
                self.ll_,
                self.lp_,
                b,
                loglargs=self.ll_args_,
                logpargs=self.lp_args_,
                loglkwargs={},
                logpkwargs={},
            )
            for b in self.betas
        )
        self.sampler = [emcee.EnsembleSampler(self.nwalkers, self.ndim,
                        self.my_probs_fn[t], pool=self.pool,
                        moves=self.moves[t], backend=self.backend[t],
                        vectorize=self.vectorize[t], blobs_dtype=self.blobs_dtype[t],
                        ) for t in range(self.ntemps)]

    def __str__(self):
        return 'My sampler, ntemps = %i' % self.ntemps


    def __getitem__(self, n):
        return self.sampler[n]


    def __setitem__(self, n, thing):
        self.sampler[n] = thing


    def sample(
        self,
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
            for self.samp[t] in self[t].sample(self.samp[t],
                                               iterations=iterations,
                                               tune=tune,
                                               skip_initial_state_check=skip_initial_state_check,
                                               thin_by=thin_by,
                                               thin=thin,
                                               store=store,
                                               progress=progress,
                                               progress_kwargs=progress_kwargs):
               pass


        self.temp_swaps_()
        if self.adaptative:
            self.ladder_adjustment()


        yield self.samp


    def temp_swaps_(self):
        for t in range(self.ntemps-1, 0, -1):
            dbeta = self.betas[t-1] - self.betas[t]

            #iperm = np.ones(self.nwalkers, int)
            #i1perm = np.ones(self.nwalkers, int)

            ll1 = self.samp[t].blobs
            ll2 = self.samp[t-1].blobs

            raccept = np.log(np.random.uniform(0, 1, self.nwalkers))
            paccept = dbeta * (ll1 - ll2)

            asel = paccept > raccept

            self.n_swap_accept[t-1] = np.sum(asel)

            self.samp[t].coords[asel], self.samp[t-1].coords[asel] = self.samp[t-1].coords[asel], self.samp[t].coords[asel]
            self.samp[t].log_prob[asel], self.samp[t-1].log_prob[asel] = self.samp[t-1].log_prob[asel] - dbeta*ll2[asel], self.samp[t].log_prob[asel] + dbeta*ll1[asel]
            self.samp[t].blobs[asel], self.samp[t-1].blobs[asel] = self.samp[t-1].blobs[asel], self.samp[t].blobs[asel]
        self.ratios = self.n_swap_accept / self.nwalkers


    def ladder_adjustment(self):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.
        """

        betas = self.betas.copy()
        time = self[0].iteration

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.config_adaptation_lag / (time + self.config_adaptation_lag)
        kappa = decay / self.config_adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (self.ratios[:-1] - self.ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        # Mutate Ladder
        self.betas = betas
        # Mutate my_probs_fn
        for t in range(self.ntemps):
            self.my_probs_fn[t].beta = self.betas[t]


    def thermodynamic_integration(self):
        logls = np.mean(self.get_func('get_log_prob', kwargs={'flat':True}), axis=1)

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


    def run_mcmc(self, initial_state, nsteps, progress=True):
        if initial_state is None:
            print('Initial state is none')
            if self.__previous_state[0] is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called.")
            initial_state = self.__previous_state

        self.sampler = [emcee.EnsembleSampler(self.nwalkers, self.ndim,
                        self.my_probs_fn[t], pool=self.pool,
                        moves=self.moves[t], backend=self.backend[t],
                        vectorize=self.vectorize[t], blobs_dtype=self.blobs_dtype[t],
                        ) for t in range(self.ntemps)]

        results = None
        pbar = tqdm(total=nsteps, disable=not progress)
        for _ in range(nsteps):
            for results in self.sample(initial_state):
                pbar.update(1)

        pbar.close()

        return results


    def get_attr(self, x):
        return [getattr(sampler_instance, x) for sampler_instance in self]


    def get_func(self, x, kwargs=None):
        if kwargs is None:
            kwargs = {}
        return [getattr(sampler_instance, x)(**kwargs) for sampler_instance in self]


    pass
