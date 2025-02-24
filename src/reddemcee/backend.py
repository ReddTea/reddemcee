# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

import numpy as np
from .state import State, PTState

from emcee import autocorr


class PTBackend(object):
    def __init__(self, dtype=None):
        self.initialized = False
        self.random_state = None
        self._list = []
        self.dtype = dtype or np.float64

    def reset(self, ntemps, nwalkers, ndim, smd_hist=False, tsw_hist=False):
        """Initialize the PTBackend and per-temperature backends.

        Args:
            ntemps (int): Number of temperatures.
            nwalkers (int): Number of walkers.
            ndim (int): Number of dimensions.
            smd_hist (bool): Whether to store SMD history.
            tsw_hist (bool): Whether to store temperature swap history.
        """
        self.ntemps = int(ntemps)
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.iteration = 0
        self.ts_accepted = np.zeros(self.ntemps-1, dtype=self.dtype)

        self.tsw_history_bool = tsw_hist
        self.smd_history_bool = smd_hist

        if tsw_hist:
            self.tsw_history = np.empty((0, self.ntemps-1), dtype=self.dtype)
        if smd_hist:
            self.smd_history = np.empty((0, self.ntemps-1), dtype=self.dtype)

        self.backends = [Backend_plus(dtype=self.dtype) for _ in range(self.ntemps)]
        for backend in self.backends:
            backend.reset(self.nwalkers, self.ndim)
        self.initialized = True
        
    def save_step(self, states, ts_accepted, smd=None):
        """Save a step for all temperatures.

        Args:
            states (list of State): States for each temperature.
            accepted (list of ndarray): Acceptance arrays for each temperature.
            ts_accepted (ndarray): Temperature swap acceptance array.
            smd (Optional[ndarray]): SMD history data.
        """

        # Save temperature swapped/adapted state
        for t, backend in enumerate(self):
            backend.chain[-1, :, :] = states[t].coords
            backend.log_like[-1, :] = states[t].log_like
            backend.log_prob[-1, :] = states[t].log_prob

        # Handle PTBackend-specific data
        if self.tsw_history_bool:
            self.tsw_history[self.iteration] = ts_accepted

        if self.smd_history_bool:
            self.smd_history[self.iteration] = smd
        
        self.ts_accepted += ts_accepted
        self.iteration += 1


    def grow(self, ngrow):
        i = ngrow - (len(self.tsw_history) - self.iteration)
        a = np.empty((i, self.ntemps-1), dtype=self.dtype)

        if self.tsw_history_bool:
            self.tsw_history = np.concatenate((self.tsw_history, a), axis=0)
        if self.smd_history_bool:
            self.smd_history = np.concatenate((self.smd_history, a), axis=0)


    def get_autocorr_time(self, discard=0, thin=1, **kwargs):
        return [b.get_autocorr_time(discard=discard, thin=thin, **kwargs) for b in self.backends]


    def get_log_prob(self, **kwargs):
        """Get log probabilities from all temperatures."""
        return self.get_value('log_prob', **kwargs)

    def get_log_like(self, **kwargs):
        """Get log likelihoods from all temperatures."""
        return self.get_value('log_like', **kwargs)

    def get_chain(self, **kwargs):
        """Get chains from all temperatures."""
        return self.get_value('chain', **kwargs)

    def get_betas(self, **kwargs):
        return self.get_value('beta_history', **kwargs)


    def get_value(self, name, **kwargs):
        return np.array([b.get_value(name, **kwargs) for b in self])


    def get_tsw(self, **kwargs):
        # TODO implement this
        return self.get_value_over('tsw_history', **kwargs)
    

    def get_smd(self, **kwargs):
        # TODO implement this
        return self.get_value_over('smd_history', **kwargs)


    def get_value_over(self, name, flat=False, thin=1, discard=0):
        if self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        if name == "blobs" and not self.has_blobs():
            return None

        v = getattr(self, name)[discard + thin - 1 : self.iteration : thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

    def get_last_sample(self):
        """Access the most recent sample in the chain"""
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        states = PTState([b.get_last_sample() for b in self])
        states.random_state = self.random_state
        return states

    @property
    def shape(self):
        """The dimensions of the ensemble `(ntemps, nwalkers, ndim)`."""
        return self.ntemps, self.nwalkers, self.ndim

    @property
    def accepted(self):
        """Acceptance arrays from all temperatures."""
        return np.array([b.accepted for b in self])

        
    def __getitem__(self, n):
        return self.backends[n]


    def __iter__(self):
        return iter(self.backends)



class Backend_plus(object):
    """A backend that extends emcee's Backend with additional storage for log_like and beta_history."""

    def __init__(self, dtype=None):
        self.initialized = False
        if dtype is None:
            dtype = np.float64
        self.dtype = dtype


    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.iteration = 0
        self.accepted = np.zeros(self.nwalkers, dtype=self.dtype)
        self.chain = np.empty((0, self.nwalkers, self.ndim), dtype=self.dtype)
        self.log_like = np.empty((0, self.nwalkers), dtype=self.dtype)
        self.log_prob = np.empty((0, self.nwalkers), dtype=self.dtype)
        self.beta_history = np.empty((0,), dtype=self.dtype)
        self.blobs = None
        self.random_state = None
        self.initialized = True


    def get_log_like(self, **kwargs):
        """Get the chain of log likelihoods evaluated at the MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log likelihoods.
        """
        return self.get_value("log_like", **kwargs)
    
    def get_log_prob(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples
        """
        return self.get_value("log_prob", **kwargs)
    
    def get_chain(self, **kwargs):
        """Get the chain of log probabilities evaluated at the MCMC samples
        """
        return self.get_value("chain", **kwargs)

    def get_betas(self, **kwargs):
        """Get the chain of log likelihoods evaluated at the MCMC samples

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of log likelihoods.
        """
        return self.get_value("beta_history", **kwargs)

    def get_blobs(self, **kwargs):
        return self.get_value("blobs", **kwargs)

    def get_last_sample(self):
        """Access the most recent sample in the chain"""
        if not self.initialized or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with 'store == True' before accessing the results"
            )
        it = self.iteration
        blobs = self.get_blobs(discard=it - 1)
        if blobs is not None:
            blobs = blobs[0]
        return State(
            self.get_chain(discard=it - 1)[0],
            log_prob=self.get_log_prob(discard=it - 1)[0],
            log_like=self.get_log_like(discard=it - 1)[0],
            beta=self.get_betas(discard=it - 1)[0],
            blobs=blobs,
        )


    def get_value(self, name, flat=False, thin=1, discard=0):
        if self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        if name == "blobs" and not self.has_blobs():
            return None

        v = getattr(self, name)[discard + thin - 1 : self.iteration : thin]
        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v
    

    def get_autocorr_time(self, discard=0, thin=1, **kwargs):
        """Compute an estimate of the autocorrelation time for each parameter

        Args:
            thin (Optional[int]): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Other arguments are passed directly to
        :func:`emcee.autocorr.integrated_time`.

        Returns:
            array[ndim]: The integrated autocorrelation time estimate for the
                chain for each parameter.

        """
        x = self.get_chain(discard=discard, thin=thin)
        return thin * autocorr.integrated_time(x, **kwargs)


    @property
    def beta(self):
        """Access the most recent beta in the chain"""
        return self.beta_history[-1]


    @property
    def shape(self):
        """The dimensions of the ensemble ``(nwalkers, ndim)``"""
        return self.nwalkers, self.ndim

    def has_blobs(self):
        """Returns ``True`` if the model includes blobs"""
        return self.blobs is not None

    def _check_blobs(self, blobs):
        has_blobs = self.has_blobs()
        if has_blobs and blobs is None:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs is not None and not has_blobs:
            raise ValueError("inconsistent use of blobs")

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """
        i = ngrow - (len(self.chain) - self.iteration)
        if i > 0:
            self._check_blobs(blobs)
            a = np.empty((i, self.nwalkers, self.ndim), dtype=self.dtype)
            self.chain = np.concatenate((self.chain, a), axis=0)

            a = np.empty((i, self.nwalkers), dtype=self.dtype)
            self.log_prob = np.concatenate((self.log_prob, a), axis=0)
            
            a = np.empty((i, self.nwalkers), dtype=self.dtype)
            self.log_like = np.concatenate((self.log_like, a), axis=0)

            a = np.empty((i), dtype=self.dtype)
            self.beta_history = np.concatenate((self.beta_history, a), axis=0)
        

            if blobs is not None:
                dt = np.dtype((blobs.dtype, blobs.shape[1:]))
                a = np.empty((i, self.nwalkers), dtype=dt)
                if self.blobs is None:
                    self.blobs = a
                else:
                    self.blobs = np.concatenate((self.blobs, a), axis=0)

    def _check(self, state, accepted):
        self._check_blobs(state.blobs)
        nwalkers, ndim = self.shape
        has_blobs = self.has_blobs()
        if state.coords.shape != (nwalkers, ndim):
            raise ValueError(
                "invalid coordinate dimensions; expected {0}".format(
                    (nwalkers, ndim)
                )
            )
        if state.log_prob.shape != (nwalkers,):
            raise ValueError(
                "invalid log probability size; expected {0}".format(nwalkers)
            )
        if state.blobs is not None and not has_blobs:
            raise ValueError("unexpected blobs")
        if state.blobs is None and has_blobs:
            raise ValueError("expected blobs, but none were given")
        if state.blobs is not None and len(state.blobs) != nwalkers:
            raise ValueError(
                "invalid blobs size; expected {0}".format(nwalkers)
            )
        if accepted.shape != (nwalkers,):
            raise ValueError(
                "invalid acceptance size; expected {0}".format(nwalkers)
            )

    def save_step(self, state, accepted):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        self._check(state, accepted)

        self.chain[self.iteration, :, :] = state.coords
        self.log_like[self.iteration, :] = state.log_like
        self.log_prob[self.iteration, :] = state.log_prob
        self.beta_history[self.iteration] = state.beta
        if state.blobs is not None:
            self.blobs[self.iteration, :] = state.blobs
        self.accepted += accepted
        self.iteration += 1


    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass


