# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

from copy import deepcopy
import numpy as np


class PTState(object):
    """Parallel Tempering State that manages a list of StatePlus instances."""

    def __init__(self, coords, copy=False):
        if hasattr(coords[0], "coords"):
            self.ntemps = len(coords)
            self.nwalkers, self.ndim = coords[0].coords.shape
            #self.random_state = coords.random_state
        else:
            self.ntemps, self.nwalkers, self.ndim = coords.shape
        
        self.random_state = None
        self.states = [State(coords[t], copy=copy) for t in range(self.ntemps)]


    def __getitem__(self, n):
        return self.states[n]
    
    def __setitem__(self, n, target):
        self.states[n] = target

    def __len__(self):
        return self.ntemps

    def __iter__(self):
        return iter(self.states)


    @property
    def shape(self):
        return self.ntemps, self.nwalkers, self.ndim

    @property
    def betas(self):
        return np.array([state.beta for state in self])

    @betas.setter
    def betas(self, new_betas):
        for t in range(self.ntemps):
            self[t].beta = new_betas[t]
            


class State(object):
    """The state of the ensemble during an MCMC run

    For backwards compatibility, this will unpack into ``coords, log_prob,
    (blobs), random_state`` when iterated over (where ``blobs`` will only be
    included if it exists and is not ``None``).

    Args:
        coords (ndarray[nwalkers, ndim]): The current positions of the walkers
            in the parameter space.
        log_prob (ndarray[nwalkers, ndim], Optional): Log posterior
            probabilities for the  walkers at positions given by ``coords``.
        blobs (Optional): The metadata “blobs” associated with the current
            position. The value is only returned if lnpostfn returns blobs too.
        random_state (Optional): The current state of the random number
            generator.
    """

    __slots__ = "coords", "log_prob", "log_like", "beta", "blobs"

    def __init__(
        self, coords, log_prob=None, log_like=None, blobs=None, beta=1, copy=False
    ):
        dc = deepcopy if copy else lambda x: x

        if hasattr(coords, "coords"):
            self.coords = dc(coords.coords)
            self.log_prob = dc(coords.log_prob)
            self.log_like = dc(coords.log_like)
            self.beta = dc(coords.beta)
            self.blobs = dc(coords.blobs)
            return

        self.coords = dc(np.atleast_2d(coords))
        self.log_prob = dc(log_prob)
        self.log_like = dc(log_like)
        self.blobs = dc(blobs)
        self.beta = dc(beta)

    def __len__(self):
        return 4 if self.blobs is None else 5

    def __repr__(self):
        return "State({0}, log_prob={1}, log_like={2}, beta={3}, blobs={4})".format(
            self.coords, self.log_prob, self.log_like, self.beta, self.blobs
        )

    def __iter__(self):
        if self.blobs is None:
            return iter((self.coords, self.log_prob, self.log_like, self.beta))
        return iter(
            (self.coords, self.log_prob, self.log_like, self.beta, self.blobs)
        )

    def __getitem__(self, index):
        if index < 0:
            return self[len(self) + index]
        if index == 0:
            return self.coords
        elif index == 1:
            return self.log_prob
        elif index == 2:
            return self.log_like
        elif index == 3:
            return self.beta
        elif index == 4 and self.blobs is not None:
            return self.blobs
        raise IndexError("Invalid index '{0}'".format(index))