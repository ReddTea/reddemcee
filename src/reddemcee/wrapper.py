# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

import numpy as np


        class PTWrapper_debug(object):
    """
    Make logl and logp pickable
    """

    def __init__(self, logl, logp, loglargs, loglkwargs, logpargs, logpkwargs):
        self.logl = logl
        self.logp = logp

        self.loglargs = loglargs or []
        self.loglkwargs = loglkwargs or {}
        self.logpargs = logpargs or []
        self.logpkwargs = logpkwargs or {}

    def __call__(self, x, beta):
        try:
            lp = self.logp(x, *self.logpargs, **self.logpkwargs)
            ll = self.logl(x, *self.loglargs, **self.loglkwargs)

            log_prob = lp + ll*beta

            return log_prob, ll
        
        except Exception as e:
            import traceback

            print("Exception while calling your likelihood function:")
            print("  params:", x)
            print("  exception:")
            traceback.print_exc()
            raise


class PTWrapper(object):
    """
    Make logl and logp pickable
    """

    def __init__(self, logl, logp, loglargs, loglkwargs, logpargs, logpkwargs):
        self.logl = logl
        self.logp = logp

        self.loglargs = loglargs or []
        self.loglkwargs = loglkwargs or {}
        self.logpargs = logpargs or []
        self.logpkwargs = logpkwargs or {}

    def __call__(self, x_beta):
        x, beta = x_beta
        
        lp = self.logp(x, *self.logpargs, **self.logpkwargs)
        if lp == float('-inf'):
            return lp, lp

        ll = self.logl(x, *self.loglargs, **self.loglkwargs)
        return lp + ll * beta, ll

    def vectorized(self, coords, beta):
        """Evaluate walker batches and skip likelihoods outside the prior.

        Callbacks receive ``(npoints, ndim)`` coordinates and return one value
        per point, or a scalar constant.
        """
        coords = np.asarray(coords)
        lp = self._vector_values(
            self.logp(coords, *self.logpargs, **self.logpkwargs), len(coords))
        valid = lp != -np.inf
        if np.all(valid):
            ll = self._vector_values(
                self.logl(coords, *self.loglargs, **self.loglkwargs), len(coords))
            return lp + ll * beta, ll

        log_prob = lp.copy()
        log_like = np.full(len(coords), -np.inf)
        if np.any(valid):
            ll = self._vector_values(
                self.logl(coords[valid], *self.loglargs, **self.loglkwargs),
                np.count_nonzero(valid))
            log_like[valid] = ll
            log_prob[valid] += ll * beta
        return log_prob, log_like

    def _vector_values(self, values, npoints):
        # A callback may return coordinate views or reuse one scratch array.
        # State probabilities must own their storage before another call.
        values = np.array(values, dtype=float, copy=True)
        if values.ndim == 0:
            return np.full(npoints, values.item())
        if values.shape != (npoints,):
            raise ValueError("Vectorized probability function must return one value per walker")
        return values
