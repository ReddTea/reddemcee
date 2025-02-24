# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT


class PTWrapper_1(object):
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