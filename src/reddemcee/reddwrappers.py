# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.3.3
# date 17 jan 2023

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT


class PTWrapper(object):
    def __init__(self, logl, logp, beta, loglargs=None, logpargs=None, loglkwargs=None, logpkwargs=None):
        if loglargs is None:
            loglargs = []
        if logpargs is None:
            logpargs = []
        if loglkwargs is None:
            loglkwargs = {}
        if logpkwargs is None:
            logpkwargs = {}
        self.logl = logl
        self.logp = logp

        self.loglargs = loglargs
        self.logpargs = logpargs
        self.loglkwargs = loglkwargs
        self.logpkwargs = logpkwargs

        self.beta = beta

    def __call__(self, x):
        lp = self.logp(x, *self.logpargs, **self.logpkwargs)
        if lp == float('-inf'):
            return lp, lp

        ll = self.logl(x, *self.loglargs, **self.loglkwargs)
        return lp + ll * self.beta, ll



#
