# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# version 0.3
# date 14 nov 2022

import numpy as np
import pandas as pd
import os
# first one is a wrapper for models, takes
# second one is a wrapper for the PT, only works for emcee3


# coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT


class ModelWrapper(object):
    def __init__(self, func_model, fargs=[], fkwargs={}):
        self.func = func_model
        self.fargs = fargs
        self.fkwargs = fkwargs
        pass

    def __call__(self, x):
        return self.func(x, *self.fargs, **self.fkwargs)


class PTWrapper(object):
    def __init__(self, logl, logp, beta,
                 loglargs=[], logpargs=[],
                 loglkwargs={}, logpkwargs={}):

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


class PTWrapperModel(object):
    ###DEL
    def __init__(self, model, beta):
        self.model = model
        self.beta = beta

    def __call__(self, x):
        lp = self.model.evaluate_prior(x)
        if lp == float('-inf'):
            return lp, lp

        ll = self.model(x)

        return lp + ll * self.beta, ll


class DataWrapper(object):
    def __init__(self, target_name):
        self.target_name = target_name
        self.PATH = 'datafiles/%s/RV/' % self.target_name

        empty_lists = ['ndata', 'ncols', 'nsai', 'data', 'labels']
        for attribute in empty_lists:
            setattr(self, attribute, [])


    def add_data__(self, filename):
        data = np.loadtxt('{0}{1}'.format(self.PATH, filename))

        ndat, ncol = data.shape

        self.ndata.append(ndat)
        self.ncols.append(ncol)
        self.labels.append(filename)

        names = ['BJD', 'RV', 'eRV']

        # identify and name SAI
        nsa = ncol - 3
        if nsa > 0:
            for j in range(nsa):
                names.append('Staract %i' % (j))
        self.nsai.append(nsa)

        df = pd.DataFrame(data, columns=names)

        # substract RV
        if abs(df.mean()['RV']) > 1e6:
            df['RV'] -= df.mean()['RV']

        # create another column containing flags for the instrument
        df['Flag'] = np.ones(ndat, int) * len(self.ndata)
        self.data.append(df)

        return 'Reading data from {0}'.format(filename)


    def add_all__(self):
        x = ''
        for file in np.sort(os.listdir(self.PATH)):
            x += self.add_data__(file)+' \n '
        print(x)


    def get_data__(self, sortby='BJD'):
        return pd.concat(self.data).sort_values(sortby)


    def get_metadata__(self):
        return [getattr(self, attribute) for attribute in ['ndata', 'ncols', 'nsai', 'labels']]


    def get_data_raw(self):
        holder = pd.concat(self.data).sort_values(sortby)
        x, y, yerr = holder.BJD, holder.RV, holder.eRV
        return [x, y, yerr]














#
