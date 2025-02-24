#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import reddemcee
import math
import time
#import multiprocessing
import emcee
from multiprocessing import Pool

import matplotlib.pyplot as pl


# defining constants
ndim_ = 2
r_ = 2.  # radius
w_ = 0.1  # width
hard_limit = 6
analytic_z = {2:-1.75, 3:-2.84, 5:-5.6736, 10:-14.59, 20:-36.09}

if True:
    limits = [-hard_limit,  hard_limit]
    c1_ = np.zeros(ndim_)
    c1_[0] = -3.5
    c2_ = np.zeros(ndim_)
    c2_[0] = 3.5
    const_ = math.log(1. / math.sqrt(2. * math.pi * w_**2))  # normalization constant

    # log-likelihood of a single shell
    def logcirc(theta, c):
        d = np.sqrt(np.sum((theta - c)**2, axis=-1))  # |theta - c|
        return const_ - (d - r_)**2 / (2. * w_**2)

    # log-likelihood of two shells
    def loglike(theta):
        return np.logaddexp(logcirc(theta, c1_), logcirc(theta, c2_))

    # our prior transform
    def prior_transform(x):
        return (limits[1] - limits[0]) * x - limits[1]

    def reddprior(theta):
        lp = 0.
        for i in range(ndim_):
            if  theta[i] <= limits[0] or limits[1] <= theta[i]:
                return -np.inf
            else:
                lp += 1/(limits[1] - limits[0])
                #lp += 1/(limits[1] - limits[0])
        return np.log(lp)
    
ntemps_ = 12
nwalkers_ = 200
setup = np.array([ntemps_, nwalkers_, 300, 2])

if True:
#with Pool(20) as mypool:
    ntemps, nwalkers, nsweeps, nsteps = setup
    p0 = list(np.random.uniform(limits[0], limits[1], [ntemps, nwalkers, ndim_]))
    my_betas = np.linspace(1, 0, ntemps)

    time_start = time.time()
    #mypool = Pool(ncores_)
    #my_moves = [[(emcee.moves.DEMove(), 0.8),
    #         (emcee.moves.DESnookerMove(), 0.2)] for _ in range(ntemps)]

    sampler = reddemcee.PTSampler(nwalkers, ndim_,
                                  loglike, reddprior, ntemps=ntemps,
                                  #pool=mypool,
                                  adaptative=True,
                                  #moves=my_moves
                                  betas=my_betas,
                                  config_adaptation_decay=1
                                  #betas=np.array([1.00000000e+00, 6.39396482e-01, 3.95339709e-01, 2.51711228e-01,
                                  #                 1.52989587e-01, 9.86033203e-02, 6.04822861e-02, 4.09715831e-02,
                                  #                 2.54359686e-02, 1.49462952e-02, 8.92801860e-03, 5.00517725e-03,
                                  #                 2.90971855e-03, 1.29839577e-03, 1.47444114e-12]),
                                  )#False
    
    sampler.af_history_bool = True

    sampler.config_adaptation_halflife = 500
    sampler.config_adaptation_rate = 60/nwalkers
    sampler.run_mcmc(p0, nsweeps, nsteps)
    
    time_end = time.time()
    tot_time = time_end-time_start
    print(f'Total time: {tot_time} sec')

discard0 = 500

z1 = sampler.thermodynamic_integration_classic(discard=discard0)
z2 = sampler.thermodynamic_integration(discard=discard0)

logls0 = sampler.get_logls(flat=True, discard=discard0)
logls = np.mean(logls0, axis=1)

if True:
    bh = sampler.betas_history
    rh = sampler.ts_ratios_history
    af = sampler.af_history

    fig, axes = pl.subplots(3, 1, figsize=(14, 6), sharex=True)

    bh1 = bh.reshape((setup[2], setup[0]))
    rh1 = rh.reshape((setup[2], setup[0]-1))
    af1 = np.append(np.zeros(setup[0]), af).reshape((setup[2]+1, setup[0]))
    af1 = np.diff(af1, axis=0)

    for i in range(setup[0]-2):
            bh_sel = bh1[:, i]
            b = 1/np.array(bh_sel)
            axes[0].plot(np.arange(setup[2])*setup[3], b)
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')

    for i in np.arange(setup[0]):
        af_sel = af1[:, i]
        axes[1].plot(np.arange(setup[2])*setup[3], af_sel, alpha=0.5)
            
    for i in np.arange(setup[0]-1):
        r = rh1[:, i]
        axes[2].plot(np.arange(setup[2])*setup[3], r, alpha=0.5)
    
    if True:
        axes[2].set_xlabel("N Step")

        axes[0].set_ylabel(r"$\beta^{-1}$")
        axes[1].set_ylabel(r"$\bar{A_{f}}$")
        axes[2].set_ylabel(r"$a_{frac}$")
            
    pl.tight_layout()
    pl.savefig('test_frac.png')

if True:
    my_text = rf'Evidence: {np.round(z2[0], 3)} $\pm$ {np.round(z2[1], 3)}'
    c = ['C0', 'C1', 'C2', 'C4', 'C7', 'C8', 'C9']
    colors = np.array([c,c,c,c,c]).flatten()
    if True:
        fig, ax = pl.subplots()
        for ti in range(ntemps_):
            bet = bh1.T[ti]
            ax.plot(bet, np.ones_like(bet)*logls[ti], colors[ti])
            ax.plot(bet[-1], logls[ti], colors[ti]+'o')

        ylims = ax.get_ylim()
        
        betas0 = [x[-1] for x in bh1.T]

        ax.fill_between(np.append(betas0, 0),
                        np.append(logls, logls[-1]),
                        y2=ylims[1],
                        #color=rc.fg,
                        color='w',
                        alpha=0.25)
        
        ax.set_ylim(ylims)
    if True:
        #ax.scatter([], [], alpha=0, label=my_text)
        pl.legend(loc=4)
        ax.set_xlabel('temps')
        ax.set_ylabel('mean(logls)')
        
        ax.set_xlim([0, 1])
        pl.tight_layout()
        pl.savefig('test_evidence.png',
                   bbox_inches='tight')
        #pl.show()
        pl.close()


#