# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT


from itertools import product as iter_prod
from collections import namedtuple
from collections.abc import Iterable

from typing import Dict, List, Optional, Union

from .state import PTState
from .moves import StretchMove, PTMove
from .backend import PTBackend
from .wrapper import PTWrapper
from .utils import set_temp_ladder

from tqdm import tqdm
import numpy as np
from emcee import autocorr


class LadderAdaptation(object):
    def calc_decay(self):
        return self.config_adaptation_halflife / (self.backend[0].iteration + self.config_adaptation_halflife)

    def calc_dss(self):
        kappa = self.calc_decay() / self.config_adaptation_rate
        adjust = self.adjustment_fn()
        adjust[np.abs(adjust) < 1e-8] = 0   # Failsafe for very small values
        return kappa * adjust
    
    def select_adjustment(self, option=0):
        """Select an adjustment method.
        It tries to equalize the following:

        0: Temperature Swap Rate

        1: Swap Mean Distance

        2: Specific Heat

        3: dE/sig . Approximation propossed by...
        

        Args:
            option (int): Selects the method.
        """
        self.adjustment_fn = getattr(self, f'calc_adjust{option}')


    def calc_adjust00(self):
        # No adjustment
        return np.zeros(self.ntemps-2)

    def calc_adjust0(self):
        # Uniform Swap Acceptance Rate
        return -np.diff(self.ts_accepted)
    
    def calc_adjust1(self):
        # Swap Mean Distance
        return -np.diff(self.smd)
    
    def calc_adjust2(self):
        # SGG
        logls = self.get_log_like(flat=True)[:, -1000:]
        db = -np.diff(self.betas)
        varL = np.var(logls, axis=1)
        a = np.exp(-varL[:-1] * db**2)
        return -np.diff(a)

    def calc_adjust3(self):
        # dE_m/sig_m; rathore et al 2005
        # SAO
        logls = self.get_log_like(flat=True)[:, -1000:]
        Esig = np.std(logls, axis=1)
        Emean = np.mean(logls, axis=1)
        sig1 = Esig[:-1]
        sig2 = Esig[1:]
        sigmean = (sig1+sig2)/2
        diffE = np.diff(Emean)
        a = diffE/sigmean        
        return -np.diff(a)

    def calc_adjust4(self):
        '''THERMODYNAMIC LENGTH'''
        betas = np.asarray(self.betas)
        logls = self.get_log_like(flat=True)[:, -1000:]
        Esig = np.std(logls, axis=1)

        dL = -np.diff(betas) * (Esig[1:] + Esig[:-1]) / 2

        norm = -np.trapz(Esig, betas)/(len(betas)-1)
        dLn = dL/norm

        return np.diff(dLn)

    def calc_adjust5(self):
        E = np.mean(self.get_log_like(flat=True)[:, -1000:], axis=1)
        dE = np.diff(E)
        db = np.diff(self.betas)
        logA = -dE * db

        return -np.diff(logA)

    def calc_adjust6(self):
        # beta density
        betas = np.asarray(self.betas)[:-1]
        #t = 1/betas[:-1]
        #gamma = t[1:]/t[:-1]
        #nu = 1/np.log(gamma)
        #tmean = (t[:-1] - t[1:]) / (np.log(t[:-1]) - np.log(t[1:]))
        
        nu = -1/np.diff(np.log(betas))
        betamean = (betas[:-1] - betas[1:]) / (np.log(betas[:-1]) - np.log(betas[1:]))

        # CvL
        logls = self.get_log_like(flat=True)[:, -1000:]
        #Esig = np.std(logls, axis=1)[:-1]
        Esig = np.std(logls, axis=1)[:-1]

        Esigmean = (Esig[:-1] - Esig[1:]) / (np.log(Esig[:-1]) - np.log(Esig[1:]))
        #cvmean = Esigmean / tmean
        cvmean = Esigmean * betamean


        #nu_norm = -np.trapz(tmean, nu)
        #cv_norm = -np.trapz(tmean, sqrt_cvmean)
        nu_norm = np.sum(nu)
        cv_norm = np.sum(cvmean)
        #print(f'{betas=}')
        
        nu_normed = nu / nu_norm
        cv_normed = cvmean / cv_norm

        #print(f'{nu_normed=}')
        #print(f'{cv_normed=}')
        #print(f'coso = {(nu_normed-cv_normed)}')
        return (nu_normed-cv_normed)

    def calc_adjust7(self):
        # beta density
        betas = np.asarray(self.betas)
        betamean = (betas[:-1] + betas[1:]) / 2
        nub = -1/np.diff(np.log(betas))


        logls = self.get_log_like(flat=True)[:, -1000:]
        Esig = np.std(logls, axis=1)
        #Esigmean = (Esig[:-1] - Esig[1:]) / (np.log(Esig[:-1]) - np.log(Esig[1:]))
        Esigmean = (Esig[:-1] + Esig[1:]) / 2
        #cvmean = Esigmean / tmean
        cvmean = Esigmean * betamean


        nub_norm = np.sum(nub)
        cv_norm = np.sum(cvmean)

        a = (nub/nub_norm)-(cvmean/cv_norm)
        return -np.diff(a)

    def calc_adjust8(self):
        return -np.diff(np.log(self.ts_accepted))

    def calc_adjust9(self):
        # beta density
        betas = np.asarray(self.betas)
        logls = self.get_log_like(flat=True)[:, -1000:]
        Esig = np.std(logls, axis=1)

        dL = -np.diff(betas) * (Esig[1:] + Esig[:-1]) / 2

        norm = -np.trapz(Esig, betas)/(len(betas)-1)
        dLn = dL - norm
        print(f'{dL=}')
        print(f'{dLn=}')

        return np.diff(dLn)


class PTSampler(LadderAdaptation):
    def __init__(
        self,
        nwalkers,
        ndim,
        log_like,
        log_prior,
        betas=None,
        ntemps=None,
        pool=None,
        moves=None,
        loglargs=None,
        loglkwargs=None,
        logpargs=None,
        logpkwargs=None,
        backend=None,
        vectorize=False,
        blobs_dtype=None,
        smd_history=False,
        tsw_history=True,
        adapt_tau=1000,
        adapt_nu=1,
        adapt_mode=0,
        parameter_names: Optional[Union[Dict[str, int], List[str]]] = None,
    ):
        """Initialize the adaptive parallel tempering MCMC ensemble sampler.

        Args:
            nwalkers (int): The number of walkers in each ensemble.
            ndim (int): The number of dimensions in the parameter space.
            log_like (callable): The log-likelihood function.
            log_prior (callable): The log-prior function.
            betas (Optional[Iterable[float]]): The inverse temperatures of the parallel chains.
            ntemps (Optional[int]): The number of temperatures. If None, determined from betas.
            pool (Optional[Pool]): A pool object for parallel processing.
            moves (Optional[List[Move]]): A list of moves to use.
            loglargs (Optional[tuple]): Positional arguments for the log-likelihood function.
            loglkwargs (Optional[dict]): Keyword arguments for the log-likelihood function.
            logpargs (Optional[tuple]): Positional arguments for the log-prior function.
            logpkwargs (Optional[dict]): Keyword arguments for the log-prior function.
            backend (Optional[Backend]): A backend object for storing the chain.
            vectorize (Optional[bool]): Whether to vectorize the log-probability function.
            blobs_dtype (Optional[dtype]): The dtype of the blobs.
            smd_history (Optional[bool]): Whether to store swap mean distance history.
            tsw_history (Optional[bool]): Whether to store temperature swap history.
            adapt_tau (Optional[int]): Halflife of adaptation.
            adapt_nu (Optional[int]): Adaptation rate.
            adapt_mode (Optional[int]): Adaptation mode.
            parameter_names (Optional[Union[Dict[str, int], List[str]]]): Parameter names.

        """
        # Parse Move
        self._parse_moves(moves, tsw_history, smd_history)
        
        self.config_adaptation_halflife = adapt_tau
        self.config_adaptation_rate = adapt_nu
        self.config_adaptation_mode = adapt_mode
        self.select_adjustment(self.config_adaptation_mode)

        self.z_ngrid = 10001
        self.z_nsim = 1000 

        self.pool = pool
        self.vectorize = vectorize
        self.blobs_dtype = blobs_dtype

        self.ndim = ndim
        self.nwalkers = nwalkers
        self.ntemps = ntemps or len(betas)

        self.betas = betas or set_temp_ladder(ntemps, ndim)


        # Initialize random number generator
        self._random = np.random.RandomState()
        # Initialize the Backend
        self._init_backend(backend)

        # Wrap the log-probability functions
        self.log_prob_fn = PTWrapper(log_like, log_prior, loglargs, loglkwargs, logpargs, logpkwargs)

        # Save the parameter names
        self._parameter_names(parameter_names)


    def sample(
        self,
        initial_state, 
        nsteps=1,
        nsweeps=1,
        tune=False,
        thin_by=1,
        store=True,
        progress=False
    ):
        """Advance the chain as a generator

        Args:
            initial_state (State or ndarray[nwalkers, ndim]): The initial state of the walkers.
            nsteps (Optional[int]): The number of steps to generate.
            nsweeps (Optional[int]): The number of sweeps to generate.
            tune (Optional[bool]): If True, tune the parameters of some moves.
            thin_by (Optional[int]): Store every `thin_by` samples.
            store (Optional[bool]): If True, store the positions and log-probabilities.
            progress (Optional[bool or str]): Show a progress bar if True.

        Yields:
            State (State): The state of the ensemble at each step.

        """
        self._check_sample_init(nsteps, store, thin_by)
        # modify swap move in case of smd
        #self._swap_move.D_ = self.D_

        # Interpret the input as a walker state.
        state = PTState(initial_state, copy=True)

        # Check and Initialize states
        self._init_states(state)

        # Thin
        thin_by = int(thin_by)
        yield_step = checkpoint_step = thin_by

        iterations = int(nsteps * nsweeps)
        # Store
        if store:
            self.backend.grow(nsweeps)
            for t in range(self.ntemps):
                self.backend[t].grow(iterations, state[t].blobs)

        # Set up a wrapper around the relevant model functions
        map_fn = self.pool.map if self.pool is not None else map
        model = self._create_model(map_fn)

        # Inject the progress bar
        total = None if nsteps is None else iterations * yield_step * self.ntemps

        with tqdm(total=total, disable=not progress) as pbar:
            i = 0
            for _ in range(nsweeps):
                # SELECT SWAP FROM RANDOM CHOICE
                swap_move = self._swap_move
                state, self.ts_accepted, self.smd = swap_move.swap(state)

                for t in range(self.ntemps):
                    for _, _ in iter_prod(range(nsteps), range(yield_step)):
                        # Choose a random move
                        move = self._random.choice(self._moves, p=self._weights)

                        # Propose
                        state[t], accepted = move.propose(model, state[t])
                        state.random_state = self._random.get_state()

                        if tune:
                            move.tune(state[t], accepted)

                        # Save the new step
                        if store and (i + 1) % checkpoint_step == 0:
                            self.backend[t].save_step(state[t], accepted)

                        pbar.update(1)
                        i += 1

                # TEMPERATURE SWAP
                state, self.ts_accepted, self.smd = swap_move.swap(state)

                # TEMPERATURE LADDER ADAPTATION
                state = swap_move.adapt(state, self.calc_dss())

                # SAVE
                self.backend.save_step(state, self.ts_accepted, self.smd)

                # Overwrite sampler betas, this is for user
                self.betas = state.betas
                # Yield the result as an iterator
                yield state


    def run_mcmc(self, initial_state, nsteps, nsweeps, **kwargs):
        """
        Iterate :func:`sample` for ``nsweeps`` times ``nsteps`` iterations and return the result

        Args:
            initial_state (State or ndarray[nwalkers, ndim]): The initial state or position vector. Can also be
                ``None`` to resume from where :func:``run_mcmc`` left off the
                last time it executed.
            nsteps (int): The number of steps to run.
            nsweeps (int): The number of sweeps to run.


        Other parameters are directly passed to :func:`sample`.

        This method returns the most recent result from :func:`sample`.

        """
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called."
                )
            initial_state = self._previous_state
        
        #self.ntemps__ = initial_state.shape[0]

        results = None
        for results in self.sample(initial_state,
                                   nsteps=nsteps,
                                   nsweeps=nsweeps,
                                   **kwargs):
            pass

        # Store so that the ``initial_state=None`` case will work
        self._previous_state = results

        return results


    def run_auto_mcmc(self, initial_state, maxiter,
                      init_steps=100, repeats=1, **kwargs):
        """
        Iterate :func:`sample` for ``nsweeps`` times ``nsteps`` iterations and return the result

        Args:
            initial_state (State or ndarray[nwalkers, ndim]): The initial state or position vector. Can also be
                ``None`` to resume from where :func:``run_mcmc`` left off the
                last time it executed.
            nsteps (int): The number of steps to run.
            nsweeps (int): The number of sweeps to run.


        Other parameters are directly passed to :func:`sample`.

        This method returns the most recent result from :func:`sample`.

        """
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called."
                )
            initial_state = self._previous_state

        self.steps_per_sweep = []
        results = None
        ns_ = 0
        rp_ = 0
        act = init_steps
        while ns_ <= maxiter:
            for results in self.sample(initial_state,
                                    nsteps=act,
                                    nsweeps=repeats,
                                    **kwargs):
                pass
            ns_ += act
            self.steps_per_sweep.append(act)
            #if rp_ % repeats == 0:
            x = self.backend[0].get_log_like(discard=ns_//2)
            act = round(autocorr.integrated_time(x, tol=0)[0])
            #act = round(self.backend[0].get_autocorr_time(tol=0)))
            print(f'{ns_=} | {act=}')
            #if remaining == 1:
            #    self.select_adjustment(self.config_adaptation_mode)
            #    remaining = int(np.mean(self.backend[0].get_autocorr_time(tol=0)))
            #    print(f'{remaining=}')
            #else:
            #    self.select_adjustment('00')
            #    remaining -= 1
            # Store so that the ``initial_state=None`` case will work
            
            self._previous_state = results

        return results


    def compute_log_prob(self, coords, beta):
        """Calculate the vector of log-probability for the walkers.

        This method calculates the log-probability for each walker given their
        coordinates and the inverse temperature. It handles parameter naming,
        vectorization, and parallel processing.

        Args:
            coords (ndarray): The coordinates of the walkers.
            beta (float): The inverse temperature.

        Returns:
            Tuple[ndarray, ndarray, Optional[ndarray]]: A tuple containing the log
                probability, log likelihood, and blobs (if any).

        Raises:
            ValueError: If any parameter value is infinite or NaN, or if the
                probability function returns NaN.
        """
        p = coords

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN")

        if self.params_are_named:
            p = ndarray_to_list_of_dicts(p, self.parameter_names)

        beta_array = np.repeat(beta, len(p))

        if self.vectorize:
            results = self.log_prob_fn(zip(p, beta_array))
        else:
            map_func = self.pool.map if self.pool is not None else map
            results = list(map_func(self.log_prob_fn, zip(p, beta_array)))


        # Unpack results
        try:
            # Assume results are tuples: (log_prob, log_like, *blobs)
            log_prob = np.array([_scalar(l[0]) for l in results])
            log_like = np.array([_scalar(l[1]) for l in results])
            blobs = [l[2:] for l in results if len(l) > 2]
            blobs = self._process_blobs(blobs) if blobs else None
        except Exception as e:
            raise ValueError(f"Error in log_prob_fn: {e}") from e

        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")

        return log_prob, log_like, blobs


    def reset(self):
        """Reset the book-keeping parameters.

        This method resets the backend, clearing all stored data and resetting the
        iteration counter. It also re-initializes the backend with the current
        number of temperatures, walkers, dimensions, and swap history settings.
        """
        self.backend.reset(self.ntemps, self.nwalkers, self.ndim,
                           smd_hist=self._swap_move.smd_hist, tsw_hist=self._swap_move.tsw_hist)


    def get_evidence_ti(self, discard=0, thin=1, mode='obm',
                        ba=None, bb=None, pchip=True,
                        fixed_ladder=True, batch_size=None,
                        spe_kernel='bartlett'):

        L = self.get_log_like(discard=discard, thin=thin)  # shape=(ntemps, nsweeps, nwalkers)
        B = self.get_betas(discard=discard, thin=thin)  # shape=(ntemps, nsweeps)

        from .utils import get_ti
        return get_ti(B, L,
                      pchip=pchip,
                      ba=ba,
                      bb=bb,
                      fixed_ladder=fixed_ladder,
                      mode=mode,
                      batch_size=batch_size,
                      spe_kernel=spe_kernel)


    def get_evidence_ss(self, discard=0, thin=1, mode='obm',
                        ba=None, bb=None, bridge=True,
                        fixed_ladder=True, batch_size=None,
                        spe_kernel='bartlett'):
        
        L = self.get_log_like(discard=discard)  # shape=(ntemps, nsweeps, nwalkers)
        B = self.get_betas(discard=discard)  # shape=(ntemps, nsweeps)

        from .utils import get_ss
        return get_ss(B, L,
                      mode=mode,
                      ba=ba, bb=bb, bridge=bridge,
                      fixed_ladder=fixed_ladder, batch_size=batch_size,
                      spe_kernel=spe_kernel)


    def get_evidence_hybrid(self, discard=0, nbetacut=None,
                            nbetacut_crit='max',
                            ti_args=None, ss_args=None):
        if ti_args is None:
            ti_args = {'mode':'obm',
                       'ba':None, 'bb':None, 'pchip':True,
                       'fixed_ladder':True, 'batch_size':None,
                       'spe_kernel':'bartlett'}

        if ss_args is None:
            ss_args = {'mode':'obm',
                       'ba':None, 'bb':None, 'bridge':True,
                       'fixed_ladder':True, 'batch_size':None,
                       'spe_kernel':'bartlett'}


        L = self.get_log_like(discard=discard)  # shape=(ntemps, nsweeps, nwalkers)
        B = self.get_betas(discard=discard)  # shape=(ntemps, nsweeps)    

        

        if nbetacut is None:
            if nbetacut_crit == 'max':
                all_cuts = self._hybrid_max(B)
                        #print(f'{max_index=}')
            elif nbetacut_crit=='tri':
                all_cuts = self._hybrid_tri(B, L)
                        #print(max_index)
            else:
                print('MODE NOT DETECTED')
        else:
            all_cuts = self._hybrid_def(nbetacut)
        
        ti_args['ba'] = all_cuts[0]
        ti_args['bb'] = all_cuts[1]

        ss_args['ba'] = all_cuts[2]
        ss_args['bb'] = all_cuts[3]

        from .utils import get_ss, get_ti

        zti, ztierr = get_ti(B, L, **ti_args)

        zss, zsserr = get_ss(B, L, **ss_args)

        z_hat = zti + zss
        err_tot = np.sqrt(zsserr**2 + ztierr**2)

        return z_hat, err_tot




    def _hybrid_max(self, B):
        betas = B[::-1, -1]
        if betas[0] == 0:
            betas[0] = np.finfo(np.float32).eps
        y = 1/np.diff(np.log(betas))
        max_index = np.argmax(y)
        return self._hybrid_def(max_index)

    def _hybrid_tri(self, B, L):
        x = B[::-1, -1]
        y = np.mean(np.mean(L, axis=2), axis=1)[::-1]

        cuad = np.diff(x) * y[:-1]
        triang = np.diff(x) * np.diff(y)/2
        mask = triang<=cuad

        max_index = np.sum(mask)
        return self._hybrid_def(max_index)

    def _hybrid_def(self, nbetacut):
        ba_ti = 0
        bb_ti = nbetacut
        ba_ss = nbetacut
        bb_ss = self.ntemps - 1

        return ba_ti, bb_ti, ba_ss, bb_ss




    def get_autocorr_time(self, **kwargs):
        """Get the estimated autocorrelation time.

        This method returns the estimated autocorrelation time from the backend. It is useful
        for assessing the convergence of the chains.

        Args:
            **kwargs (Optional[dict]): Additional keyword arguments passed to the backend.

        Returns:
            (ndarray): The estimated autocorrelation time.
        """
        return self.backend.get_autocorr_time(**kwargs)

    def get_betas(self, **kwargs):
        """Get the betas of the chains.

        This method returns the betas history of the chains from the backend. It is useful
        for checking the temperature ladder adaptation.

        Args:
            **kwargs (Optional[dict]): Additional keyword arguments passed to the backend.

        Returns:
            (ndarray): The betas of the chains.
        """
        return self.get_value("beta_history", **kwargs)

    def get_chain(self, **kwargs):
        """Get the chain of samples.

        This method returns the chain of samples from the backend. It is useful
        for analyzing the posterior distribution.

        Args:
            **kwargs (Optional[dict]): Additional keyword arguments passed to the backend.

        Returns:
            (ndarray): The chain of samples.
        """
        return self.get_value("chain", **kwargs)

    def get_blobs(self, **kwargs):
        """Get the blobs.

        This method returns the blobs from the backend. Blobs are extra information
        returned by the log-probability function.
        Args:
            **kwargs (Optional[dict]): Additional keyword arguments passed to the backend.

        Returns:
            (ndarray): The chain of samples.
        """
        return self.get_value("blobs", **kwargs)

    def get_log_prob(self, **kwargs):
        """Get the log probability.

        This method returns the log probability from the backend. It is useful for
        analyzing the convergence of the chains.

        Args:
            **kwargs (Optional[dict]): Additional keyword arguments passed to the backend.

        Returns:
            (ndarray): The log probability.
        """
        return self.get_value("log_prob", **kwargs)

    def get_log_like(self, **kwargs):
        """Get the log likelihood.

        This method returns the log likelihood from the backend. It is useful for
        analyzing the posterior distribution.

        Args:
            **kwargs (Optional[dict]): Additional keyword arguments passed to the backend.

        Returns:
            (ndarray): The log likelihood.
        """
        return self.get_value("log_like", **kwargs)

    def get_last_sample(self, **kwargs):
        """Get the last sample.

        This method returns the last sample from the backend. It is useful for
        checking the current state of the chains.

        Args:
            **kwargs (Optional[dict]): Additional keyword arguments passed to the backend.

        Returns:
            (State): The last sample.
        """
        return self.backend.get_last_sample()

    def get_value(self, name, **kwargs):
        """Get a value from the backend.

        This method returns a value from the backend by its name. It is useful for
        accessing specific data stored during sampling.

        Args:
            name (str): The name of the value to retrieve.
            **kwargs (Optional[dict]): Additional keyword arguments passed to the backend.

        Returns:
            The value from the backend.
        """
        return self.backend.get_value(name, **kwargs)


    def _parse_moves(self, moves, tsw_history, smd_history):
        """Parse and initialize the moves"""
        if moves is None:
            self._moves = [StretchMove()]
            self._weights = [1.0]
        elif isinstance(moves, Iterable):
            # Check if moves is a sequence of (move, weight) tuples
            first_elem = next(iter(moves))
            if isinstance(first_elem, tuple) and len(first_elem) == 2:
                self._moves, self._weights = zip(*moves)
            else:
                self._moves = list(moves)
                self._weights = np.ones(len(self._moves))
        else:
            self._moves = [moves]
            self._weights = [1.0]

        self._weights = np.array(self._weights, dtype=float)
        self._weights /= np.sum(self._weights)

        # Parse the APT move schedule
        self._swap_move = PTMove()
        if tsw_history:
            self._swap_move.tsw_hist = True

        if smd_history:
            self._swap_move.smd_hist = True

    def _process_blobs(self, blobs):
        """Process blobs to ensure consistent dtype and shape"""
        if self.blobs_dtype is not None:
            dt = self.blobs_dtype
        else:
            try:
                dt = np.array(blobs[0]).dtype
            except Exception:
                dt = np.dtype('object')
        return np.array(blobs, dtype=dt)

    def _check_sample_init(self, nsteps, store, thin_by):
        if nsteps is None and store:
            raise ValueError("'store' must be False when 'nsteps' is None")
        if thin_by <= 0:
            raise ValueError("Invalid thinning argument")

    def _check_states(self, states):
        states_shape = np.shape(states)
        if states_shape != (self.ntemps, self.nwalkers, self.ndim):
            raise ValueError(f"incompatible input dimensions {states_shape}")
        
        for st in states:
            if not walkers_independent(st.coords):
                raise ValueError(
                    "Initial state has a large condition number. "
                    "Make sure that your walkers are linearly independent for the "
                    "best performance"
                )

    def _init_states(self, states):
        # Check the dimensions and walker independence
        self._check_states(states)

        for t in range(self.ntemps):
            state = states[t]
            beta = self.betas[t]

            if state.log_prob is None:
                state.beta = beta
                state.log_prob, state.log_like, state.blobs = self.compute_log_prob(state.coords, beta)

            if np.shape(state.log_prob) != (self.nwalkers,):
                raise ValueError("incompatible input dimensions")
            if np.any(np.isnan(state.log_prob)):
                raise ValueError("The initial log_prob was NaN")        

    def _create_model(self, map_fn):
        """Create a model for log probability calculations."""
        return namedtuple("Model", ("log_prob_fn", "compute_log_prob_fn", "map_fn", "random"))(
            self.log_prob_fn, self.compute_log_prob, map_fn, self._random
        )

    def _init_backend(self, backend):

        self.backend = PTBackend() if backend is None else backend
        if not self.backend.initialized:
            self._previous_state = None
            self.backend.reset(self.ntemps, self.nwalkers, self.ndim,
                               smd_hist=self._swap_move.smd_hist, tsw_hist=self._swap_move.tsw_hist)

            rstate = np.random.get_state()
            self.backend.random_state = rstate  # MARK1
        else:
            # TODO check previous backend shape?

            rstate = self.backend.random_state
            if rstate is None:
                rstate = np.random.get_state()

            # Grab the last step so that we can restart
            it = self.backend.iteration
            if it > 0:
                self._previous_state = self.get_last_sample()

        self._random.set_state(rstate)  # MARK1

    def _parameter_names(self, parameter_names):
        self.params_are_named: bool = parameter_names is not None
        if self.params_are_named:
            assert isinstance(parameter_names, (list, dict))
            assert not self.vectorize, "Named parameters with vectorization unsupported for now"

            if isinstance(parameter_names, list):
                assert len(parameter_names) == self.ndim, "Name all parameters or set `parameter_names` to `None`"
                parameter_names = {name: i for i, name in enumerate(parameter_names)}

            values = [
                v if isinstance(v, list) else [v]
                for v in parameter_names.values()
            ]
            values = {item for sublist in values for item in sublist}
            assert values == set(range(self.ndim)), f"Not all values appear -- set should be 0 to {self.ndim-1}"
            self.parameter_names = parameter_names

    @property
    def iteration(self):
        return self.backend.iteration


    @property
    def acceptance_fraction(self):
        """The fraction of proposed steps that were accepted."""
        return self.backend.accepted / float(self.backend[0].iteration)


    #@property
    def get_tsw(self, **kwargs):
        #temperature_swap_fraction
        """The fraction of proposed swaps that were accepted."""
        return self.backend.get_tsw(**kwargs)


    #@property
    def get_smd(self, **kwargs):
        # swap_mean_distance
        """The swap mean distance, normalised."""
        return self.backend.get_smd(**kwargs)


    def __getstate__(self):
        # In order to be generally picklable, we need to discard the pool
        # object before trying.
        d = self.__dict__.copy()
        d["pool"] = None
        return d



def walkers_independent(coords):
    if not np.all(np.isfinite(coords)):
        return False
    C = coords - np.mean(coords, axis=0)[None, :]
    C_colmax = np.amax(np.abs(C), axis=0)
    if np.any(C_colmax == 0):
        return False
    C /= C_colmax
    C_colsum = np.sqrt(np.sum(C**2, axis=0))
    C /= C_colsum
    return np.linalg.cond(C.astype(float)) <= 1e8


def ndarray_to_list_of_dicts(
    x: np.ndarray, key_map: Dict[str, Union[int, List[int]]]
) -> List[Dict[str, Union[np.number, np.ndarray]]]:
    """
    A helper function to convert a ``np.ndarray`` into a list
    of dictionaries of parameters. Used when parameters are named.

    Args:
      x (np.ndarray): parameter array of shape ``(N, n_dim)``, where
        ``N`` is an integer
      key_map (Dict[str, Union[int, List[int]]):

    Returns:
      list of dictionaries of parameters
    """
    return [{key: xi[val] for key, val in key_map.items()} for xi in x]


def _scalar(fx):
    # Make sure a value is a true scalar
    # 1.0, np.float64(1.0), np.array([1.0]), np.array(1.0)
    if not np.isscalar(fx):
        try:
            fx = np.asarray(fx).item()
        except (TypeError, ValueError) as e:
            raise ValueError("log_prob_fn should return scalar") from e
    return float(fx)