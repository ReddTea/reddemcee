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

class LadderAdaptation(object):
    def calc_decay(self):
        return self.config_adaptation_halflife / (self.backend[0].iteration + self.config_adaptation_halflife)

    def calc_dss(self):
        kappa = self.calc_decay() / self.config_adaptation_rate
        adjust = self.adjustment_fn()
        adjust[np.abs(adjust) < 1e-8] = 0   # Failsafe for very small values
        return kappa * adjust
    
    def select_adjustment(self, option=0):
        self.adjustment_fn = getattr(self, f'calc_adjust{option}')

    def calc_adjust0(self):
        return -np.diff(self.ts_accepted)
    
    def calc_adjust1(self):
        return -np.diff(self.smd)
    
    def calc_adjust2(self):
        # CvL
        logls = self.get_log_like(flat=True)[:, -1000:]
        db = -np.diff(self.betas)
        varL = np.var(logls, axis=1)
        a = np.exp(-varL[:-1] * db**2)
        return -np.diff(a)

    def calc_adjust3(self):
        # dE_m/sig_m
        logls = self.get_log_like(flat=True)[:, -1000:]
        Esig = np.std(logls, axis=1)
        Emean = np.mean(logls, axis=1)
        sig1 = Esig[:-1]
        sig2 = Esig[1:]
        sigmean = (sig1+sig2)/2
        diffE = np.diff(Emean)
        a = diffE/sigmean        
        return -np.diff(a)



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
            State: The state of the ensemble at each step.

        """
        self._check_sample_init(nsteps, store, thin_by)

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
        Iterate :func:`sample` for ``nsteps`` iterations and return the result

        Args:
            initial_state: The initial state or position vector. Can also be
                ``None`` to resume from where :func:``run_mcmc`` left off the
                last time it executed.
            nsteps: The number of steps to run.
            nsweeps: The number of sweeps to run.


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

        results = None
        for results in self.sample(initial_state, nsteps=nsteps, nsweeps=nsweeps, **kwargs):
            pass

        # Store so that the ``initial_state=None`` case will work
        self._previous_state = results

        return results


    def compute_log_prob(self, coords, beta):
        """Calculate the vector of log-probability for the walkers"""
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
        """
        Reset the bookkeeping parameters

        """
        self.backend.reset(self.ntemps, self.nwalkers, self.ndim,
                           smd_hist=self._swap_move.smd_hist, tsw_hist=self._swap_move.tsw_hist)


    def thermodynamic_integration_classic(self, **kwargs):
        from .utils import thermodynamic_integration_classic
        logls0 = self.get_log_like(flat=True, **kwargs)
        logls = np.mean(logls0, axis=1)

        return thermodynamic_integration_classic(self.betas, logls)


    def thermodynamic_integration(self, **kwargs):
        from .utils import thermodynamic_integration
        # Sort Betas And Logls
        x = self.betas[::-1]
        logls0 = self.get_log_like(flat=True, **kwargs)
        logls1 = logls0[::-1]

        # if hot chain beta !=0, add it
        # we approximate it's value to logls1[0]
        # TODO add an option so this value can be provided
        if x[0] != 0:
            x = np.concatenate(([0], x))
            logls1 = np.concatenate(([logls1[0]], logls1))

        return thermodynamic_integration(x, logls1, ngrid=self.z_ngrid, nsim=self.z_nsim)
    

    def get_betas(self, **kwargs):
        return self.get_value("beta_history", **kwargs)

    def get_chain(self, **kwargs):
        return self.get_value("chain", **kwargs)

    def get_blobs(self, **kwargs):
        return self.get_value("blobs", **kwargs)

    def get_log_prob(self, **kwargs):
        return self.get_value("log_prob", **kwargs)

    def get_log_like(self, **kwargs):
        return self.get_value("log_like", **kwargs)

    def get_last_sample(self, **kwargs):
        return self.backend.get_last_sample()

    def get_value(self, name, **kwargs):
        return self.backend.get_value(name, **kwargs)

    def get_autocorr_time(self, **kwargs):
        return self.backend.get_autocorr_time(**kwargs)

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
        """The fraction of proposed steps that were accepted"""
        return self.backend.accepted / float(self.backend[0].iteration)


    #@property
    def get_tsw(self, **kwargs):
        #temperature_swap_fraction
        """The fraction of proposed swaps that were accepted"""
        return self.backend.get_tsw(**kwargs)


    #@property
    def get_smd(self, **kwargs):
        # swap_mean_distance
        """The swap mean distance, normalised"""
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