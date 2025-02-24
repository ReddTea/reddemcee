# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

import numpy as np
import os
try:
    import h5py
except ImportError:
    h5py = None

from .state import State, PTState

from .backend import Backend_plus

class HDFBackend_plus(Backend_plus):
    """A backend that stores the chain in an HDF5 file using h5py"""
    
    def __init__(
        self,
        filename,
        name="mcmc",
        read_only=False,
        dtype=None,
        compression=None,
        compression_opts=None,
    ):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")
        self.filename = filename
        self.name = name
        self.read_only = read_only
        self.compression = compression
        self.compression_opts = compression_opts
        if dtype is None:
            self.dtype_set = False
            self.dtype = np.float64
        else:
            self.dtype_set = True
            self.dtype = dtype

    @property
    def initialized(self):
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False


    def open(self, mode="r"):
        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only "
                "mode. Set `read_only = False` to make "
                "changes."
            )
        f = h5py.File(self.filename, mode)
        if not self.dtype_set and self.name in f:
            g = f[self.name]
            if "chain" in g:
                self.dtype = g["chain"].dtype
                self.dtype_set = True
        return f
    

    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        with self.open("a") as f:
            if self.name in f:
                del f[self.name]

            g = f.create_group(self.name)
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = 0
            g.create_dataset(
                "accepted",
                data=np.zeros(nwalkers),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "chain",
                (0, nwalkers, ndim),
                maxshape=(None, nwalkers, ndim),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "log_prob",
                (0, nwalkers),
                maxshape=(None, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "log_like",
                (0, nwalkers),
                maxshape=(None, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "beta_history",
                (0,),
                maxshape=(None,),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

    def has_blobs(self):
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, flat=False, thin=1, discard=0):
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError(
                    "You must run the sampler with "
                    "'store == True' before accessing the "
                    "results"
                )

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            v = g[name][discard + thin - 1 : self.iteration : thin]
            if flat:
                s = list(v.shape[1:])
                s[0] = np.prod(v.shape[:2])
                return v.reshape(s)
            return v


    @property
    def shape(self):
        with self.open() as f:
            g = f[self.name]
            return g.attrs["nwalkers"], g.attrs["ndim"]

    @property
    def iteration(self):
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted(self):
        with self.open() as f:
            return f[self.name]["accepted"][...]
        

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain"].resize(ntot, axis=0)
            g["log_like"].resize(ntot, axis=0)
            g["log_prob"].resize(ntot, axis=0)
            g["beta_history"].resize(ntot, axis=0)

            if blobs is not None:
                has_blobs = g.attrs["has_blobs"]
                if not has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    dt = np.dtype((blobs.dtype, blobs.shape[1:]))
                    g.create_dataset(
                        "blobs",
                        (ntot, nwalkers),
                        maxshape=(None, nwalkers),
                        dtype=dt,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )
                else:
                    g["blobs"].resize(ntot, axis=0)
                    if g["blobs"].dtype.shape != blobs.shape[1:]:
                        raise ValueError(
                            f'Existing blobs have shape {g["blobs"].dtype.shape} but new blobs requested with shape {blobs.shape[1:]}'
                        )
                g.attrs["has_blobs"] = True

    def save_step(self, state, accepted):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        self._check(state, accepted)

        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]

            g["chain"][iteration, :, :] = state.coords
            g["log_like"][iteration, :] = state.log_like
            g["log_prob"][iteration, :] = state.log_prob
            g["beta_history"][iteration] = state.beta

            if state.blobs is not None:
                g["blobs"][iteration, :] = state.blobs
            g["accepted"][:] += accepted

            g.attrs["iteration"] = iteration + 1


    def swap_step(self, state):
         with self.open("a") as f:
            g = f[self.name]
            #iteration = g.attrs["iteration"]

            g["chain"][-1, :, :] = state.coords
            g["log_like"][-1, :] = state.log_like
            g["log_prob"][-1, :] = state.log_prob

            if state.blobs is not None:
                g["blobs"][-1, :] = state.blobs



class PTHDFBackend(object):
    def __init__(
        self,
        filename,
        name="mcmc",
        read_only=False,
        dtype=None,
        compression=None,
        compression_opts=None,
    ):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDFBackend")
        self.filename = filename
        self.filename_raw = filename.split('.')[0]
        self.name = name
        self.read_only = read_only
        self.compression = compression
        self.compression_opts = compression_opts
        self.backends = []
        if dtype is None:
            self.dtype_set = False
            self.dtype = np.float64
        else:
            self.dtype_set = True
            self.dtype = dtype

    
    @property
    def initialized(self):
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False
        
    
    def open(self, mode="r"):
        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only "
                "mode. Set `read_only = False` to make "
                "changes."
            )
        f = h5py.File(self.filename, mode)

        # TODO use another attr
        if not self.dtype_set and self.name in f:
            g = f[self.name]
            if "chain" in g:
                self.dtype = g["chain"].dtype
                self.dtype_set = True
        return f
    

    def reset(self, ntemps, nwalkers, ndim, tsw_hist=True, smd_hist=False):
        """Clear the state of the chain and empty the backend

        Args:
            ntemps (int): Number of temperatures.
            nwalkers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        with self.open("a") as f:
            if self.name in f:
                del f[self.name]

            g = f.create_group(self.name)
            g.attrs["ntemps"] = ntemps
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = 0
            g.create_dataset(
                "ts_accepted",
                data=np.zeros(ntemps-1),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            
            g.attrs['tsw_history_bool'] = tsw_hist
            g.attrs['smd_history_bool'] = smd_hist
            
            if tsw_hist:
                g.create_dataset(
                    "tsw_history",
                    (0, ntemps-1),
                    maxshape=(None, ntemps-1),
                    dtype=self.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

            if smd_hist:
                g.create_dataset(
                    "smd_history",
                    (0, ntemps-1),
                    maxshape=(None, ntemps-1),
                    dtype=self.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

            self.backends = [HDFBackend_plus(f'{self.filename_raw}_{t}.h5') for t in range(ntemps)]
            for backend in self.backends:
                backend.reset(nwalkers, ndim)


    def grow(self, ngrow):
        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            #tsw_history_bool = g["tsw_history_bool"]
            #smd_history_bool = g["smd_history_bool"]

            if self.tsw_history_bool:
                g["tsw_history"].resize(ntot, axis=0)

            if self.smd_history_bool:
                g["smd_history"].resize(ntot, axis=0)


    def save_step(self, states, ts_accepted, smd=None):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        for t, backend in enumerate(self.backends):
            backend.swap_step(states[t])


        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            tsw_history_bool = g.attrs["tsw_history_bool"]
            smd_history_bool = g.attrs["smd_history_bool"]

            if tsw_history_bool:
                g["tsw_history"][iteration] = ts_accepted

            if smd_history_bool:
                g["smd_history"][iteration] = smd

            g["ts_accepted"][:] += ts_accepted
            g.attrs["iteration"] = iteration + 1
    


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
        return np.array([b.get_value(name, **kwargs) for b in self.backends])


    def get_tsw(self, **kwargs):
        return self.get_value_over('tsw_history', **kwargs)


    def get_smd(self, **kwargs):
        return self.get_value_over('smd_history', **kwargs)
        

    def get_value_over(self, name, flat=False, thin=1, discard=0):
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError(
                    "You must run the sampler with "
                    "'store == True' before accessing the "
                    "results"
                )

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            v = g[name][discard + thin - 1 : self.iteration : thin]
            if flat:
                s = list(v.shape[1:])
                s[0] = np.prod(v.shape[:2])
                return v.reshape(s)
            return v


    @property
    def shape(self):
        """The dimensions of the ensemble `(ntemps, nwalkers, ndim)`."""
        with self.open() as f:
            g = f[self.name]
            return g.attrs["ntemps"], g.attrs["nwalkers"], g.attrs["ndim"]

    @property
    def accepted(self):
        """Acceptance arrays from all temperatures."""
        return np.array([b.accepted for b in self.backends])

    
    @property
    def random_state(self):
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]
        return elements if len(elements) else None


    @property
    def tsw_history_bool(self):
        with self.open() as f:
            return f[self.name].attrs["tsw_history_bool"]


    @property
    def smd_history_bool(self):
        with self.open() as f:
            return f[self.name].attrs["smd_history_bool"]
        
    @property
    def iteration(self):
        with self.open() as f:
            return f[self.name].attrs["iteration"]
        

    def get_last_sample(self):
        """Access the most recent sample in the chain"""
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        states = PTState([b.get_last_sample() for b in self.backends])
        states.random_state = self.random_state
        return states


    def __getitem__(self, n):
        return self.backends[n]


    def __iter__(self):
        return iter(self.backends)
    
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass    