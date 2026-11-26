"""HDF storage and restart regressions for the parallel-tempering sampler."""
import tracemalloc

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from reddemcee import PTSampler
from reddemcee.hdf import PTHDFBackend
from reddemcee.state import PTState, State


def log_like(x):
    return -0.5 * np.sum((x - 1) ** 2)


def log_prior(x):
    return -0.5 * np.sum(x ** 2) / 9


def assert_random_state_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert tuple(actual[2:]) == tuple(expected[2:])


@pytest.mark.parametrize("nsteps", [1, 3])
def test_fresh_hdf_sampler_and_reopened_resume(tmp_path, nsteps):
    backend = PTHDFBackend(tmp_path / "chain.h5")
    sampler = PTSampler(8, 2, log_like, log_prior, betas=[1, 0.1, 0],
                        backend=backend, adapt_mode="SAR", adapt_nu=10)
    rng = np.random.RandomState(213)
    sampler._random.set_state(rng.get_state())
    for sweep, state in enumerate(sampler.sample(rng.normal(size=(3, 8, 2)), 3, nsteps), 1):
        stored = backend.get_last_sample()
        for t in range(3):
            assert backend[t].iteration == sweep * nsteps
            np.testing.assert_array_equal(stored[t].coords, state[t].coords)
            np.testing.assert_array_equal(stored[t].log_like, state[t].log_like)
            np.testing.assert_array_equal(stored[t].log_prob, state[t].log_prob)
            assert stored[t].beta == state[t].beta
        assert_random_state_equal(stored.random_state, state.random_state)

    reopened = PTHDFBackend(tmp_path / "chain.h5")
    np.testing.assert_array_equal(reopened.get_chain(), backend.get_chain())
    np.testing.assert_array_equal(reopened.get_betas(), backend.get_betas())
    resumed = PTSampler(8, 2, log_like, log_prior, betas=[1, 0.1, 0],
                        backend=reopened, adapt_mode="SAR", adapt_nu=10)
    assert_random_state_equal(resumed._random.get_state(), sampler._random.get_state())
    resumed.run_mcmc(None, 1, nsteps)
    assert reopened.iteration == 4
    assert reopened.get_chain().shape == (3, 4 * nsteps, 8, 2)


def test_hdf_swap_updates_written_endpoint_including_blobs(tmp_path):
    backend = PTHDFBackend(tmp_path / "chain.h5")
    backend.reset(2, 4, 2)
    backend.grow(3)
    states = PTState([State(np.full((4, 2), t + 1.),
                            log_prob=np.full(4, t + 2.),
                            log_like=np.full(4, t + 3.),
                            beta=1., blobs=np.full((4, 2), t + 4.))
                      for t in range(2)])
    for t, child in enumerate(backend):
        child.grow(6, states[t].blobs)
        child.save_step(states[t], np.ones(4))
        states[t].coords += 10
        states[t].log_prob += 20
        states[t].log_like += 30
        states[t].blobs += 40
        states[t].beta = 0.5
    states.random_state = np.random.RandomState(4).get_state()
    backend.save_step(states, np.ones(1))

    for t, child in enumerate(backend):
        stored = child.get_last_sample()
        for name in ("coords", "log_prob", "log_like", "blobs"):
            np.testing.assert_array_equal(getattr(stored, name), getattr(states[t], name))
        assert stored.beta == 0.5
        with child.open() as f:
            assert not np.any(f[child.name]["chain"][-1])
    np.testing.assert_array_equal(backend.get_tsw(), np.ones((1, 1)))
    assert_random_state_equal(backend.random_state, states.random_state)


def test_hdf_child_options_dotted_path_and_read_only(tmp_path):
    directory = tmp_path / "run.v1"
    directory.mkdir()
    path = directory / "chain.results.h5"
    backend = PTHDFBackend(path, name="custom", dtype=np.float32,
                           compression="gzip", compression_opts=1)
    backend.reset(2, 4, 2, tsw_hist=False)
    for t, child in enumerate(backend):
        assert child.filename == str(directory / f"chain.results_{t}.h5")
        with child.open() as f:
            dataset = f["custom"]["chain"]
            assert dataset.dtype == np.float32
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 1
    reopened = PTHDFBackend(path, name="custom", read_only=True)
    assert reopened.dtype == np.float32
    assert len(reopened.backends) == 2
    for child in reopened:
        assert child.read_only
        with pytest.raises(RuntimeError, match="read-only"):
            child.reset(4, 2)


def test_hdf_reopens_existing_file_without_dtype_metadata(tmp_path):
    path = tmp_path / "chain.h5"
    backend = PTHDFBackend(path, dtype=np.float32)
    backend.reset(2, 4, 2, tsw_hist=False)
    with backend.open("a") as f:
        del f[backend.name].attrs["dtype"]
    reopened = PTHDFBackend(path)
    assert reopened.dtype == np.float32
    assert len(reopened.backends) == 2


def test_hdf_random_state_can_be_replaced_and_cleared(tmp_path):
    backend = PTHDFBackend(tmp_path / "chain.h5")
    backend.reset(2, 4, 2)
    for seed in (123, 456):
        expected = np.random.RandomState(seed).get_state()
        backend.random_state = expected
        assert_random_state_equal(backend.random_state, expected)
    backend.random_state = None
    assert backend.random_state is None


def test_hdf_aggregate_read_does_not_duplicate_full_history(tmp_path):
    backend = PTHDFBackend(tmp_path / "chain.h5")
    backend.reset(4, 16, 2)
    for t, child in enumerate(backend):
        child.grow(8192, None)
        with child.open("a") as f:
            f[child.name]["chain"][:] = t
            f[child.name].attrs["iteration"] = 8192

    tracemalloc.start()
    try:
        values = backend.get_chain()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert values.shape == (4, 8192, 16, 2)
    assert peak < 1.6 * values.nbytes
    np.testing.assert_array_equal(values[:, 0, 0, 0], np.arange(4))
    np.testing.assert_array_equal(backend.get_chain(discard=5, thin=3, flat=True),
                                  values[:, 7::3].reshape(4, -1, 2))


def test_hdf_repeated_reads_close_file_handles(tmp_path):
    backend = PTHDFBackend(tmp_path / "chain.h5")
    sampler = PTSampler(8, 2, log_like, log_prior, betas=[1, 0.1, 0],
                        backend=backend, adapt_mode="NONE")
    sampler.run_mcmc(np.random.RandomState(3).normal(size=(3, 8, 2)), 2)
    before = h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE)
    for _ in range(20):
        backend.get_chain()
        backend.get_last_sample()
    assert h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE) == before
