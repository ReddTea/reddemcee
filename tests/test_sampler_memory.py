"""Regressions for sampler memory ownership and bounded no-storage runs."""

import gc
import math
import pickle
import weakref

import numpy as np
import pytest

from reddemcee import PTSampler
from reddemcee.backend import PTBackend


def log_like(coords):
    return -0.5 * np.sum((coords - 0.7) ** 2)


def log_prior(coords):
    return -0.05 * np.sum(coords ** 2)


def make_sampler(**kwargs):
    return PTSampler(7, 2, log_like, log_prior,
                     betas=[1.0, 0.4, 0.1, 0.03], **kwargs)


def populated_backend(nrows=220, blobs=False):
    """Allocate deterministic history without spending time sampling it."""
    backend = PTBackend()
    backend.reset(4, 7, 2, tsw_hist=True)
    backend.grow(nrows)
    backend.tsw_history[:] = 0.5
    backend.iteration = nrows
    for temperature in range(4):
        child = backend[temperature]
        initial_blobs = np.zeros((7, 2)) if blobs else None
        child.grow(nrows, initial_blobs)
        values = np.arange(nrows * 7).reshape(nrows, 7)
        child.chain[:] = values[..., None] + [0.1, 0.2]
        child.log_like[:] = -(temperature + 1) * (values + 1) ** 1.2
        child.log_prob[:] = child.log_like / (temperature + 1)
        child.beta_history[:] = [1.0, 0.4, 0.1, 0.03][temperature]
        if blobs:
            child.blobs[:] = values[..., None] + [0.3, 0.4]
        child.iteration = nrows
    return backend


def history_references(backend):
    return [weakref.ref(getattr(child, name)) for child in backend
            for name in ("chain", "log_like", "log_prob", "beta_history", "blobs")
            if getattr(child, name) is not None]


def test_last_sample_is_detached_and_does_not_keep_history_alive():
    backend = populated_backend(nrows=12, blobs=True)
    references = history_references(backend)
    last = backend.get_last_sample()
    for temperature in range(4):
        for state_name, history_name in (("coords", "chain"), ("log_like", "log_like"),
                                         ("log_prob", "log_prob"), ("blobs", "blobs")):
            value = getattr(last[temperature], state_name)
            history = getattr(backend[temperature], history_name)
            np.testing.assert_array_equal(value, history[-1])
            assert not np.shares_memory(value, history)
    # Do not let this test's own loop locals keep a historical allocation alive.
    del history
    backend.reset(4, 7, 2, tsw_hist=True)
    gc.collect()
    assert all(reference() is None for reference in references)
    assert last[0].coords.shape == (7, 2)


def test_resumed_sampler_reset_releases_history_and_restart_state():
    backend = populated_backend(nrows=12)
    references = history_references(backend)
    sampler = make_sampler(backend=backend, adapt_mode="NONE")
    previous = weakref.ref(sampler._previous_state)
    sampler.reset()
    gc.collect()
    assert sampler._previous_state is None
    assert previous() is None
    assert all(reference() is None for reference in references)
    with pytest.raises(ValueError, match="initial_state=None"):
        sampler.run_mcmc(None)


def test_resumed_sampler_restores_the_persisted_temperature_ladder():
    backend = populated_backend(nrows=12)
    expected = backend.get_last_sample().betas
    sampler = PTSampler(7, 2, log_like, log_prior, backend=backend,
                        betas=[1.0, 0.7, 0.2, 0.01], adapt_mode="SGG")
    np.testing.assert_array_equal(sampler.betas, expected)
    np.testing.assert_array_equal(sampler.betas, sampler._previous_state.betas)


def test_reset_releases_optional_histories_when_disabling_them():
    backend = PTBackend()
    backend.reset(4, 7, 2, tsw_hist=True, smd_hist=True)
    backend.grow(20)
    histories = [weakref.ref(backend.tsw_history), weakref.ref(backend.smd_history)]
    backend.reset(4, 7, 2, tsw_hist=False, smd_hist=False)
    gc.collect()
    assert all(reference() is None for reference in histories)
    assert not backend.tsw_history_bool
    assert not backend.smd_history_bool


@pytest.mark.parametrize("stored_first", [False, True])
@pytest.mark.parametrize("swap_history,distance_history", [
    (False, False), (False, True), (True, False), (True, True),
])
def test_store_false_leaves_backend_unchanged(stored_first, swap_history, distance_history):
    sampler = make_sampler(adapt_mode="SAR", tsw_history=swap_history,
                           smd_history=distance_history)
    initial = np.random.RandomState(71).normal(size=(4, 7, 2))
    if stored_first:
        sampler.run_mcmc(initial, nsweeps=2, nsteps=2)
        initial = None
    before = pickle.dumps(sampler.backend, protocol=5)
    state = sampler.run_mcmc(initial, nsweeps=3, nsteps=2, thin_by=2, store=False)
    assert pickle.dumps(sampler.backend, protocol=5) == before
    assert all(np.all(np.isfinite(part.coords)) for part in state)
    assert all(np.all(np.isfinite(part.log_prob)) for part in state)
    # Resume once more to cover a no-storage run after another no-storage run.
    sampler.run_mcmc(None, nsweeps=2, nsteps=1, store=False)
    assert pickle.dumps(sampler.backend, protocol=5) == before


def test_store_false_adaptation_decay_advances_and_resets():
    sampler = make_sampler(adapt_mode="NONE", adapt_tau=10)
    decays = []

    def adjustment():
        decays.append(sampler.calc_decay())
        return np.zeros(2)

    sampler.adjustment_fn = adjustment
    sampler.run_mcmc(np.random.RandomState(12).normal(size=(4, 7, 2)),
                     nsweeps=3, nsteps=2, thin_by=3, store=False)
    sampler.run_mcmc(None, nsweeps=2, nsteps=3, store=False)
    np.testing.assert_allclose(decays, 10 / (10 + np.array([2, 4, 6, 9, 12])))
    assert sampler.backend[0].iteration == 0
    sampler.reset()
    assert sampler.calc_decay() == 1


@pytest.mark.parametrize("nrows", [10, 143, 220])
def test_recent_likelihood_read_discards_before_materializing(nrows, monkeypatch):
    sampler = make_sampler(backend=populated_backend(nrows), adapt_mode="NONE")
    expected = sampler.get_log_like(flat=True)[:, -1000:]
    original = sampler.get_log_like
    requests = []

    def bounded_read(**kwargs):
        requests.append(kwargs)
        assert kwargs["discard"] == max(0, nrows - math.ceil(1000 / 7))
        return original(**kwargs)

    monkeypatch.setattr(sampler, "get_log_like", bounded_read)
    recent = sampler._get_recent_log_like()
    np.testing.assert_array_equal(recent, expected)
    assert len(requests) == 1
    assert recent.shape == (4, min(1000, nrows * 7))


@pytest.mark.parametrize("mode", ["SGG", "GAO", "ETL", "5", "6", "7", "9"])
def test_history_adaptation_uses_bounded_window_without_changing_result(mode, monkeypatch):
    sampler = make_sampler(backend=populated_backend(), adapt_mode=mode)
    expected_window = sampler.get_log_like(flat=True)[:, -1000:]
    recent_reader = sampler._get_recent_log_like
    monkeypatch.setattr(sampler, "_get_recent_log_like", lambda: expected_window)
    expected = sampler.adjustment_fn()
    monkeypatch.setattr(sampler, "_get_recent_log_like", recent_reader)
    original = sampler.get_log_like
    requests = []

    def bounded_read(**kwargs):
        requests.append(kwargs)
        assert kwargs["discard"] == 220 - math.ceil(1000 / 7)
        return original(**kwargs)

    monkeypatch.setattr(sampler, "get_log_like", bounded_read)
    np.testing.assert_allclose(sampler.adjustment_fn(), expected)
    assert len(requests) == 1


@pytest.mark.parametrize("stored_first", [False, True])
def test_store_false_history_adaptation_has_bounded_transient_storage(stored_first):
    sampler = make_sampler(adapt_mode="ETL", adapt_nu=1000)
    initial = np.random.RandomState(97).normal(size=(4, 7, 2))
    if stored_first:
        sampler.run_mcmc(initial, nsweeps=2, nsteps=2)
        initial = None
    before = pickle.dumps(sampler.backend, protocol=5)
    state = sampler.run_mcmc(initial, nsweeps=2, nsteps=160, store=False)
    assert pickle.dumps(sampler.backend, protocol=5) == before
    assert sampler._get_recent_log_like().shape == (4, 1000)
    assert all(len(window) <= math.ceil(1000 / 7) for window in sampler._log_like_window)
    assert all(np.all(np.isfinite(part.log_prob)) for part in state)


@pytest.mark.parametrize("mode", ["SAR", "SGG", "GAO", "ETL", "5", "6", "7", "9"])
def test_disabling_storage_preserves_seeded_sampling_and_adaptation(mode):
    initial = np.random.RandomState(46).normal(size=(4, 7, 2))
    global_random_state = np.random.get_state()
    results = []
    try:
        for store in (True, False):
            np.random.seed(125)
            sampler = make_sampler(adapt_mode=mode, adapt_nu=1000)
            sampler.run_mcmc(initial, nsweeps=3, nsteps=2, thin_by=2, store=store)
            results.append(sampler.run_mcmc(None, nsweeps=2, nsteps=3, store=store))
    finally:
        np.random.set_state(global_random_state)
    for temperature in range(4):
        for name in ("coords", "log_like", "log_prob", "beta"):
            np.testing.assert_array_equal(getattr(results[0][temperature], name),
                                          getattr(results[1][temperature], name))
