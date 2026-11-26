"""Stored sweep endpoints must agree with the returned, tempered state."""

import numpy as np
import pytest

from reddemcee import PTSampler
from reddemcee.moves import StretchMove
from reddemcee.utils import get_ss


def log_like(coords):
    return -0.5 * np.sum((coords - 1) ** 2)


def log_prior(coords):
    return -0.5 * np.sum(coords ** 2) / 9


@pytest.fixture(autouse=True)
def preserve_random_state():
    previous = np.random.get_state()
    np.random.seed(216)
    try:
        yield
    finally:
        np.random.set_state(previous)


@pytest.mark.parametrize("nsteps", [1, 3])
def test_stored_endpoint_matches_swapped_adapted_state(nsteps):
    sampler = PTSampler(12, 2, log_like, log_prior, betas=[1, 0.1, 0],
                        adapt_mode="SAR", adapt_nu=10)
    initial = np.random.RandomState(91).normal(size=(3, 12, 2))
    for sweep, state in enumerate(sampler.sample(initial, 5, nsteps), 1):
        stored = sampler.get_last_sample()
        for t in range(3):
            assert sampler.backend[t].iteration == sweep * nsteps
            for name in ("coords", "log_like", "log_prob", "beta"):
                np.testing.assert_array_equal(getattr(stored[t], name), getattr(state[t], name))
        chain = sampler.get_chain()
        likelihood = -0.5 * np.sum((chain - 1) ** 2, axis=-1)
        prior = -0.5 * np.sum(chain ** 2, axis=-1) / 9
        np.testing.assert_allclose(sampler.get_log_like(), likelihood)
        np.testing.assert_allclose(sampler.get_log_prob(),
                                   prior + likelihood * sampler.get_betas()[..., None], atol=1e-12)


@pytest.mark.parametrize("store", [True, False])
def test_single_temperature_without_adaptation_can_resume(store):
    sampler = PTSampler(8, 2, log_like, log_prior, betas=[1], adapt_mode="NONE")
    initial = np.random.RandomState(123).normal(size=(1, 8, 2))
    sampler.run_mcmc(initial, 3, 2, store=store)
    state = sampler.run_mcmc(None, 2, 3, store=store)
    assert np.all(np.isfinite(state[0].log_prob))
    assert state[0].beta == 1
    assert sampler.backend[0].iteration == (12 if store else 0)


def test_stepping_stone_evidence_honors_retained_sample_thinning():
    sampler = PTSampler(12, 2, log_like, log_prior, betas=[1, 0.1, 0],
                        adapt_mode="NONE")
    sampler.run_mcmc(np.random.RandomState(8).normal(size=(3, 12, 2)), 18)
    expected = get_ss(sampler.get_betas(discard=2, thin=3),
                       sampler.get_log_like(discard=2, thin=3))
    np.testing.assert_allclose(sampler.get_evidence_ss(discard=2, thin=3), expected)


def test_hybrid_cut_selection_does_not_modify_temperature_history():
    sampler = PTSampler(8, 2, log_like, log_prior, betas=[1, 0.3, 0],
                        adapt_mode="NONE")
    history = np.tile(np.array([1., .3, 0.])[:, None], (1, 4))
    original = history.copy()
    sampler._hybrid_max(history)
    np.testing.assert_array_equal(history, original)


@pytest.mark.parametrize("store", [True, False])
@pytest.mark.parametrize("tune", [True, False])
def test_weighted_identical_moves_preserve_single_move_trajectory_and_rng(store, tune):
    initial = np.random.RandomState(91).normal(size=(3, 12, 2))
    reference = None
    for moves in (None, [(StretchMove(), .3), (StretchMove(), .7)]):
        np.random.seed(22)
        sampler = PTSampler(12, 2, log_like, log_prior, betas=[1, .5, 0],
                            adapt_mode="NONE", moves=moves)
        state = sampler.run_mcmc(initial, 4, 2, store=store, tune=tune)
        arrays = [np.stack([getattr(part, name) for part in state])
                  for name in ("coords", "log_prob", "log_like")]
        if store:
            arrays.extend([sampler.get_chain(), sampler.get_log_prob(), sampler.get_betas()])
        for random_state in (sampler._random.get_state(), np.random.get_state(), state.random_state):
            arrays.extend([random_state[1], np.asarray(random_state[2:])])
        if reference is None:
            reference = arrays
        else:
            for actual, expected in zip(arrays, reference):
                np.testing.assert_array_equal(actual, expected)
