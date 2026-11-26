"""Array evaluation keeps scalar, pooled and batched walkers consistent."""
from multiprocessing.pool import ThreadPool
from contextlib import contextmanager
import pickle
import random

import numpy as np
import pytest

from reddemcee import PTSampler
from reddemcee.state import PTState, State


@contextmanager
def seeded(seed):
    """Keep prior, proposal and swap draws independent and restore global RNG."""
    previous = np.random.get_state()
    source = random.Random(seed)
    prior_seed, proposal_seed, swap_seed = [source.getrandbits(32) for _ in range(3)]
    np.random.seed(proposal_seed)
    try:
        yield np.random.RandomState(prior_seed), swap_seed
    finally:
        np.random.set_state(previous)


def log_like(coords):
    return -0.5 * np.sum((np.asarray(coords) - 0.25) ** 2, axis=-1)


def log_prior(coords):
    coords = np.asarray(coords)
    return np.where(np.all(np.abs(coords) < 4, axis=-1), -coords.shape[-1] * np.log(8), -np.inf)


def make_sampler(**kwargs):
    return PTSampler(16, 3, log_like, log_prior, betas=[1.0, 0.2, 0.0],
                     adapt_mode="NONE", **kwargs)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("beta", [0.0, 0.37, 1.0])
def test_scalar_and_vectorized_evaluation_match(dtype, beta):
    coords = np.random.RandomState(5).normal(size=(16, 3)).astype(dtype)
    coords[0] = 5  # Outside the normalized prior, including at beta zero.
    scalar = make_sampler().compute_log_prob(coords, beta)
    batch = make_sampler(vectorize=True).compute_log_prob(coords, beta)
    np.testing.assert_array_equal(scalar[0], batch[0])
    np.testing.assert_array_equal(scalar[1], batch[1])
    assert scalar[2] is None and batch[2] is None
    assert scalar[0][0] == -np.inf and batch[1][0] == -np.inf
    assert scalar[0].flags.c_contiguous and scalar[1].flags.c_contiguous
    assert not np.shares_memory(scalar[0], scalar[1])


def test_vectorized_likelihood_skips_prior_rejections_and_receives_a_batch():
    calls = []

    def likelihood(coords):
        assert coords.ndim == 2 and coords.shape[1] == 3
        assert np.all(np.abs(coords) < 4)
        calls.append(coords.copy())
        return log_like(coords)

    sampler = PTSampler(16, 3, likelihood, log_prior, betas=[1, .2, 0],
                        adapt_mode="NONE", vectorize=True)
    coords = np.zeros((16, 3))
    coords[[0, 5]] = 10
    sampler.compute_log_prob(coords, 0)
    assert len(calls) == 1 and calls[0].shape == (14, 3)
    calls.clear()
    sampler.compute_log_prob(np.full((16, 3), 10.), 0)
    assert calls == []


def test_vectorized_constant_functions_and_shape_validation():
    sampler = PTSampler(16, 3, lambda q: 2., lambda q: -3.,
                        betas=[1, .2, 0], vectorize=True)
    lp, ll, blobs = sampler.compute_log_prob(np.zeros((16, 3)), .5)
    np.testing.assert_array_equal(lp, np.full(16, -2.))
    np.testing.assert_array_equal(ll, np.full(16, 2.))
    assert blobs is None
    sampler.log_prob_fn.logp = lambda q: np.zeros((len(q), 1))
    with pytest.raises(ValueError, match="one value per walker"):
        sampler.compute_log_prob(np.zeros((16, 3)), .5)


def test_batch_outputs_detach_coordinate_views_and_shared_callback_scratch():
    coords = np.random.RandomState(44).normal(size=(16, 3))
    sampler = PTSampler(16, 3, lambda q: q[:, 0], lambda q: 0.,
                        betas=[1, .2, 0], vectorize=True)
    lp, ll, _ = sampler.compute_log_prob(coords, .5)
    expected = coords[:, 0].copy()
    assert not np.shares_memory(ll, coords)
    coords[:, 0] += 10
    np.testing.assert_array_equal(ll, expected)
    np.testing.assert_array_equal(lp, .5 * expected)

    scratch = np.empty(16)

    def prior(q):
        scratch[:len(q)] = -np.sum(q ** 2, axis=1)
        return scratch[:len(q)]

    def likelihood(q):
        scratch[:len(q)] = q[:, 0]
        return scratch[:len(q)]

    sampler = PTSampler(16, 3, likelihood, prior, betas=[1, .2, 0], vectorize=True)
    lp, ll, _ = sampler.compute_log_prob(coords, .5)
    np.testing.assert_array_equal(lp, -np.sum(coords ** 2, axis=1) + .5 * coords[:, 0])
    np.testing.assert_array_equal(ll, coords[:, 0])
    scratch[:] = 0
    np.testing.assert_array_equal(ll, coords[:, 0])


def test_float32_beta_batch_calculates_float64_values_with_matching_precision():
    coords = np.random.RandomState(12).normal(size=(16, 3)).astype(np.float32)
    beta = np.float32(.37)
    scalar = make_sampler().compute_log_prob(coords, beta)
    batch = make_sampler(vectorize=True).compute_log_prob(coords, beta)
    assert batch[0].dtype == batch[1].dtype == np.float64
    np.testing.assert_allclose(batch[0], scalar[0], rtol=2e-7)
    np.testing.assert_allclose(batch[1], scalar[1], rtol=2e-7)


def test_scalar_pool_and_pickled_wrapper_keep_order_and_values():
    coords = np.random.RandomState(22).normal(size=(16, 3))
    expected = make_sampler().compute_log_prob(coords, .4)
    with ThreadPool(2) as pool:
        sampler = make_sampler(pool=pool)
        actual = sampler.compute_log_prob(coords, .4)
        restored = pickle.loads(pickle.dumps(sampler.log_prob_fn))
        values = list(pool.map(restored, [(q, .4) for q in coords]))
    for value, reference in zip(actual[:2], expected[:2]):
        np.testing.assert_array_equal(value, reference)
    np.testing.assert_array_equal(np.asarray(values).T, np.asarray(expected[:2]))


def test_named_parameters_still_use_scalar_dictionary_evaluation():
    sampler = PTSampler(16, 3, lambda p: -np.sum(p["xy"] ** 2) - p["z"] ** 2,
                        lambda p: 0., betas=[1, .2, 0], parameter_names={"xy": [0, 1], "z": 2})
    coords = np.random.RandomState(27).normal(size=(16, 3))
    lp, ll, _ = sampler.compute_log_prob(coords, .3)
    expected = -np.sum(coords ** 2, axis=1)
    np.testing.assert_allclose(ll, expected)
    np.testing.assert_allclose(lp, .3 * expected)


def test_singleton_scalar_arrays_and_custom_blob_wrappers_remain_supported():
    sampler = make_sampler()
    coords = np.random.RandomState(2).normal(size=(16, 3))

    def arrays(pair):
        q, beta = pair
        ll = float(log_like(q))
        return np.array([beta * ll]), np.array([ll])

    sampler.log_prob_fn = arrays
    lp, ll, blobs = sampler.compute_log_prob(coords, .5)
    np.testing.assert_allclose(lp, .5 * log_like(coords))
    np.testing.assert_allclose(ll, log_like(coords))
    assert blobs is None

    sampler.log_prob_fn = lambda pair: (pair[1] * log_like(pair[0]), log_like(pair[0]), pair[0][0])
    lp, ll, blobs = sampler.compute_log_prob(coords, .5)
    np.testing.assert_allclose(blobs[:, 0], coords[:, 0])


@pytest.mark.parametrize("vectorize", [False, True])
def test_invalid_positions_and_nan_probabilities_raise(vectorize):
    sampler = make_sampler(vectorize=vectorize)
    for value, message in ((np.inf, "infinite"), (np.nan, "NaN")):
        coords = np.zeros((16, 3))
        coords[0, 0] = value
        with pytest.raises(ValueError, match=message):
            sampler.compute_log_prob(coords, .2)
    sampler.log_prob_fn.logl = lambda coords: np.nan
    with pytest.raises(ValueError, match="returned NaN"):
        sampler.compute_log_prob(np.zeros((16, 3)), .2)


class CountingRandom:
    def __init__(self, random):
        self.random = random
        self.snapshots = 0

    def __getattr__(self, name):
        return getattr(self.random, name)

    def get_state(self):
        self.snapshots += 1
        return self.random.get_state()


@pytest.mark.parametrize("tune", [False, True])
def test_rng_snapshot_frequency_preserves_exported_state(tune):
    with seeded(72) as (rng, swap_seed):
        sampler = make_sampler()
        sampler._random = CountingRandom(sampler._random)
        np.random.seed(swap_seed)
        final = sampler.run_mcmc(rng.normal(size=(3, 16, 3)), 4, 3, tune=tune)
        assert sampler._random.snapshots == (4 * 3 * 3 if tune else 4)
        np.testing.assert_array_equal(final.random_state[1], sampler._random.random.get_state()[1])
        np.testing.assert_array_equal(sampler.backend.random_state[1], final.random_state[1])


def test_batched_and_scalar_sampling_preserve_seeded_trajectories():
    chains = []
    for vectorize in (False, True):
        with seeded(903) as (rng, swap_seed):
            sampler = make_sampler(vectorize=vectorize)
            np.random.seed(swap_seed)
            sampler.run_mcmc(rng.normal(size=(3, 16, 3)), 12, 2)
            chains.append(sampler.get_chain())
    np.testing.assert_array_equal(*chains)


@pytest.mark.parametrize("representation", ["array", "list", "states", "ptstate"])
def test_initial_coordinate_representations_keep_identical_samples(representation):
    expected = None
    for form in ("array", representation):
        with seeded(15) as (rng, swap_seed):
            sampler = make_sampler()
            coords = rng.normal(size=(3, 16, 3))
            before = coords.copy()
            if form == "list":
                initial = coords.tolist()
            elif form == "states":
                initial = [State(q) for q in coords]
            elif form == "ptstate":
                initial = PTState(coords)
            else:
                initial = coords
            np.random.seed(swap_seed)
            sampler.run_mcmc(initial, 6, 2)
            np.testing.assert_array_equal(coords, before)
            if expected is None:
                expected = sampler.get_chain()
            else:
                np.testing.assert_array_equal(sampler.get_chain(), expected)
