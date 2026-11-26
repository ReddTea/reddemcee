"""MBAR as a public sampler API over actual production backends."""

import numpy as np
import pytest

from reddemcee import PTSampler, get_mbar
from reddemcee.evidence import get_mbar as array_mbar


def log_like(x):
    return -0.5 * np.sum(x ** 2) - 0.5 * len(x) * np.log(2 * np.pi)


def log_prior(x):
    return -0.5 * np.sum(x ** 2) / 4 - 0.5 * len(x) * np.log(8 * np.pi)


@pytest.fixture(autouse=True)
def preserve_swap_random_state():
    previous = np.random.get_state()
    np.random.seed(286)
    try:
        yield
    finally:
        np.random.set_state(previous)


def make_sampler(betas=None, **kwargs):
    sampler = PTSampler(16, 2, log_like, log_prior,
                        betas=[1, 0.5, 0.15, 0] if betas is None else betas,
                        **kwargs)
    sampler._random.seed(72)
    return sampler


def test_mbar_is_available_as_a_public_standalone_estimator():
    assert get_mbar is array_mbar
    value, error = get_mbar([1, 0.3, 0], np.full((3, 12, 4), -7.0))
    assert value == pytest.approx(-7.0)
    assert error == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("storage", ["memory", "hdf"])
@pytest.mark.parametrize("nsteps", [1, 3])
def test_mbar_reads_retained_production_without_probability_calls(tmp_path, storage, nsteps):
    options = {"adapt_mode": "NONE"}
    if storage == "hdf":
        pytest.importorskip("h5py")
        from reddemcee.hdf import PTHDFBackend
        options["backend"] = PTHDFBackend(tmp_path / "production.h5")
    sampler = make_sampler(**options)
    initial = np.random.RandomState(16).normal(0, 2, (4, 16, 2))
    sampler.run_mcmc(initial, 16, nsteps=nsteps)
    before_betas = sampler.get_betas().copy()
    before_likelihoods = sampler.get_log_like().copy()
    retained_betas = sampler.get_betas(discard=2, thin=2)
    retained_likelihoods = sampler.get_log_like(discard=2, thin=2)
    options = dict(mode="spe", batch_size=3, spe_kernel="parzen",
                   return_diagnostics=True, tol=1e-9, maxiter=200,
                   overlap_threshold=0.0)
    expected = get_mbar(retained_betas, retained_likelihoods, **options)

    # Postprocessing must work even when the model is no longer available.
    def unavailable(*args, **kwargs):
        raise AssertionError("MBAR must use stored log likelihoods")

    sampler.compute_log_prob = unavailable
    sampler.log_prob_fn = unavailable
    actual = sampler.get_evidence_mbar(discard=2, thin=2, **options)
    np.testing.assert_allclose(actual[:2], expected[:2], rtol=1e-12)
    assert actual[2]["nsweeps"] == retained_likelihoods.shape[1]
    assert actual[2]["converged"]
    assert actual[2]["mcse_reliable"]
    np.testing.assert_allclose(actual[2]["log_normalizers"], expected[2]["log_normalizers"])
    np.testing.assert_array_equal(sampler.get_betas(), before_betas)
    np.testing.assert_array_equal(sampler.get_log_like(), before_likelihoods)

    if storage == "hdf":
        # An HDF-backed sampler reopened in a fresh object must expose the
        # same evidence API without resampling or reevaluating the model.
        reopened = PTHDFBackend(tmp_path / "production.h5", read_only=True)
        reader = make_sampler(backend=reopened, adapt_mode="NONE")
        reader.log_prob_fn = unavailable
        reader.compute_log_prob = unavailable
        np.testing.assert_allclose(
            reader.get_evidence_mbar(discard=2, thin=2, **options)[:2],
            actual[:2], rtol=1e-12)


def test_mbar_rejects_retained_adaptation_and_accepts_frozen_production():
    sampler = make_sampler(adapt_mode="SAR", adapt_nu=10)
    initial = np.random.RandomState(48).normal(0, 2, (4, 16, 2))
    state = sampler.run_mcmc(initial, 12)
    assert np.ptp(sampler.get_betas()[1]) > 1e-8
    with pytest.raises(ValueError, match="fixed ladder"):
        sampler.get_evidence_mbar()
    sampler.select_adjustment("NONE")
    state = sampler.run_mcmc(state, 12)
    sampler.reset()
    sampler.run_mcmc(state, 16)
    value, error = sampler.get_evidence_mbar()
    assert np.isfinite(value)
    assert np.isfinite(error) and error > 0


def test_mbar_partial_ladder_is_an_explicit_normalizer_ratio():
    sampler = make_sampler(betas=[1, 0.5, 0.15], adapt_mode="NONE")
    sampler.run_mcmc(np.random.RandomState(18).normal(0, 2, (3, 16, 2)), 16)
    with pytest.raises(ValueError, match="beta=0 and beta=1"):
        sampler.get_evidence_mbar()
    value, error, diagnostics = sampler.get_evidence_mbar(
        ba=0, bb=1, allow_partial=True, return_diagnostics=True)
    assert np.isfinite(value) and np.isfinite(error)
    assert not diagnostics["absolute_evidence"]
    assert diagnostics["beta_min"] == pytest.approx(0.15)
    assert diagnostics["beta_max"] == pytest.approx(0.5)


def test_mbar_requires_at_least_three_retained_iterations():
    sampler = make_sampler(adapt_mode="NONE")
    sampler.run_mcmc(np.random.RandomState(18).normal(0, 2, (4, 16, 2)), 8)
    with pytest.raises(ValueError, match="three sweeps"):
        sampler.get_evidence_mbar(discard=6)


@pytest.mark.statistical
def test_mbar_recovers_normalized_gaussian_evidence_after_adaptation():
    sampler = make_sampler(betas=[1, 0.7, 0.4, 0.2, 0.07, 0],
                           adapt_mode="SAR", adapt_nu=10)
    initial = np.random.RandomState(81).normal(0, 2, (6, 16, 2))
    state = sampler.run_mcmc(initial, 150)
    sampler.select_adjustment("NONE")
    state = sampler.run_mcmc(state, 100)
    sampler.reset()
    sampler.run_mcmc(state, 600)
    value, error, diagnostics = sampler.get_evidence_mbar(return_diagnostics=True)
    # Integral N(theta;0,I) N(theta;0,4I) dtheta = N(0;0,5I).
    expected = -np.log(10 * np.pi)
    assert error > 0 and np.isfinite(error)
    assert abs(value - expected) < max(5 * error, 0.04)
    assert diagnostics["absolute_evidence"]
    assert diagnostics["converged"]
    assert diagnostics["overlap_spectral_gap"] > 1e-3
