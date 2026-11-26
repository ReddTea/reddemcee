"""Portable numerical MBAR regressions, independent of benchmark helpers."""

import numpy as np
import pytest

from reddemcee.evidence import get_bar, get_mbar


def gaussian_samples(seed=17, beta=None, sweeps=400, walkers=16, prior_variance=4.):
    """Exact draws for N(0,a) prior and likelihood exp(-theta**2/2)."""
    if beta is None:
        beta = np.array([1., .7, .4, .2, .08, 0.])
    beta = np.asarray(beta)
    variance = prior_variance / (1 + prior_variance * beta)
    rng = np.random.RandomState(seed)
    theta = rng.normal(size=(beta.size, sweeps, walkers)) * np.sqrt(variance[:, None, None])
    return np.repeat(beta[:, None], sweeps, axis=1), -.5 * theta ** 2


@pytest.mark.parametrize("constant", [-1e5, 2e4, -2.])
def test_constant_likelihood_has_exact_stable_evidence(constant):
    beta = np.array([1., .4, .02, 0.])
    likelihoods = np.full((4, 30, 8), constant)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        value, error, diagnostics = get_mbar(beta, likelihoods, return_diagnostics=True)
    assert value == pytest.approx(constant)
    assert error == pytest.approx(0, abs=1e-10)
    assert diagnostics["absolute_evidence"]
    assert diagnostics["converged"]
    np.testing.assert_allclose(diagnostics["log_normalizers"], beta[::-1] * constant,
                               rtol=1e-12, atol=1e-10)


def test_iid_gaussian_evidence_matches_analytic_normalizer():
    beta, likelihoods = gaussian_samples()
    value, error, diagnostics = get_mbar(beta, likelihoods, return_diagnostics=True)
    exact = -.5 * np.log(5)
    assert np.isfinite(error) and error > 0
    assert abs(value - exact) < 5 * error
    assert diagnostics["score_residual"] < 1e-8
    overlap = np.asarray(diagnostics["overlap_matrix"])
    np.testing.assert_allclose(overlap.sum(axis=1), 1, atol=1e-8)


def test_orientation_and_common_log_likelihood_offset_preserve_results():
    beta, likelihoods = gaussian_samples()
    expected = get_mbar(beta, likelihoods)
    np.testing.assert_allclose(get_mbar(beta[::-1], likelihoods[::-1]), expected,
                               rtol=1e-12, atol=1e-12)
    shifted = get_mbar(beta, likelihoods - 1e5)
    np.testing.assert_allclose(shifted, (expected[0] - 1e5, expected[1]),
                               rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("ascending", [False, True])
def test_partial_normalizer_ratio_requires_explicit_opt_in(ascending):
    beta = np.array([1., .8, .2, 0.])
    likelihoods = np.full((4, 20, 3), -7.)
    if ascending:
        beta, likelihoods = beta[::-1], likelihoods[::-1]
    with pytest.raises(ValueError, match="beta=0 and beta=1"):
        get_mbar(beta, likelihoods, ba=1, bb=2)
    value, error, diagnostics = get_mbar(
        beta, likelihoods, ba=1, bb=2, allow_partial=True, return_diagnostics=True)
    assert value == pytest.approx(-7 * .6)
    assert error == pytest.approx(0, abs=1e-12)
    assert not diagnostics["absolute_evidence"]
    assert diagnostics["beta_min"] == pytest.approx(.2)
    assert diagnostics["beta_max"] == pytest.approx(.8)


def test_two_temperature_mbar_matches_bar_including_mcse():
    beta, likelihoods = gaussian_samples(beta=[1., 0.], prior_variance=1.)
    np.testing.assert_allclose(get_mbar(beta, likelihoods), get_bar(beta, likelihoods),
                               rtol=2e-8, atol=2e-10)


def test_duplicated_walkers_do_not_invent_independent_information():
    beta, likelihoods = gaussian_samples(sweeps=120, walkers=8)
    expected = get_mbar(beta, likelihoods)
    duplicated = np.repeat(likelihoods, 3, axis=2)
    np.testing.assert_allclose(get_mbar(beta, duplicated), expected,
                               rtol=2e-8, atol=2e-10)


def test_poor_overlap_marks_mcse_unavailable():
    # Constant groups have no observed score variation. A local MCSE alone
    # would falsely be zero despite the absence of the necessary bridge tails.
    likelihoods = np.empty((2, 40, 4))
    likelihoods[0], likelihoods[1] = -20, 0
    with pytest.warns(RuntimeWarning, match="small overlap spectral gap"):
        value, error, diagnostics = get_mbar(
            [0., 1.], likelihoods, return_diagnostics=True)
    assert np.isfinite(value)
    assert error == np.inf
    assert not diagnostics["mcse_reliable"]
    assert diagnostics["overlap_spectral_gap"] < 1e-3


def test_numerically_disconnected_ladder_is_rejected():
    likelihoods = np.empty((2, 20, 4))
    likelihoods[0], likelihoods[1] = -1e6, 0
    with pytest.raises(ValueError, match="overlap"):
        get_mbar([0., 1.], likelihoods)


@pytest.mark.parametrize("fixed_ladder", [True, False])
def test_drifting_ladder_is_rejected_even_if_claimed_fixed(fixed_ladder):
    beta, likelihoods = gaussian_samples(sweeps=12, walkers=2)
    beta[2] += np.linspace(0, .01, 12)
    with pytest.raises(ValueError, match="fixed ladder"):
        get_mbar(beta, likelihoods, fixed_ladder=fixed_ladder)
