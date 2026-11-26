"""Exact-result regressions for the move-array allocation optimizations.

The small reference routines retain the pre-optimization indexing operations
so changes in selection order, arithmetic, or random draws remain visible.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from reddemcee.moves import Move, PTMove, RedBlueMove, StretchMove
from reddemcee.state import PTState, State


def evaluate(coords, beta):
    log_like = -.5 * np.sum((coords - .2) ** 2, axis=1)
    log_prior = -.1 * np.sum(coords ** 2, axis=1)
    return beta * log_like + log_prior, log_like, np.column_stack((coords[:, 0], log_like))


def state_at(coords, beta=.3):
    lp, ll, blobs = evaluate(coords, beta)
    return State(coords.copy(), log_prob=lp, log_like=ll, beta=beta, blobs=blobs)


def assert_state_equal(actual, expected):
    for name in ("coords", "log_prob", "log_like", "blobs"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.beta == expected.beta


def assert_random_state_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


class LegacyUpdate:
    def update(self, old_state, new_state, accepted, subset=None):
        if subset is None:
            subset = np.ones(len(old_state.coords), dtype=bool)
        subset_indices = np.where(subset)[0]
        accepted_subset = accepted[subset]
        m1 = subset_indices[accepted_subset]
        m2 = np.where(accepted_subset)[0]
        for name in ("coords", "log_prob", "log_like", "blobs"):
            old = getattr(old_state, name)
            new = getattr(new_state, name)
            if new is not None:
                old[m1] = new[m2]
        return old_state


class LegacyRedBlueMove(LegacyUpdate):
    def propose(self, model, state):
        nwalkers = len(state.coords)
        self.setup(state.coords)
        accepted = np.zeros(nwalkers, dtype=bool)
        indices = np.arange(nwalkers) % self.nsplits
        if self.randomize_split:
            model.random.shuffle(indices)
        for split in range(self.nsplits):
            selected = indices == split
            sample = state.coords[selected]
            complement = state.coords[~selected]
            q, factors = self.get_proposal(sample, complement, model.random)
            lp, ll, blobs = model.compute_log_prob_fn(q, state.beta)
            log_ratio = factors + lp - state.log_prob[selected]
            accepted[selected] = log_ratio > np.log(model.random.rand(len(log_ratio)))
            new = State(q, log_prob=lp, log_like=ll, beta=state.beta, blobs=blobs)
            self.update(state, new, accepted, selected)
        return state, accepted


class LegacyStretchMove(LegacyRedBlueMove, StretchMove):
    def get_proposal(self, s, c, random):
        ns, nc = len(s), len(c)
        zz = ((self.a - 1.) * random.rand(ns) + 1) ** 2. / self.a
        factors = (s.shape[1] - 1.) * np.log(zz)
        indices = random.randint(nc, size=ns)
        return c[indices] - (c[indices] - s) * zz[:, None], factors


@pytest.mark.parametrize("factory,reference_factory", [
    (StretchMove, LegacyStretchMove),
])
@pytest.mark.parametrize("nwalkers", [32, 127])
def test_move_trajectories_and_random_draws_match_previous_indexing(factory, reference_factory, nwalkers):
    coords = np.random.RandomState(9).normal(size=(nwalkers, 3))
    actual, expected = state_at(coords), state_at(coords)
    actual_model = SimpleNamespace(random=np.random.RandomState(15), compute_log_prob_fn=evaluate)
    expected_model = SimpleNamespace(random=np.random.RandomState(15), compute_log_prob_fn=evaluate)
    move, reference = factory(), reference_factory()
    for _ in range(20):
        actual, accepted = move.propose(actual_model, actual)
        expected, expected_accepted = reference.propose(expected_model, expected)
        np.testing.assert_array_equal(accepted, expected_accepted)
        assert_state_equal(actual, expected)
        assert_random_state_equal(actual_model.random.get_state(), expected_model.random.get_state())


@pytest.mark.parametrize("subset_mode", ["full", "even", "empty"])
@pytest.mark.parametrize("acceptance", [0., .4, 1.])
def test_update_matches_previous_indexing_for_empty_partial_and_full_selections(subset_mode, acceptance):
    rng = np.random.RandomState(18)
    actual = state_at(rng.normal(size=(31, 5)))
    expected = State(actual, copy=True)
    if subset_mode == "full":
        subset, size = None, 31
    else:
        subset = np.arange(31) % 2 == 0 if subset_mode == "even" else np.zeros(31, dtype=bool)
        size = subset.sum()
    proposal = state_at(rng.normal(size=(size, 5)))
    accepted = rng.rand(31) < acceptance
    Move().update(actual, proposal, accepted, subset)
    LegacyUpdate().update(expected, proposal, accepted, subset)
    assert_state_equal(actual, expected)


class LegacyPTMove(PTMove):
    def swap(self, state):
        ntemps, nwalkers, _ = state.shape
        counts = np.zeros(ntemps - 1)
        smd = np.zeros(ntemps - 1) if self.smd_hist else None
        dbetas = -np.diff(state.betas)
        for t in self.beta_range(ntemps):
            hot, cold = state[t], state[t - 1]
            ll1, ll2 = hot.log_like, cold.log_like
            dbeta = dbetas[t - 1]
            raccept = np.log(np.random.uniform(size=nwalkers))
            selected = dbeta * (ll1 - ll2) > raccept
            counts[t - 1] = selected.sum()
            aa, bb = hot.coords[selected], cold.coords[selected]
            if self.smd_hist:
                if selected.any():
                    diff = (aa - bb) / self.D_
                    distances = np.sqrt(np.sum(np.square(diff), axis=1))
                    zeros = np.zeros(int(nwalkers - counts[t - 1]))
                    smd[t - 1] = np.mean(np.append(distances, zeros))
                else:
                    smd[t - 1] = 0.
            hot.coords[selected], cold.coords[selected] = bb, aa
            hot.log_prob[selected], cold.log_prob[selected] = (
                cold.log_prob[selected] - dbeta * ll2[selected],
                hot.log_prob[selected] + dbeta * ll1[selected],
            )
            hot.log_like[selected], cold.log_like[selected] = ll2[selected], ll1[selected]
            hot.blobs[selected], cold.blobs[selected] = cold.blobs[selected], hot.blobs[selected]
        return state, counts / nwalkers, smd

    def adapt(self, state, dSs, copy=True):
        betas = deepcopy(state.betas) if copy else state.betas
        delta_ts = np.diff(1 / betas[:-1])
        delta_ts *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(delta_ts) + 1 / betas[0])
        dbetas = betas - state.betas
        state.betas = betas
        for t in range(len(state) - 1, 0, -1):
            state[t].log_prob += state[t].log_like * dbetas[t]
        return state


@pytest.mark.parametrize("smd_history", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_temperature_swap_matches_previous_masks_and_padded_smd(smd_history, dtype):
    rng = np.random.RandomState(6)
    states = [state_at(rng.normal(size=(127, 5)).astype(dtype), beta)
              for beta in np.linspace(1., 0., 8)]
    actual, expected = PTState(states, copy=True), PTState(states, copy=True)
    move, reference = PTMove(), LegacyPTMove()
    move.smd_hist = reference.smd_hist = smd_history
    move.D_ = reference.D_ = np.array([1., .1, 4., .3, .8])
    saved_random_state = np.random.get_state()
    try:
        for iteration in range(12):
            np.random.seed(iteration)
            actual, acceptance, smd = move.swap(actual)
            actual_random_state = np.random.get_state()
            np.random.seed(iteration)
            expected, expected_acceptance, expected_smd = reference.swap(expected)
            for a, e in zip(actual, expected):
                assert_state_equal(a, e)
            np.testing.assert_array_equal(acceptance, expected_acceptance)
            np.testing.assert_array_equal(smd, expected_smd)
            assert_random_state_equal(actual_random_state, np.random.get_state())
    finally:
        np.random.set_state(saved_random_state)


@pytest.mark.parametrize("accept_all", [False, True])
def test_temperature_swap_empty_and_full_edges_match_previous_masks(monkeypatch, accept_all):
    states = [state_at(np.full((13, 2), t + 1.), beta)
              for t, beta in enumerate([1., .2, 0.])]
    actual, expected = PTState(states, copy=True), PTState(states, copy=True)
    move, reference = PTMove(), LegacyPTMove()
    move.smd_hist = reference.smd_hist = True
    value = np.finfo(float).tiny if accept_all else 1.
    monkeypatch.setattr(np.random, "uniform", lambda size: np.full(size, value))
    actual, acceptance, smd = move.swap(actual)
    expected, expected_acceptance, expected_smd = reference.swap(expected)
    for a, e in zip(actual, expected):
        assert_state_equal(a, e)
    np.testing.assert_array_equal(acceptance, expected_acceptance)
    np.testing.assert_array_equal(acceptance, float(accept_all))
    np.testing.assert_array_equal(smd, expected_smd)


@pytest.mark.parametrize("copy", [False, True])
def test_ladder_adaptation_matches_previous_beta_copies_exactly(copy):
    rng = np.random.RandomState(73)
    states = [state_at(rng.normal(size=(23, 4)), beta)
              for beta in np.r_[np.geomspace(1., 1.e-3, 7), 0.]]
    actual, expected = PTState(states, copy=True), PTState(states, copy=True)
    move, reference = PTMove(), LegacyPTMove()
    for _ in range(20):
        adjustment = rng.normal(scale=.01, size=6)
        move.adapt(actual, adjustment, copy=copy)
        reference.adapt(expected, adjustment, copy=copy)
        for a, e in zip(actual, expected):
            assert_state_equal(a, e)


def test_zero_ladder_adjustment_preserves_betas_and_log_prob_exactly():
    rng = np.random.RandomState(17)
    states = [state_at(rng.normal(size=(8, 2)), beta)
              for beta in [1., .37, .013, 0.]]
    state = PTState(states, copy=True)
    previous = PTState(state, copy=True)
    result = PTMove().adapt(state, np.zeros(2))
    assert result is state
    for actual, expected in zip(state, previous):
        assert_state_equal(actual, expected)


@pytest.mark.parametrize("nsplits", [0, 1, 2.5])
def test_redblue_rejects_invalid_split_counts(nsplits):
    with pytest.raises(ValueError, match="nsplits"):
        RedBlueMove(nsplits=nsplits)


def test_redblue_rejects_empty_complement_before_updating_state():
    state = state_at(np.ones((1, 2)))
    previous = State(state, copy=True)
    model = SimpleNamespace(random=np.random.RandomState(1), compute_log_prob_fn=evaluate)
    with pytest.raises(RuntimeError, match="nsplits"):
        StretchMove(live_dangerously=True).propose(model, state)
    assert_state_equal(state, previous)
