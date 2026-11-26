"""Amortized backend growth without changing public history lengths or values."""

import gc
import copy
import pickle
import weakref

import numpy as np
import pytest

from reddemcee.backend import Backend_plus, PTBackend
from reddemcee.state import PTState, State


def state_at(step, dtype=np.float64, blobs=False):
    return State(np.full((4, 2), step + 0.25, dtype=dtype),
                 log_prob=np.full(4, -step - 1, dtype=dtype),
                 log_like=np.full(4, -step - 2, dtype=dtype), beta=1 / (step + 1),
                 blobs=np.full((4, 2), step + 0.5, dtype=dtype) if blobs else None)


def legacy_roundtrip(backend, monkeypatch):
    """Serialize the public-only state written before capacity was introduced."""
    capacity_keys = {"_buffers", "_capacity", "_history_buffers", "_history_capacity"}

    def legacy_state(self):
        return {name: value for name, value in self.__dict__.items()
                if name not in capacity_keys}

    with monkeypatch.context() as patch:
        patch.setattr(Backend_plus, "__getstate__", legacy_state)
        patch.setattr(PTBackend, "__getstate__", legacy_state)
        payload = pickle.dumps(backend)
    return pickle.loads(payload)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("blobs", [False, True])
def test_repeated_growth_preserves_history_and_requested_endpoints(dtype, blobs):
    backend = Backend_plus(dtype=dtype)
    backend.reset(4, 2)
    for step in range(33):
        state = state_at(step, dtype=dtype, blobs=blobs)
        backend.grow(1, state.blobs)
        assert len(backend.chain) == step + 1
        assert step + 1 <= backend._capacity < 2 * (step + 1) or step == 0
        backend.save_step(state, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(backend.chain[-1], state.coords)
        assert backend.beta_history[-1] == pytest.approx(state.beta)
    assert backend.chain.flags.c_contiguous
    assert backend.chain.dtype == dtype
    expected = np.arange(33, dtype=dtype) + 0.25
    np.testing.assert_array_equal(backend.get_chain()[:, 0, 0], expected)
    np.testing.assert_array_equal(backend.get_log_like()[:, 0], -np.arange(33) - 2)
    np.testing.assert_array_equal(backend.get_chain(discard=2, thin=3, flat=True),
                                  backend.chain[4:33:3].reshape(-1, 2))
    if blobs:
        assert backend.blobs.dtype == dtype
        np.testing.assert_array_equal(backend.get_blobs()[:, 0, 0], np.arange(33) + 0.5)


def test_large_initial_reservation_is_exact_and_growth_within_capacity_does_not_allocate():
    backend = Backend_plus()
    backend.reset(4, 2)
    backend.grow(100, None)
    assert backend._capacity == len(backend.chain) == 100
    initial = dict(backend._buffers)
    backend.grow(30, None)
    assert all(backend._buffers[name] is value for name, value in initial.items())
    backend.iteration = 100
    backend.grow(1, None)
    assert backend._capacity == 200
    assert len(backend.chain) == 101
    reserved = dict(backend._buffers)
    backend.iteration = 101
    backend.grow(1, None)
    assert len(backend.chain) == 102
    assert all(backend._buffers[name] is value for name, value in reserved.items())
    assert backend.chain.base is backend._buffers["chain"]


def test_growth_preserves_old_public_views_and_detaches_when_reallocated():
    backend = Backend_plus()
    backend.reset(4, 2)
    backend.grow(2, None)
    for step in range(2):
        backend.save_step(state_at(step), np.ones(4, dtype=bool))
    old = backend.get_chain()
    snapshot = old.copy()
    backend.grow(1, None)
    assert not np.shares_memory(old, backend.chain)
    backend.save_step(state_at(2), np.ones(4, dtype=bool))
    np.testing.assert_array_equal(old, snapshot)
    np.testing.assert_array_equal(backend.get_chain()[:2], snapshot)
    # The next reservation fits capacity. An old view of completed samples is
    # unchanged by appending a new step to the shared allocation.
    old = backend.get_chain()
    snapshot = old.copy()
    backend.grow(1, None)
    assert np.shares_memory(old, backend.chain)
    backend.save_step(state_at(3), np.ones(4, dtype=bool))
    np.testing.assert_array_equal(old, snapshot)


def test_blobs_can_be_initialized_inside_existing_capacity_before_first_save():
    backend = Backend_plus()
    backend.reset(4, 2)
    backend.grow(8, None)
    state = state_at(0, blobs=True)
    backend.grow(2, state.blobs)
    assert backend.blobs.shape == (8, 4, 2)
    backend.save_step(state, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(backend.get_blobs()[0], state.blobs)
    with pytest.raises(ValueError, match="inconsistent use of blobs"):
        backend.grow(1, None)


def test_structured_object_blobs_keep_dtype_and_payloads_during_growth():
    backend = Backend_plus()
    backend.reset(4, 2)
    blobs = np.empty(4, dtype=[("index", np.int32), ("metadata", object)])
    for step in range(5):
        blobs["index"] = step
        for walker in range(4):
            blobs["metadata"][walker] = {"step": step, "walker": walker}
        state = state_at(step)
        state.blobs = blobs.copy()
        backend.grow(1, state.blobs)
        backend.save_step(state, np.ones(4, dtype=bool))
    assert backend.blobs.dtype == blobs.dtype
    np.testing.assert_array_equal(backend.get_blobs()["index"][:, 0], np.arange(5))
    assert backend.get_blobs()["metadata"][0, 2] == {"step": 0, "walker": 2}


@pytest.mark.parametrize("swap,distance", [(False, False), (True, False), (False, True), (True, True)])
def test_parallel_history_growth_preserves_enabled_flags_and_visible_lengths(swap, distance):
    backend = PTBackend(dtype=np.float32)
    backend.reset(3, 4, 2, tsw_hist=swap, smd_hist=distance)
    for step in range(9):
        backend.grow(1)
        for enabled, name in [(swap, "tsw_history"), (distance, "smd_history")]:
            history = getattr(backend, name)
            assert len(history) == step + 1 if enabled else len(history) == 0
            if enabled:
                history[step] = step + 0.5
        backend.iteration += 1
    for enabled, name in [(swap, "tsw_history"), (distance, "smd_history")]:
        if enabled:
            np.testing.assert_array_equal(getattr(backend, name)[:, 0], np.arange(9) + 0.5)
            assert getattr(backend, name).dtype == np.float32
    reserved = dict(backend._history_buffers)
    backend.grow(1)
    assert all(backend._history_buffers[name] is value for name, value in reserved.items())


def test_reset_releases_private_capacity_allocations_and_last_sample_stays_detached():
    backend = Backend_plus()
    backend.reset(4, 2)
    for step in range(5):
        backend.grow(1, None)
        backend.save_step(state_at(step), np.ones(4, dtype=bool))
    last = backend.get_last_sample()
    references = [weakref.ref(buffer) for buffer in backend._buffers.values()]
    assert not np.shares_memory(last.coords, backend._buffers["chain"])
    backend.reset(4, 2)
    gc.collect()
    assert backend._capacity == 0
    assert all(reference() is None for reference in references)
    np.testing.assert_array_equal(last.coords, 4.25)


def test_reset_releases_private_parallel_history_buffers_when_disabled():
    backend = PTBackend()
    backend.reset(3, 4, 2, tsw_hist=True, smd_hist=True)
    for step in range(5):
        backend.grow(1)
        backend.iteration += 1
    references = [weakref.ref(buffer) for buffer in backend._history_buffers.values()]
    backend.reset(3, 4, 2, tsw_hist=False, smd_hist=False)
    gc.collect()
    assert backend._history_capacity == 0
    assert backend._history_buffers == {}
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("clone", [copy.deepcopy, lambda backend: pickle.loads(pickle.dumps(backend))])
def test_copied_backend_growth_keeps_public_storage_authoritative(clone):
    backend = Backend_plus()
    backend.reset(4, 2)
    for step in range(3):
        state = state_at(step, blobs=True)
        backend.grow(1, state.blobs)
        backend.save_step(state, np.ones(4, dtype=bool))

    restored = clone(backend)
    for name in ("chain", "log_prob", "log_like", "beta_history", "blobs"):
        getattr(restored, name)[0] = 99
    for step in range(3, 6):
        state = state_at(step, blobs=True)
        restored.grow(1, state.blobs)
        restored.save_step(state, np.ones(4, dtype=bool))
    for name in ("chain", "log_prob", "log_like", "beta_history", "blobs"):
        np.testing.assert_array_equal(getattr(restored, name)[0], 99)
        assert np.shares_memory(getattr(restored, name), restored._buffers[name])
    np.testing.assert_array_equal(backend.chain[0], 0.25)


@pytest.mark.parametrize("clone", [copy.deepcopy, lambda backend: pickle.loads(pickle.dumps(backend))])
def test_copied_parallel_history_growth_keeps_public_storage_authoritative(clone):
    backend = PTBackend()
    backend.reset(3, 4, 2, tsw_hist=True, smd_hist=True)
    for step in range(3):
        backend.grow(1)
        backend.tsw_history[step] = step
        backend.smd_history[step] = step
        backend.iteration += 1
    restored = clone(backend)
    restored.tsw_history[0] = 99
    restored.smd_history[0] = 98
    for step in range(3, 6):
        restored.grow(1)
        restored.iteration += 1
    np.testing.assert_array_equal(restored.tsw_history[0], 99)
    np.testing.assert_array_equal(restored.smd_history[0], 98)
    np.testing.assert_array_equal(backend.tsw_history[0], 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("blobs", [False, True])
def test_legacy_backend_pickle_resumes_with_public_history_capacity(dtype, blobs, monkeypatch):
    backend = Backend_plus(dtype=dtype)
    backend.reset(4, 2)
    for step in range(3):
        state = state_at(step, dtype=dtype, blobs=blobs)
        backend.grow(1, state.blobs)
        backend.save_step(state, np.ones(4, dtype=bool))
    backend.grow(2, state.blobs)
    restored = legacy_roundtrip(backend, monkeypatch)
    assert restored._capacity == len(restored.chain) == 5
    assert restored.iteration == 3
    assert restored.chain.dtype == dtype
    for name, buffer in restored._buffers.items():
        assert buffer is getattr(restored, name)
    for step in range(3, 8):
        state = state_at(step, dtype=dtype, blobs=blobs)
        restored.grow(1, state.blobs)
        restored.save_step(state, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(restored.get_chain()[:, 0, 0], np.arange(8) + 0.25)
    np.testing.assert_array_equal(restored.get_log_like()[:, 0], -np.arange(8) - 2)
    np.testing.assert_array_equal(restored.accepted, 8)
    if blobs:
        np.testing.assert_array_equal(restored.get_blobs()[:, 0, 0], np.arange(8) + 0.5)


@pytest.mark.parametrize("swap,distance", [(False, False), (True, False), (False, True), (True, True)])
def test_legacy_parallel_backend_pickle_resumes_nested_storage(swap, distance, monkeypatch):
    backend = PTBackend(dtype=np.float32)
    backend.reset(2, 4, 2, tsw_hist=swap, smd_hist=distance)

    def append(target, step):
        target.grow(1)
        states = PTState([state_at(step + t, dtype=np.float32, blobs=True)
                          for t in range(2)])
        states.random_state = np.random.RandomState(step).get_state()
        for child, state in zip(target, states):
            child.grow(1, state.blobs)
            child.save_step(state, np.ones(4, dtype=bool))
        target.save_step(states, np.full(1, step + 0.5), smd=np.full(1, step + 0.25))

    for step in range(3):
        append(backend, step)
    backend.grow(2)
    for child in backend:
        child.grow(2, np.zeros((4, 2), dtype=np.float32))
    restored = legacy_roundtrip(backend, monkeypatch)
    assert restored.iteration == 3
    assert restored._history_capacity == (5 if swap or distance else 0)
    assert restored.tsw_history_bool == swap
    assert restored.smd_history_bool == distance
    for name, buffer in restored._history_buffers.items():
        assert buffer is getattr(restored, name)
    for child in restored:
        assert child._capacity == len(child.chain) == 5
        assert child._buffers["chain"] is child.chain
    for step in range(3, 8):
        append(restored, step)
    np.testing.assert_array_equal(restored.get_chain()[:, :, 0, 0],
                                  np.arange(8)[None, :] + np.arange(2)[:, None] + 0.25)
    if swap:
        np.testing.assert_array_equal(restored.get_tsw()[:, 0], np.arange(8) + 0.5)
    if distance:
        np.testing.assert_array_equal(restored.get_smd()[:, 0], np.arange(8) + 0.25)
    np.testing.assert_array_equal(restored.ts_accepted, np.sum(np.arange(8) + 0.5))


@pytest.mark.parametrize("backend_class", [Backend_plus, PTBackend])
def test_legacy_uninitialized_backend_pickle_can_still_be_reset(backend_class, monkeypatch):
    restored = legacy_roundtrip(backend_class(), monkeypatch)
    assert not restored.initialized
    if backend_class is PTBackend:
        restored.reset(2, 4, 2, tsw_hist=True)
        restored.grow(1)
        assert restored.tsw_history.shape == (1, 1)
    else:
        restored.reset(4, 2)
        restored.grow(1, None)
        assert restored.chain.shape == (1, 4, 2)


def test_parallel_swap_updates_written_endpoint_inside_reserved_capacity():
    backend = PTBackend()
    backend.reset(2, 4, 2, tsw_hist=True, smd_hist=True)
    backend.grow(3)
    states = PTState([state_at(0, blobs=True), state_at(1, blobs=True)])
    states.random_state = np.random.RandomState(17).get_state()
    for child, state in zip(backend, states):
        child.grow(6, state.blobs)
        child.chain[:] = -123
        child.save_step(state, np.ones(4, dtype=bool))
        state.coords += 10
        state.log_prob += 20
        state.log_like += 30
        state.blobs += 40
        state.beta = 0.125
    backend.save_step(states, np.ones(1), smd=np.full(1, 0.5))
    for child, state in zip(backend, states):
        stored = child.get_last_sample()
        for name in ("coords", "log_prob", "log_like", "blobs"):
            np.testing.assert_array_equal(getattr(stored, name), getattr(state, name))
        assert stored.beta == 0.125
        np.testing.assert_array_equal(child.chain[1:], -123)
    np.testing.assert_array_equal(backend.get_tsw(), np.ones((1, 1)))
    np.testing.assert_array_equal(backend.get_smd(), np.full((1, 1), 0.5))
