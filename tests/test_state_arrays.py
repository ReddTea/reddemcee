"""Coordinate coercion and State copy/aliasing contracts."""

import copy

import numpy as np
import pytest

from reddemcee import PTSampler
from reddemcee.moves import Move
from reddemcee.state import PTState, State


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_coordinates_preserve_dtype_and_copy_false_shares_memory(dtype):
    coords = np.arange(12, dtype=dtype).reshape(4, 3)
    state = State(coords)
    assert state.coords is coords
    assert state.coords.dtype == dtype
    state.coords[0, 0] = 42
    assert coords[0, 0] == 42
    clone = State(state, copy=False)
    assert clone.coords is state.coords


def test_copy_true_detaches_numeric_arrays_and_preserves_fortran_layout():
    coords = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(4, 3))
    original = State(coords, log_prob=np.arange(4, dtype=float),
                     log_like=np.arange(4, dtype=float) + 1, blobs=np.arange(8).reshape(4, 2), beta=0.4)
    clone = State(original, copy=True)
    assert clone.coords.flags.f_contiguous
    assert clone.coords.dtype == np.float32
    for name in ("coords", "log_prob", "log_like", "blobs"):
        np.testing.assert_array_equal(getattr(clone, name), getattr(original, name))
        assert not np.shares_memory(getattr(clone, name), getattr(original, name))
        getattr(clone, name).flat[0] = -10
        assert getattr(original, name).flat[0] != -10
    assert clone.beta == original.beta


@pytest.mark.parametrize("structured", [False, True])
def test_object_blobs_are_recursively_copied(structured):
    blobs = np.empty(2, dtype=[("payload", object)] if structured else object)
    payload = blobs["payload"] if structured else blobs
    payload[0], payload[1] = {"nested": [1, 2]}, {"nested": [3, 4]}
    original = State(np.zeros((2, 2)), blobs=blobs)
    clone = State(original, copy=True)
    cloned_payload = clone.blobs["payload"] if structured else clone.blobs
    cloned_payload[0]["nested"].append(99)
    assert payload[0]["nested"] == [1, 2]
    assert cloned_payload[0]["nested"] == [1, 2, 99]


def test_ndarray_subclass_retains_deepcopy_metadata_semantics():
    class MetadataArray(np.ndarray):
        def __deepcopy__(self, memo):
            result = self.copy()
            result.metadata = copy.deepcopy(self.metadata, memo)
            return result

    blobs = np.arange(4).view(MetadataArray)
    blobs.metadata = {"tags": ["original"]}
    clone = State(np.zeros((4, 2)), blobs=blobs, copy=True)
    assert isinstance(clone.blobs, MetadataArray)
    clone.blobs.metadata["tags"].append("clone")
    assert blobs.metadata["tags"] == ["original"]


def test_promoted_integer_coordinate_subclass_still_copies_nested_metadata():
    class MetadataArray(np.ndarray):
        def __array_finalize__(self, original):
            self.metadata = getattr(original, "metadata", None)

        def __deepcopy__(self, memo):
            result = self.copy()
            result.metadata = copy.deepcopy(self.metadata, memo)
            return result

    coords = np.arange(8).reshape(4, 2).view(MetadataArray)
    coords.metadata = {"tags": ["original"]}
    clone = State(coords, copy=True)
    assert clone.coords.dtype.kind == "f"
    clone.coords.metadata["tags"].append("clone")
    assert coords.metadata["tags"] == ["original"]


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint32, np.bool_])
def test_integral_coordinates_are_promoted_before_accepted_moves(dtype):
    coords = np.array([[0, 0], [1, 1]], dtype=dtype)
    original = coords.copy()
    state = State(coords, log_prob=np.zeros(2), log_like=np.zeros(2))
    assert state.coords.dtype.kind == "f"
    proposal = State(np.full((2, 2), 0.75), log_prob=np.ones(2), log_like=np.ones(2))
    Move().update(state, proposal, np.ones(2, dtype=bool))
    np.testing.assert_array_equal(state.coords, 0.75)
    np.testing.assert_array_equal(coords, original)


@pytest.mark.parametrize("as_list", [False, True])
def test_ptstate_numeric_input_and_state_sequence_have_equivalent_values(as_list):
    coords = np.arange(48, dtype=float).reshape(3, 4, 4)
    state = PTState(coords.tolist() if as_list else coords, copy=True)
    clone = PTState(state, copy=True)
    sequence = PTState([State(points) for points in coords], copy=True)
    assert state.shape == clone.shape == sequence.shape == coords.shape
    for t in range(3):
        np.testing.assert_array_equal(state[t].coords, coords[t])
        np.testing.assert_array_equal(clone[t].coords, coords[t])
        np.testing.assert_array_equal(sequence[t].coords, coords[t])
        assert not np.shares_memory(state[t].coords, coords)
        assert not np.shares_memory(clone[t].coords, state[t].coords)


def test_ptstate_nested_integer_lists_promote_to_float():
    state = PTState([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    assert state.shape == (2, 2, 2)
    assert all(part.coords.dtype.kind == "f" for part in state)


@pytest.mark.parametrize("coords", [[], np.empty((0, 4, 2)), [[0, 1], [2, 3]]])
def test_ptstate_rejects_empty_or_wrong_rank(coords):
    with pytest.raises(ValueError):
        PTState(coords)


def test_nested_lists_and_arrays_produce_identical_seeded_sampling():
    initial = np.random.RandomState(271).normal(size=(3, 8, 2))
    previous = np.random.get_state()
    chains = []
    try:
        for coords in (initial, initial.tolist()):
            np.random.seed(182)
            sampler = PTSampler(8, 2, lambda x: -0.5 * np.sum(x * x), lambda x: 0.0,
                                betas=[1, 0.2, 0], adapt_mode="NONE")
            sampler.run_mcmc(coords, 3, 2)
            chains.append(sampler.get_chain())
    finally:
        np.random.set_state(previous)
    np.testing.assert_array_equal(*chains)
