"""Production solver controls and convergence diagnostics for MBAR."""
from types import SimpleNamespace

import numpy as np
import pytest

import reddemcee.evidence as evidence


def constant_samples():
    return np.array([0., .2, 1.]), np.full((3, 12, 4), -3.)


@pytest.mark.parametrize("tol", [0, -1, np.nan, np.inf, -np.inf, True,
                                 [1e-10], 1 + 2j, "nonsense"])
def test_invalid_solver_tolerance_is_rejected(tol):
    with pytest.raises(ValueError, match="tol must be a finite positive scalar"):
        evidence.get_mbar(*constant_samples(), tol=tol)


@pytest.mark.parametrize("maxiter", [0, -1, True, np.bool_(False), 1.5, 10.,
                                     np.nan, np.inf, [10], 10 ** 100])
def test_invalid_solver_budget_is_rejected(maxiter):
    with pytest.raises(ValueError, match="maxiter must be a positive integer"):
        evidence.get_mbar(*constant_samples(), maxiter=maxiter)


@pytest.mark.parametrize("threshold", [-.1, 1., np.nan, np.inf, True, [.1]])
def test_invalid_overlap_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="overlap_threshold must be a finite scalar"):
        evidence.get_mbar(*constant_samples(), overlap_threshold=threshold)


def test_numpy_scalar_solver_controls_and_diagnostics_are_supported():
    value, error, diagnostics = evidence.get_mbar(
        *constant_samples(), tol=np.float64(1e-10), maxiter=np.int64(100),
        overlap_threshold=np.float64(.001), return_diagnostics=True)
    assert value == pytest.approx(-3)
    assert error == pytest.approx(0, abs=1e-12)
    assert diagnostics["score_residual"] < 1e-10
    assert isinstance(diagnostics["solver_success"], bool)
    assert isinstance(diagnostics["solver_status"], int)
    assert diagnostics["solver_message"]


def solver_result(x, success=False):
    return SimpleNamespace(x=np.asarray(x), nfev=2, success=success,
                           status=1 if success else 4,
                           message="test solver status")


def test_full_balance_check_includes_anchored_state(monkeypatch):
    # Three identical sampled states have known f=(0,0,0). Perturb both
    # unanchored states by2e-8: each reduced score is only6.67e-9, but their
    # common anchored state's imbalance is1.33e-8, above the residual limit.
    monkeypatch.setattr(evidence, "root", lambda *args, **kwargs:
                        solver_result([2e-8, 2e-8]))
    with pytest.raises(RuntimeError, match="MBAR failed to converge: residual"):
        evidence.get_mbar(*constant_samples())


def test_converged_balance_can_override_solver_lack_of_progress(monkeypatch):
    # MINPACK can signal lack of progress at an already balanced root.
    monkeypatch.setattr(evidence, "root", lambda *args, **kwargs:
                        solver_result([0., 0.]))
    value, _, diagnostics = evidence.get_mbar(*constant_samples(), return_diagnostics=True)
    assert value == pytest.approx(-3)
    assert diagnostics["converged"]
    assert not diagnostics["solver_success"]
    assert diagnostics["solver_status"] == 4
    assert diagnostics["solver_message"] == "test solver status"


def test_nonfinite_score_fails_explicitly_with_finite_solver_parameters(monkeypatch):
    monkeypatch.setattr(evidence, "root", lambda *args, **kwargs:
                        solver_result([0., 0.], success=True))
    original = evidence.logsumexp
    monkeypatch.setattr(evidence, "logsumexp", lambda *args, **kwargs:
                        original(*args, **kwargs) * np.nan)
    with pytest.raises(RuntimeError, match="nonfinite score or Hessian"):
        evidence.get_mbar(*constant_samples())


def test_nonfinite_hessian_fails_even_when_score_is_finite(monkeypatch):
    monkeypatch.setattr(evidence, "root", lambda *args, **kwargs:
                        solver_result([0., 0.], success=True))
    original = evidence.np.diag
    monkeypatch.setattr(evidence.np, "diag", lambda *args, **kwargs:
                        original(*args, **kwargs) * np.nan)
    with pytest.raises(RuntimeError, match="nonfinite score or Hessian"):
        evidence.get_mbar(*constant_samples())
