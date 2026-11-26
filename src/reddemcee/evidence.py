"""Evidence estimates for a *fixed*, equilibrated power-posterior ladder.

The unnormalised density at inverse temperature beta is
``prior(theta) * likelihood(theta)**beta``. The prior must be normalised,
and its support must not shrink discontinuously at beta=0. These methods
use only the stored log likelihoods: the prior cancels from density ratios.

All functions return ``(logZ, mcse)`` or, with ``return_diagnostics=True``,
``(logZ, mcse, diagnostics)``. MCSE describes sampling uncertainty, not
quadrature bias or failure to visit modes. Sweeps are the sampling units:
walkers and temperatures within a sweep are kept together, and OBM/HAC
preserves their covariance and serial dependence.

References: Friel, Hurn & Wyse (2014), Statistics and Computing 24,
709--723; Bennett (1976), J. Comput. Phys. 22, 245--268; Shirts & Chodera
(2008), J. Chem. Phys. 129, 124105 (MBAR/reverse logistic likelihood).
"""

import warnings
import operator

import numpy as np
from scipy.optimize import brentq, root
from scipy.special import expit, logsumexp

from .utils import _obm_univariate, _spe_univariate


__all__ = ["get_ti_corrected", "get_bar", "get_mbar"]


def _prepare(B, L, ba, bb, allow_partial):
    L = np.asarray(L, dtype=float)
    B = np.asarray(B, dtype=float)
    if L.ndim != 3:
        raise ValueError("L must have shape (ntemps, nsweeps, nwalkers)")
    if B.ndim == 1:
        B = np.broadcast_to(B[:, None], L.shape[:2])
    if B.shape != L.shape[:2]:
        raise ValueError("B must have shape (ntemps, nsweeps), matching L")
    if L.shape[0] < 2 or L.shape[1] < 3 or L.shape[2] < 1:
        raise ValueError("Need at least two temperatures, three sweeps, and one walker")
    if not np.all(np.isfinite(B)) or not np.all(np.isfinite(L)):
        raise ValueError("Betas and log likelihoods must be finite")
    if not np.allclose(B, B[:, :1], rtol=1e-10, atol=1e-12):
        raise ValueError("Evidence estimation requires a fixed ladder; discard adaptation and use a frozen production run")
    beta = B.mean(axis=1)
    if np.any(beta < 0) or np.any(beta > 1):
        raise ValueError("Inverse temperatures must lie in [0, 1]")
    delta = np.diff(beta)
    if np.all(delta < 0):
        beta, L = beta[::-1], L[::-1]
    elif not np.all(delta > 0):
        raise ValueError("The ladder must be strictly monotone, without duplicate temperatures")
    lo = 0 if ba is None else ba
    hi = beta.size - 1 if bb is None else bb
    if not isinstance(lo, (int, np.integer)) or not isinstance(hi, (int, np.integer)):
        raise ValueError("ba and bb must be integer indices in the ascending ladder")
    if not 0 <= lo < hi < beta.size:
        raise ValueError("Need 0 <= ba < bb < ntemps in ascending-beta order")
    beta, L = beta[lo:hi + 1], L[lo:hi + 1]
    absolute = bool(np.isclose(beta[0], 0, rtol=0, atol=1e-12) and
                    np.isclose(beta[-1], 1, rtol=0, atol=1e-12))
    if not absolute and not allow_partial:
        raise ValueError("Absolute evidence requires beta=0 and beta=1; use allow_partial=True for a log normalising-constant ratio")
    diagnostics = {"absolute_evidence": absolute, "beta_min": float(beta[0]),
                   "beta_max": float(beta[-1]), "nsweeps": L.shape[1],
                   "nwalkers": L.shape[2], "fixed_ladder": True}
    return beta, L, diagnostics


def _mcse(influence, mode, batch_size, spe_kernel):
    if mode == "obm":
        variance = _obm_univariate(influence, batch_size=batch_size)
    elif mode == "spe":
        variance = _spe_univariate(influence, bandwidth=batch_size,
                                   kernel=spe_kernel)
    else:
        raise ValueError("mode must be 'obm' or 'spe'")
    return float(np.sqrt(max(0.0, variance) / influence.size))


def _result(value, error, diagnostics, return_diagnostics):
    result = (float(value), float(error))
    return result + (diagnostics,) if return_diagnostics else result


def get_ti_corrected(B, L, ba=None, bb=None, fixed_ladder=True,
                     mode="obm", batch_size=None, spe_kernel="bartlett",
                     allow_partial=False, return_diagnostics=False):
    """Variance-corrected trapezoidal thermodynamic integration.

    On each interval of width h, use
    ``h*(E_low + E_high)/2 - h**2*(Var_high - Var_low)/12``.
    The derivative identity ``d E_beta(log L)/d beta = Var_beta(log L)``
    needs the pooled variance of *individual* log likelihoods, rather than
    the variance of the ensemble mean. The plug-in variance uses ddof=0.

    ``B`` has shape (ntemps, nsweeps), or (ntemps,); ``L`` has shape
    (ntemps, nsweeps, nwalkers). Both ladder orientations are accepted.
    ``ba``/``bb`` select inclusive indices after sorting beta ascending.
    A partial ladder requires ``allow_partial=True`` and yields
    ``log Z(beta_max) - log Z(beta_min)``. ``fixed_ladder`` is accepted for
    API compatibility; actual drifting ladders are always rejected.

    The returned MCSE is a delta-method sweep-wise influence estimate.
    It excludes the remaining quadrature error, which can still be large
    on coarse ladders; diagnostics include the trapezoidal correction.
    """
    beta, L, diagnostics = _prepare(B, L, ba, bb, allow_partial)
    means = L.mean(axis=(1, 2))
    centered = L - means[:, None, None]
    variances = np.mean(centered ** 2, axis=(1, 2))
    h = np.diff(beta)
    mean_weights = np.zeros(beta.size)
    variance_weights = np.zeros(beta.size)
    mean_weights[:-1] += h / 2
    mean_weights[1:] += h / 2
    variance_weights[:-1] += h ** 2 / 12
    variance_weights[1:] -= h ** 2 / 12
    trapezoidal = float(mean_weights @ means)
    correction = float(variance_weights @ variances)
    value = trapezoidal + correction
    # The derivative of Var(X) contributes (X-mu)^2 - Var(X).
    # Averaging within a sweep retains ensemble dependence.
    influence = (mean_weights @ centered.mean(axis=2) +
                 variance_weights @ (np.mean(centered ** 2, axis=2) -
                                     variances[:, None]))
    error = _mcse(influence, mode, batch_size, spe_kernel)
    diagnostics.update(method="ti_corrected", trapezoidal=trapezoidal,
                       variance_correction=correction, converged=True,
                       mcse_reliable=True)
    return _result(value, error, diagnostics, return_diagnostics)


def _bar_pairs(beta, L, tol, maxiter):
    ratios, influences, overlaps, residuals, iterations = [], [], [], [], []
    for k, delta in enumerate(np.diff(beta)):
        a, b = delta * L[k], delta * L[k + 1]
        # Working relative to a common centre avoids large log-L offsets.
        center = float(np.median(np.concatenate((a.ravel(), b.ravel()))))
        a, b = a - center, b - center

        def score(value):
            return float(np.mean(expit(a - value)) - np.mean(expit(value - b)))

        lower, upper = min(a.min(), b.min()) - 40, max(a.max(), b.max()) + 40
        value, fit = brentq(score, lower, upper, xtol=tol,
                            rtol=max(tol, 4 * np.finfo(float).eps),
                            maxiter=maxiter, full_output=True, disp=False)
        if not fit.converged:
            raise RuntimeError("BAR failed to converge on adjacent interval {}".format(k))
        p, q = expit(a - value), expit(value - b)
        # expit(x)*expit(-x) remains accurate when expit(x) rounds to 1.
        derivative = float(np.mean(expit(a - value) * expit(value - a)) +
                           np.mean(expit(value - b) * expit(b - value)))
        if derivative <= np.finfo(float).tiny:
            raise ValueError("BAR has no observed overlap on interval {}; the evidence ratio is unidentified".format(k))
        ratios.append(value + center)
        influences.append((p.mean(axis=1) - q.mean(axis=1)) / derivative)
        overlaps.append(float(p.mean() + q.mean()))
        residuals.append(abs(score(value)))
        iterations.append(fit.iterations)
    return (np.array(ratios), np.sum(influences, axis=0),
            np.array(overlaps), residuals, iterations)


def get_bar(B, L, ba=None, bb=None, fixed_ladder=True, mode="obm",
            batch_size=None, spe_kernel="bartlett", allow_partial=False,
            return_diagnostics=False, tol=1e-10, maxiter=10000,
            overlap_threshold=1e-3):
    """Sum adjacent Bennett acceptance-ratio (optimal bridge) estimates.

    Each interval solves the equal-sample-size Bennett equation using
    samples from both neighbouring temperatures. See ``get_ti_corrected``
    for inputs, endpoints, ladder restrictions, and error conventions.
    This avoids TI quadrature error, but requires overlap at every bridge.

    Diagnostics report each adjacent overlap (one for identical states,
    zero for disjoint states). Below ``overlap_threshold`` a warning is
    issued and MCSE is infinity: a local delta-method error can be falsely
    tiny when tails needed to identify the ratio have not been observed.
    A numerical absence of overlap raises ValueError.
    """
    beta, L, diagnostics = _prepare(B, L, ba, bb, allow_partial)
    if tol <= 0 or maxiter < 1 or not 0 <= overlap_threshold < 1:
        raise ValueError("Require tol > 0, maxiter >= 1, and 0 <= overlap_threshold < 1")
    ratios, influence, overlap, residual, iterations = _bar_pairs(beta, L, tol, maxiter)
    error = _mcse(influence, mode, batch_size, spe_kernel)
    reliable = bool(np.all(overlap >= overlap_threshold))
    if not reliable:
        warnings.warn("BAR has poor adjacent overlap; its MCSE is unavailable (infinity). Add temperatures or collect more equilibrated samples.",
                      RuntimeWarning, stacklevel=2)
        error = np.inf
    diagnostics.update(method="bar", log_ratios=ratios.tolist(),
                       adjacent_overlap=overlap.tolist(),
                       min_overlap=float(overlap.min()),
                       score_residual=max(residual), iterations=iterations,
                       converged=True, mcse_reliable=reliable)
    return _result(ratios.sum(), error, diagnostics, return_diagnostics)


def get_mbar(B, L, ba=None, bb=None, fixed_ladder=True, mode="obm",
             batch_size=None, spe_kernel="bartlett", allow_partial=False,
             return_diagnostics=False, tol=1e-10, maxiter=10000,
             overlap_threshold=1e-3):
    """All-temperature MBAR / reverse-logistic normalising-constant fit.

    ``B`` has shape (ntemps, nsweeps), or (ntemps,) for a fixed ladder;
    ``L`` contains individual log likelihoods with shape
    (ntemps, nsweeps, nwalkers). Both ascending and descending ladders are
    accepted. At least two temperatures, three retained sweeps, and one
    walker are required. All values must be finite. Adaptation must stop
    before retaining the draws: drifting ladders are always rejected,
    including when ``fixed_ladder=False`` is supplied for compatibility.

    Absolute evidence requires a normalised prior and endpoints beta=0
    and beta=1. With ``allow_partial=True`` the result instead represents
    ``log Z(beta_max) - log Z(beta_min)``. Optional ``ba`` and ``bb`` are
    inclusive indices in the ladder sorted by *ascending* beta, so they
    select the same temperature interval for either input orientation.

    Returns ``(log_evidence, mcse)``. With ``return_diagnostics=True`` it
    returns ``(log_evidence, mcse, diagnostics)``, including normalising
    constants relative to the hottest selected state, solver convergence,
    score residual, the overlap matrix, and its spectral gap. ``tol`` is
    a numerical solver tolerance, not a statistical accuracy guarantee;
    ``maxiter`` bounds the BAR initialisation iterations and MBAR solver
    function evaluations.

    Uses every likelihood sample in every state's normalising-constant
    equation. Only numpy/scipy are needed. Adjacent BAR initialises the
    convex reverse-logistic fit, with the hottest free energy fixed to zero.

    A sweep-wise score sandwich estimate, followed by OBM/HAC, accounts
    for ensemble, cross-temperature, and serial dependence. The MBAR
    overlap spectral gap diagnoses connectedness of the sampled ladder;
    a gap below ``overlap_threshold`` gives a warning and infinite MCSE.
    Nonconvergence and numerical nonidentifiability raise explicit errors.
    This diagnostic cannot detect an unvisited mode shared by all chains.
    """
    beta, L, diagnostics = _prepare(B, L, ba, bb, allow_partial)
    try:
        tol_array = np.asarray(tol)
        if tol_array.ndim != 0 or tol_array.dtype.kind not in "fiu":
            raise ValueError
        tol = float(tol_array)
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("tol must be a finite positive scalar") from None
    try:
        if isinstance(maxiter, (bool, np.bool_)):
            raise ValueError
        maxiter = operator.index(maxiter)
        if not 1 <= maxiter <= np.iinfo(np.intc).max:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("maxiter must be a positive integer within the solver's integer range") from None
    try:
        overlap_array = np.asarray(overlap_threshold)
        if overlap_array.ndim != 0 or overlap_array.dtype.kind not in "fiu":
            raise ValueError
        overlap_threshold = float(overlap_array)
        if not np.isfinite(overlap_threshold) or not 0 <= overlap_threshold < 1:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("overlap_threshold must be a finite scalar in [0, 1)") from None
    k, t, w = L.shape
    n = t * w
    center = float(np.median(L))
    centered = L - center
    ratios, _, _, _, _ = _bar_pairs(beta, centered, tol, maxiter)
    initial = -np.cumsum(ratios)
    # Flatten time first so scores can be aggregated into synchronous sweeps.
    values = centered.transpose(1, 0, 2).reshape(-1)
    chunk_size = 20000
    cached = {}

    def evaluate(x):
        if "x" in cached and np.array_equal(x, cached["x"]):
            return cached["score"], cached["hessian"]
        f = np.concatenate(([0.0], x))
        psum = np.zeros(k)
        cross = np.zeros((k, k))
        for start in range(0, values.size, chunk_size):
            logits = values[start:start + chunk_size, None] * beta[None, :] + f
            p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
            psum += p.sum(axis=0)
            cross += p.T @ p
        score = psum / n - 1
        hessian = (np.diag(psum) - cross) / n
        if not np.all(np.isfinite(score)) or not np.all(np.isfinite(hessian)):
            raise RuntimeError("MBAR failed to converge: nonfinite score or Hessian")
        cached.update(x=np.array(x, copy=True), score=score[1:],
                      full_score=score, hessian=hessian[1:, 1:])
        return score[1:], hessian[1:, 1:]

    fit = root(lambda x: evaluate(x)[0], initial,
               jac=lambda x: evaluate(x)[1], method="hybr",
               options={"xtol": tol, "maxfev": maxiter})
    score, hessian = evaluate(fit.x)
    # All normalising-constant equations must balance, including the state
    # whose free energy is anchored. Its residual sums the other residuals.
    residual = float(np.max(np.abs(cached["full_score"])))
    # MINPACK may report lack of progress at an already converged root.
    if not np.all(np.isfinite(fit.x)) or residual > max(10 * tol, 1e-8):
        raise RuntimeError("MBAR failed to converge: residual {:.3g}; {}".format(residual, fit.message))
    f = np.concatenate(([0.0], fit.x))
    score_series = np.zeros((t, k))
    psum, cross = np.zeros(k), np.zeros((k, k))
    for start in range(0, values.size, chunk_size):
        stop = min(start + chunk_size, values.size)
        logits = values[start:stop, None] * beta[None, :] + f
        p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        psum += p.sum(axis=0)
        cross += p.T @ p
        np.add.at(score_series, np.arange(start, stop) // (k * w), p)
    score_series -= w
    overlap = cross / n
    # At the fitted root this is a symmetric stochastic overlap matrix.
    eigenvalues = np.linalg.eigvalsh(overlap)
    spectral_gap = float(max(0, 1 - eigenvalues[-2]))
    hessian_per_sweep = (np.diag(psum) - cross)[1:, 1:] / t
    if np.linalg.eigvalsh(hessian_per_sweep)[0] <= 1e-14 * w:
        raise ValueError("MBAR has insufficient overlap: free-energy differences are numerically unidentified")
    # logZ = -f[-1]; f's influence is -H^{-1} score, so signs cancel.
    contrast = np.zeros(k - 1)
    contrast[-1] = 1
    influence = score_series[:, 1:] @ np.linalg.solve(hessian_per_sweep, contrast)
    error = _mcse(influence, mode, batch_size, spe_kernel)
    reliable = bool(spectral_gap >= overlap_threshold)
    if not reliable:
        warnings.warn("MBAR has a small overlap spectral gap; its MCSE is unavailable (infinity). Add bridging temperatures or collect more equilibrated samples.",
                      RuntimeWarning, stacklevel=2)
        error = np.inf
    log_normalizers = -f + (beta - beta[0]) * center
    diagnostics.update(method="mbar", converged=True, score_residual=residual,
                       iterations=int(fit.nfev), log_normalizers=log_normalizers.tolist(),
                       solver_success=bool(fit.success), solver_status=int(fit.status),
                       solver_message=str(fit.message),
                       overlap_matrix=overlap.tolist(), overlap_spectral_gap=spectral_gap,
                       mcse_reliable=reliable)
    return _result(log_normalizers[-1], error, diagnostics, return_diagnostics)
