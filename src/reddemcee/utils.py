# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

import numpy as np

# OBM helpers

def _obm_univariate(y, batch_size=None):
    """
    Univariate OBM estimator of the long-run variance Sig for
    a stationary scalar time series {y_t}.
    Returns Sig so that Var(y) ~ Sig / nsweeps
    """
    y = np.asarray(y, dtype=float)
    T = y.size
    if T < 3:
        raise ValueError('Need at least 3 sweeps for OBM')
    b = int(np.floor(T**0.5)) if batch_size is None else int(batch_size)
    b = max(2, min(b, T - 1))
    K = T - b + 1

    mu = y.mean()
    csum = np.concatenate(([0.0], np.cumsum(y)))
    bm = (csum[b:] - csum[:-b]) / b
    diff = bm - mu
    # long-run variance estimator
    Sigma_hat = (b / K) * np.dot(diff, diff)
    return Sigma_hat


def _obm_multivariate(Y, batch_size=None):
    """
    Multivariate OBM estimator for the long-run covariance Sig
    of a vector time series Y (shape: nsweeps, ntemps).
    Returns Sig so Var(Y) ≈ Sig / nsweeps
    """
    Y = np.asarray(Y, dtype=float)
    T, p = Y.shape
    if T < 3:
        raise ValueError('Need at least 3 sweeps for OBM')
    b = int(np.floor(T**0.5)) if batch_size is None else int(batch_size)
    b = max(2, min(b, T - 1))
    K = T - b + 1

    mu_hat = Y.mean(axis=0)
    csum = np.vstack([np.zeros((1, p)), np.cumsum(Y, axis=0)])
    bm = (csum[b:] - csum[:-b]) / b
    diff = bm - mu_hat
    Sigma_hat = (b / K) * diff.T @ diff
    return Sigma_hat


# Spectral helpers
def _autocovariances(y, maxlag):
    y = np.asarray(y, dtype=float)
    T = y.size
    mu = y.mean()
    z = y - mu
    gammas = np.empty(maxlag + 1, dtype=float)
    gammas[0] = np.dot(z, z) / T
    for k in range(1, maxlag + 1):
        gammas[k] = np.dot(z[k:], z[:-k]) / T
    return gammas

def _kernel_weights(kernel, b):
    k = np.arange(1, b + 1, dtype=float)
    if kernel == "bartlett":  # Newey–West
        w = 1.0 - k / (b + 1.0)
    elif kernel == "parzen":
        x = k / (b + 0.0)
        w = np.where(
            x <= 0.5,
            1 - 6*x**2 + 6*x**3,
            2*(1 - x)**3
        )
        w[x > 1] = 0.0
    elif kernel in {"tukey", "hann", "tukey-hanning", "tukeyhanning"}:
        # Cosine (Hann) lag window: w_b = 0, w_0 = 1
        w = 0.5 * (1 + np.cos(np.pi * k / (b + 0.0)))
    else:
        raise ValueError(f"Unknown kernel '{kernel}'")
    w[w < 0] = 0.0
    return w

def _spe_univariate(y, bandwidth=None, kernel="bartlett", prewhiten=False):
    """
    Estimate long-run variance Sig = f(0) via HAC (spectral density at zero).
    Var(y) ~ Sig / T.
    - kernel: 'bartlett' (Newey–West), 'parzen', or 'tukey' (Hann).
    - bandwidth: max lag b; if None, auto b = max(2, floor(2*T**(1/3))).
    - prewhiten: simple AR(1) prewhitening
    """
    y = np.asarray(y, dtype=float)
    T = y.size
    if T < 3:
        raise ValueError("Need at least 3 observations for spectral HAC.")

    if prewhiten:
        y0 = y - y.mean()
        num = np.dot(y0[1:], y0[:-1])
        den = np.dot(y0, y0)
        rho = 0.0 if den == 0 else np.clip(num / den, -0.97, 0.97)
        v = y[1:] - rho * y[:-1]
        Sigma_v = _spe_univariate(v, bandwidth=bandwidth, kernel=kernel, prewhiten=False)
        return Sigma_v / (1.0 - rho)**2

    b = int(np.floor(2 * T**(1/3))) if bandwidth is None else int(bandwidth)
    b = max(2, min(b, T - 1))

    gam = _autocovariances(y, b)
    w = _kernel_weights(kernel, b)
    Sigma = gam[0] + 2.0 * np.dot(w, gam[1:])

    return float(max(Sigma, 0.0))

def _spe_multivariate(Y, bandwidth=None, kernel="bartlett"):
    """
    Multivariate HAC estimator of the long-run covariance
    matrix Sig for a vector time series Y (nsweeps, ntemps).

    Returns Sig so Var(Y) ~ Sig/ nsweeps
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be 2D: (nsweeps, nsteps).")
    T, p = Y.shape
    if T < 3:
        raise ValueError("Need at least 3 sweeps for spectral HAC.")

    b = int(np.floor(2 * T**(1/3))) if bandwidth is None else int(bandwidth)
    b = max(2, min(b, T - 1))

    Z = Y - Y.mean(axis=0, keepdims=True)
    Gam0 = (Z.T @ Z) / T

    
    weights = _kernel_weights(kernel, b)
    Sigma = Gam0.copy()
    for k, wk in enumerate(weights, start=1):
        Zk = Z[k: , :]
        Z0 = Z[:-k, :]
        Gamk = (Zk.T @ Z0) / T
        Sigma += wk * (Gamk + Gamk.T)

    # numerical symmetrization, better stability
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma


# general helpers

def order_by_betas(B, L):
    B_sorted = B[::-1]
    L_sorted = L[::-1, :, :]
    return B_sorted, L_sorted

def _coarse_rows(a):
    if a.size == 0:
        return a
    base = a[::2, :]
    return base if len(a) % 2 == 1 else np.vstack([base, a[-1, :]])

def _coarse_rows_drop_hot(a):
    if a.size == 0:
        return a
    base = a[::2, :]
    return base if len(a) % 2 == 1 else a[1::2, :]

def _logmeanexp(x, axis=None):
    # numerical stability
    a = np.max(x, axis=axis, keepdims=True)
    return (a + np.log(np.mean(np.exp(x - a), axis=axis, keepdims=True))).squeeze()


# actual utility
def get_act_from_obm(y, batch_size=None, ddof=1):
    Sigma = _obm_univariate(y, batch_size)
    s2 = np.var(y, ddof=ddof)
    tau = Sigma / s2
    ESS = y.size / tau
    return tau, ESS


# pchip helper
def _pchip_integral_series_cut(B, L, cut_low, cut_high):
    """
    Per-sweep PCHIP integrals using the *full ladder* but
    integrating from cut_low to cut_high.

    Returns the evidence integral.
    """
    from scipy.interpolate import PchipInterpolator
    ntemps, nsweeps = B.shape
    z = np.zeros(nsweeps, dtype=float)

    for t in range(nsweeps):
        bt = B[:, t]
        ut = L[:, t]
        # collapse duplicates
        xu, inv = np.unique(bt, return_inverse=True)
        if xu.size != bt.size:
            uu = np.zeros_like(xu)
            for k in range(xu.size):
                uu[k] = ut[inv == k].mean()
            bt, ut = xu, uu

        if bt.size < 2:
            z[t] = 0.0
            continue

        # clamp cut index to valid segment range
        #ci = min(max(cut_index, 0), bt.size - 1)
        a = max(cut_low, bt[0])
        b = min(cut_high, bt[-1])
        if a >= b:
            z[t] = 0.0
            continue

        f = PchipInterpolator(bt, ut, extrapolate=False)
        z[t] = float(f.integrate(a, b))  # positive integral
    #print(a, b)
    return z



# SS helpers
def _ss_classic(B, L, fixed_ladder):
    ntemps, nsweeps = B.shape

    # Set up for SS
    m = ntemps-1
    W_series = np.empty((nsweeps, m), dtype=float)
    # Sampling Error
    if fixed_ladder:
        delta = (B[1:, :].mean(axis=1) - B[:-1, :].mean(axis=1))  # (m,)
        for i in range(m):  # ratio i uses base temperature index i
            s = delta[i] * L[i, :, :]  # (nsweeps, nwalkers)
            a = np.max(s, axis=1, keepdims=True)  # log-sum-exp stabilizer
            W_series[:, i] = np.exp(a).ravel() * np.mean(np.exp(s - a), axis=1)
    else:
        # Use per-sweep delta beta
        for i in range(m):
            dbeta_t = B[i+1, :] - B[i, :]  # (nsweeps,)
            s = dbeta_t[:, None] * L[i, :, :]  # (nsweeps, nwalkers)
            a = np.max(s, axis=1, keepdims=True)
            W_series[:, i] = np.exp(a).ravel() * np.mean(np.exp(s - a), axis=1)

    mu_hat = W_series.mean(axis=0)
    z_hat = np.sum(np.log(mu_hat))

    # grad
    grad = 1.0 / mu_hat  # for var

    return z_hat, W_series, grad


def _ss_bridge(B, L, fixed_ladder):
    ntemps, nsweeps = B.shape

    # Set up for SS
    m = ntemps-1

    logA_series = np.empty((nsweeps, m), dtype=float)
    logC_series = np.empty((nsweeps, m), dtype=float)

    # Sampling Error
    if fixed_ladder:
        # Robust constant delta-beta (average in case of tiny jitter)
        delta = (B[1:, :].mean(axis=1) - B[:-1, :].mean(axis=1))  # (m,)
        half = 0.5 * delta
        for i in range(m):
            # per-sweep log-mean-exp across walkers delta-beta i, i+1
            sA = half[i] * L[i, :, :]  # (nsweeps, nwalkers)
            sC = -half[i] * L[i+1, :, :]
            logA_series[:, i] = _logmeanexp(sA, axis=1)
            logC_series[:, i] = _logmeanexp(sC, axis=1)
    else:
        for i in range(m):
            dbeta_t = B[i+1, :] - B[i, :]  # (nsweeps,)
            sA = (0.5 * dbeta_t)[:, None] * L[i, :, :]
            sC = (-0.5 * dbeta_t)[:, None] * L[i+1, :, :]
            logA_series[:, i] = _logmeanexp(sA, axis=1)
            logC_series[:, i] = _logmeanexp(sC, axis=1)

    # Convert to linear means per stone across sweeps
    A_series = np.exp(logA_series)  # (nsweeps, m)
    C_series = np.exp(logC_series)
    muA = A_series.mean(axis=0)  # (m,)
    muC = C_series.mean(axis=0)
    
    # Bridge estimate
    z_hat = float(np.sum(np.log(muA) - np.log(muC)))
    
    W_series = np.hstack([A_series, C_series])  # (nsweeps, 2m)
    # grad
    grad = np.concatenate([1.0 / muA, -1.0 / muC])  # for var



    return z_hat, W_series, grad






# actual TI

def get_ti(B, L, pchip=True, ba=None, bb=None, fixed_ladder=True,
           mode='obm', batch_size=None, spe_kernel='bartlett'):
    B, L = order_by_betas(B, L)
    L = L.mean(axis=2)  # shape=(ntemps,nsweeps)

    ntemps, nsweeps = B.shape
    ba, bb = ba, bb
    if ba is None:
        ba = 0
    if bb is None:
        bb = ntemps-1

    if batch_size == None:
        batch_size = int(np.sqrt(ntemps))

    # Get integration beta interval
    if pchip:
        cut_low = B[:, -1][ba]
        cut_high = B[:, -1][bb]

        z_t = _pchip_integral_series_cut(B, L, cut_low, cut_high)  # (nsweeps,)
        B2 = _coarse_rows(B)
        L2 = _coarse_rows(L)
        z2_t = _pchip_integral_series_cut(B2, L2, cut_low, cut_high)  # (nsweeps,)

    else:
        B = B[ba:bb+1, :]
        L = L[ba:bb+1, :]

        z_t = np.trapz(L, B, axis=0)    # (nsweeps,)
        B2 = _coarse_rows_drop_hot(B)
        L2 = _coarse_rows_drop_hot(L)
        z2_t = np.trapz(L2, B2, axis=0)    # (nsweeps,)


    # Evidence estimate
    z_hat = z_t.mean()

    # Discretisation error
    z2_hat  = z2_t.mean()
    err_disc = z2_hat-z_hat

    # Sampling Error
    if mode == 'obm':
        if fixed_ladder:
            Sigma_hat = _obm_univariate(z_t, batch_size=batch_size)  # long-run variance
            err_MC = np.sqrt(Sigma_hat / z_t.size)
        # multivariate
        else:
            S = 0.5 * (L[:-1, :] + L[1:, :])  # (ntemps-1, nsweeps)
            dB = B[1:, :] - B[:-1, :]  # (ntemps-1, nsweeps)
            W_series = (-dB).T  # (nsweeps, ntemps-1)
            S_series = S.T  # (nsweeps, ntemps-1)

            # mOBM on S_series to estimate Sig_S
            Sigma_S = _obm_multivariate(S_series, batch_size=batch_size)

            # Propagate with the *average* per-sweep weights (diminishing adaptation)
            vbar = W_series.mean(axis=0)
            Sigma_hat = float(vbar @ Sigma_S @ vbar)
            err_MC = np.sqrt(Sigma_hat / nsweeps)

    elif mode == 'spe':
        if fixed_ladder:
            Sigma_LR = _spe_univariate(z_t, bandwidth=None, kernel=spe_kernel, prewhiten=False)
            err_MC = np.sqrt(Sigma_LR / z_t.size)
        # multivariate
        else:
            S = 0.5 * (L[:-1, :] + L[1:, :])
            dB = B[1:, :] - B[:-1, :]
            vbar = (-dB).mean(axis=1)
            S_series = S.T

            Sigma_S = _spe_multivariate(S_series, bandwidth=None, kernel=spe_kernel)
            Sigma_z = float(vbar @ Sigma_S @ vbar)
            Sigma_z = max(Sigma_z, 0.0)
            err_MC = float(np.sqrt(Sigma_z / nsweeps))        




    # Combine with discretization error
    err_tot = np.sqrt(err_disc**2 + err_MC**2)
    return z_hat, err_tot


def get_ss(B, L, bridge=True, ba=None, bb=None, fixed_ladder=True,
           mode='obm', batch_size=None, spe_kernel='bartlett'):
    B, L = order_by_betas(B, L)
    ntemps, nsweeps = B.shape
    ba, bb = ba, bb
    if ba is None:
        ba = 0
    if bb is None:
        bb = ntemps

    B = B[ba:bb+1, :]
    L = L[ba:bb+1, :, :]

    if bridge:
        z_hat, W_series, grad = _ss_bridge(B, L, fixed_ladder)
    else:
        z_hat, W_series, grad = _ss_classic(B, L, fixed_ladder)

        
    if mode=='obm':
        Sigma_hat = _obm_multivariate(W_series, batch_size=batch_size)
    elif mode=='spe':
        Sigma_hat = _spe_multivariate(W_series, bandwidth=None, kernel="bartlett")


    cov_bar = Sigma_hat / nsweeps
    var_logZ = float(grad @ cov_bar @ grad)
    err_MC = np.sqrt(max(0.0, var_logZ))

    return z_hat, err_MC



temp_table = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                      2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                      2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                      1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                      1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                      1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                      1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                      1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                      1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                      1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                      1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                      1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                      1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                      1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                      1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                      1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                      1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                      1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                      1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                      1.26579, 1.26424, 1.26271, 1.26121,
                      1.25973])



def set_temp_ladder(ntemps, ndims, temp_table=temp_table):
    tstep = temp_table[ndims-1]
    if ndims > len(temp_table):
        tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndims)
        
    return np.geomspace(1, tstep**(1-ntemps), ntemps)