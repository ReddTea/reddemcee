# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT

import numpy as np

def thermodynamic_integration_classic(betas, logls):
    """Calculate thermodynamical integration according to Vousden et al """
    if betas[-1] != 0:
        betas1 = np.concatenate((betas, [0]))
        betas2 = np.concatenate((betas[::2], [0]))
        logls1 = np.concatenate((logls, [logls[-1]]))
        logls2 = np.concatenate((logls[::2], [logls[-1]]))
    else:
        betas1 = betas
        betas2 = np.concatenate((betas1[:-1:2], [0]))
        logls1 = logls
        logls2 = np.concatenate((logls1[:-1:2], [logls1[-1]]))

    logZ1 = -np.trapz(logls1, betas1)
    logZ2 = -np.trapz(logls2, betas2)

    return logZ1, np.abs(logZ1 - logZ2)


def thermodynamic_integration(betas, logls1, ngrid=10001, nsim=1000):
    """Calculates the evidence (Z) and its associated error using interpolation techniques. This method employs the PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) method to estimate log-likelihood values across a specified range of beta values, incorporating random noise to simulate variability.

    Args:
        **kwargs: Additional keyword arguments passed to the log-likelihood function.

    Returns:
        tuple: A tuple containing:
            - integral_trapz (float): The calculated evidence (Z).
            - err (float): The total error associated with the evidence.
            - err_disc (float): The discretization error.
            - err_samp (float): The sampling error.
    """
    from emcee.autocorr import integrated_time
    # Constants defined for this evaluation
    num_grid = ngrid
    num_simulations = nsim


    x = betas
    # Get LogL Means
    y = np.mean(logls1, axis=1)

    # Support Variables
    # TODO extract this method? autocorr outside
    n = np.array([len(l) for l in logls1])
    tau = np.array([integrated_time(logls1[i], c=5, tol=5, quiet=False)
                    for i in range(len(y))]).flatten()

    neff = n/tau
    stdyn = np.std(logls1, axis=1, ddof=1) / np.sqrt(neff)

    # Get Interpolated Values
    xnew, mean_interpolated, std_interpolated = interpolate(x, y, stdyn, num_grid, num_simulations)
    # Calculate Z
    integral_trapz = np.trapz(mean_interpolated, xnew)

    # Calculate Errors
    z_classic, zerr_classic = thermodynamic_integration_classic(x[::-1], y[::-1])
    err_disc = abs(z_classic - integral_trapz)

    err_samp = 1/num_grid**2 * (1/4*std_interpolated[0]**2 +
                                np.sum(std_interpolated[1:-1]**2) +
                                1/4*std_interpolated[-1]**2)
    err_samp = np.sqrt(err_samp)
    err = np.sqrt(err_disc**2+err_samp**2)

    return integral_trapz, err, err_disc, err_samp


def interpolate(x, y, stdyn, num_grid, num_simulations):
    from scipy.interpolate import PchipInterpolator
    xnew = np.linspace(min(x), max(x), num_grid)
    
    # Interpolated values storage
    interpolated_values = np.zeros((num_simulations, num_grid))

    for i in range(num_simulations):
        # Perturb data with random noise based on errors
        y_noisy = y + np.random.normal(0, stdyn)

        # Create PchipInterpolator object with noisy data
        pchip = PchipInterpolator(x, y_noisy)

        # Interpolate over fine grid and store results
        interpolated_values[i, :] = pchip(xnew)
    
    mean_interpolated = np.mean(interpolated_values, axis=0)
    std_interpolated = np.std(interpolated_values, axis=0)
    return xnew, mean_interpolated, std_interpolated



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